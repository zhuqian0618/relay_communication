"""PyTorch 2.12 neural prior for the measurement-budgeted CE optimizer.

Install the experiment environment from ``requirements-pytorch.txt`` before
importing this module.  The public methods intentionally match
``NeuralCodePrior`` so either backend can be passed to
``BudgetedNeuralCEOptimizer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .Neural_CE import (
    FIXED_VARIABLES,
    FREE_VARIABLES,
    STATE_COUNT,
    VARIABLE_COUNT,
    PriorPrediction,
    ProbeObservation,
    TrainingExample,
    TrajectoryState,
    _one_hot_codes,
    physics_reference_code,
    validate_joint_code,
)


HISTORY_INPUT_DIM = VARIABLE_COUNT * STATE_COUNT + 6
PROBE_INPUT_DIM = VARIABLE_COUNT * STATE_COUNT + 2


class ArcPriorNetwork(nn.Module):
    """GRU trajectory encoder, DeepSets probe encoder, and three task heads."""

    def __init__(self, hidden_dim: int = 48, probe_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.probe_dim = int(probe_dim)
        self.history_gru = nn.GRU(HISTORY_INPUT_DIM, hidden_dim, batch_first=True)
        self.probe_phi = nn.Sequential(
            nn.Linear(PROBE_INPUT_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, probe_dim),
            nn.Tanh(),
        )
        self.context_dim = hidden_dim + 2 * probe_dim + 4
        self.context_norm = nn.LayerNorm(self.context_dim)
        self.absolute_head = nn.Sequential(
            nn.Linear(self.context_dim, 128), nn.SiLU(), nn.Linear(128, VARIABLE_COUNT * STATE_COUNT)
        )
        self.transition_head = nn.Sequential(
            nn.Linear(self.context_dim, 128), nn.SiLU(), nn.Linear(128, VARIABLE_COUNT * STATE_COUNT)
        )
        candidate_dim = self.context_dim + VARIABLE_COUNT * STATE_COUNT + 11
        self.score_head = nn.Sequential(
            nn.Linear(candidate_dim, 192),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 2),
        )

    def encode(
        self,
        history: Tensor,
        history_lengths: Tensor,
        probes: Tensor,
        probe_mask: Tensor,
        current: Tensor,
    ) -> Tensor:
        batch_size = history.shape[0]
        hidden = torch.zeros(batch_size, self.hidden_dim, device=history.device, dtype=history.dtype)
        nonempty = history_lengths > 0
        if bool(nonempty.any()):
            packed = nn.utils.rnn.pack_padded_sequence(
                history[nonempty], history_lengths[nonempty].cpu(), batch_first=True, enforce_sorted=False
            )
            _, packed_hidden = self.history_gru(packed)
            hidden[nonempty] = packed_hidden[-1]

        encoded_probes = self.probe_phi(probes)
        expanded_mask = probe_mask.unsqueeze(-1)
        count = expanded_mask.sum(dim=1).clamp_min(1.0)
        probe_mean = (encoded_probes * expanded_mask).sum(dim=1) / count
        masked_for_max = encoded_probes.masked_fill(expanded_mask == 0, -1e9)
        probe_max = masked_for_max.max(dim=1).values
        no_probes = probe_mask.sum(dim=1) == 0
        probe_max[no_probes] = 0.0
        context = torch.cat((hidden, probe_mean, probe_max, current), dim=1)
        return self.context_norm(context)

    def code_logits(self, context: Tensor) -> tuple[Tensor, Tensor]:
        absolute = self.absolute_head(context).reshape(-1, VARIABLE_COUNT, STATE_COUNT)
        transition = self.transition_head(context).reshape(-1, VARIABLE_COUNT, STATE_COUNT)
        return absolute, transition

    def score(self, context: Tensor, candidate_features: Tensor) -> tuple[Tensor, Tensor]:
        if context.shape[0] == 1 and candidate_features.shape[0] != 1:
            context = context.expand(candidate_features.shape[0], -1)
        output = self.score_head(torch.cat((context, candidate_features), dim=1))
        mean = output[:, 0]
        log_variance = output[:, 1].clamp(-6.0, 5.0)
        return mean, log_variance


class TorchNeuralCodePrior:
    """Trainable five-network ensemble implementing the neural-prior API."""

    def __init__(
        self,
        ensemble_size: int = 5,
        seed: int = 20260804,
        physics_confidence: float = 0.82,
        hidden_dim: int = 48,
        probe_dim: int = 32,
        device: str | None = None,
    ) -> None:
        self.ensemble_size = int(ensemble_size)
        self.seed = int(seed)
        self.physics_confidence = float(physics_confidence)
        self.hidden_dim = int(hidden_dim)
        self.probe_dim = int(probe_dim)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models: list[ArcPriorNetwork] = []
        for index in range(self.ensemble_size):
            torch.manual_seed(self.seed + 7919 * index)
            self.models.append(ArcPriorNetwork(hidden_dim, probe_dim).to(self.device))
        self.validation_mae_dB = 1.0
        self.is_fitted = False

    def _physics_probabilities(self, angle_cmd_deg: float) -> np.ndarray:
        target = physics_reference_code(angle_cmd_deg)
        other = (1.0 - self.physics_confidence) / (STATE_COUNT - 1)
        probabilities = np.full((VARIABLE_COUNT, STATE_COUNT), other, dtype=np.float32)
        probabilities[np.arange(VARIABLE_COUNT), target] = self.physics_confidence
        probabilities[list(FIXED_VARIABLES)] = [1.0, 0.0, 0.0, 0.0]
        return probabilities

    @staticmethod
    def _history_step(summary, reference_power_dBm: float, previous_code: np.ndarray) -> np.ndarray:
        angle_rad = np.deg2rad(summary.angle_cmd_deg)
        scalars = np.asarray([
            np.sin(angle_rad),
            np.cos(angle_rad),
            np.clip(summary.delta_angle_cmd_deg / 20.0, -3.0, 3.0),
            float(summary.direction),
            np.clip((summary.best_power_dBm - reference_power_dBm) / 20.0, -3.0, 3.0),
            np.mean(summary.best_code != previous_code),
        ], dtype=np.float32)
        return np.concatenate((_one_hot_codes(summary.best_code), scalars)).astype(np.float32)

    @classmethod
    def _encode_state_numpy(
        cls,
        state: TrajectoryState,
        probes: Sequence[ProbeObservation],
        angle_cmd_deg: float,
        delta_angle_cmd_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zero_anchor = state.calibrations.get(0.0)
        reference_power = zero_anchor.power_dBm if zero_anchor is not None else state.previous_power_dBm
        previous_code = zero_anchor.code if zero_anchor is not None else physics_reference_code(0.0)
        history_steps = []
        for summary in state.history[-state.history_length:]:
            history_steps.append(cls._history_step(summary, reference_power, previous_code))
            previous_code = summary.best_code
        history = np.asarray(history_steps, dtype=np.float32)
        if not history_steps:
            history = np.zeros((0, HISTORY_INPUT_DIM), dtype=np.float32)

        repeated = [probe.power_dBm for probe in probes if probe.repeated_baseline]
        baseline = float(np.mean(repeated)) if repeated else state.previous_power_dBm
        spread = float(np.ptp(repeated)) if len(repeated) >= 2 else 0.0
        probe_steps = []
        for probe in probes:
            scalars = np.asarray([
                np.clip((probe.power_dBm - baseline) / 20.0, -3.0, 3.0),
                np.clip(spread / 5.0, 0.0, 3.0),
            ], dtype=np.float32)
            probe_steps.append(np.concatenate((_one_hot_codes(probe.code), scalars)))
        probe_array = np.asarray(probe_steps, dtype=np.float32)
        if not probe_steps:
            probe_array = np.zeros((0, PROBE_INPUT_DIM), dtype=np.float32)

        angle_rad = np.deg2rad(float(np.clip(angle_cmd_deg, -60.0, 60.0)))
        current = np.asarray([
            np.sin(angle_rad), np.cos(angle_rad),
            np.clip(delta_angle_cmd_deg / 20.0, -3.0, 3.0),
            float(np.sign(delta_angle_cmd_deg)),
        ], dtype=np.float32)
        return history, probe_array, current

    def _batch_inputs(
        self,
        examples: Sequence[tuple[TrajectoryState, Sequence[ProbeObservation], float, float]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        encoded = [self._encode_state_numpy(*example) for example in examples]
        max_history = max(1, max(item[0].shape[0] for item in encoded))
        max_probes = max(1, max(item[1].shape[0] for item in encoded))
        history = np.zeros((len(encoded), max_history, HISTORY_INPUT_DIM), dtype=np.float32)
        history_lengths = np.zeros(len(encoded), dtype=np.int64)
        probes = np.zeros((len(encoded), max_probes, PROBE_INPUT_DIM), dtype=np.float32)
        probe_mask = np.zeros((len(encoded), max_probes), dtype=np.float32)
        current = np.zeros((len(encoded), 4), dtype=np.float32)
        for index, (history_item, probe_item, current_item) in enumerate(encoded):
            history[index, : history_item.shape[0]] = history_item
            history_lengths[index] = history_item.shape[0]
            probes[index, : probe_item.shape[0]] = probe_item
            probe_mask[index, : probe_item.shape[0]] = 1.0
            current[index] = current_item
        return tuple(torch.as_tensor(value, device=self.device) for value in (
            history, history_lengths, probes, probe_mask, current
        ))

    @staticmethod
    def _map_transition_probabilities(previous_code: np.ndarray, transition: np.ndarray) -> np.ndarray:
        absolute = np.zeros_like(transition)
        for category, offset in enumerate((0, 1, -1, 2)):
            states = (previous_code + offset) % STATE_COUNT
            absolute[np.arange(VARIABLE_COUNT), states] += transition[:, category]
        return absolute

    def predict(
        self,
        state: TrajectoryState,
        probes: Sequence[ProbeObservation],
        angle_cmd_deg: float,
        delta_angle_cmd_deg: float,
    ) -> PriorPrediction:
        inputs = self._batch_inputs([(state, probes, angle_cmd_deg, delta_angle_cmd_deg)])
        base_logits = np.log(np.maximum(self._physics_probabilities(angle_cmd_deg), 1e-9))
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                context = model.encode(*inputs)
                residual, transition_logits = model.code_logits(context)
                absolute = torch.softmax(
                    torch.as_tensor(base_logits, device=self.device) + residual[0], dim=-1
                ).cpu().numpy()
                transition = torch.softmax(transition_logits[0], dim=-1).cpu().numpy()
            mapped = self._map_transition_probabilities(state.previous_code, transition)
            combined = 0.72 * absolute + 0.28 * mapped
            combined /= combined.sum(axis=1, keepdims=True)
            combined[list(FIXED_VARIABLES)] = [1.0, 0.0, 0.0, 0.0]
            predictions.append(combined)
        member_array = np.asarray(predictions)
        return PriorPrediction(
            probabilities=member_array.mean(axis=0),
            variable_uncertainty=member_array.var(axis=0).sum(axis=1),
            member_probabilities=member_array,
        )

    @staticmethod
    def _candidate_scalar_features(
        candidates: np.ndarray,
        probabilities: np.ndarray,
        previous_code: np.ndarray,
        angle_cmd_deg: float,
    ) -> np.ndarray:
        physics = physics_reference_code(angle_cmd_deg)
        selected = probabilities[np.arange(VARIABLE_COUNT)[None, :], candidates]
        delta = (candidates - previous_code[None, :]) % STATE_COUNT
        return np.column_stack((
            np.log(np.maximum(selected, 1e-9)).mean(axis=1),
            np.mean(candidates != previous_code[None, :], axis=1),
            np.mean(candidates != physics[None, :], axis=1),
            *(np.mean(delta == category, axis=1) for category in (0, 1, 3, 2)),
            np.mean(candidates[:, : VARIABLE_COUNT // 2] != physics[None, : VARIABLE_COUNT // 2], axis=1),
            np.mean(candidates[:, VARIABLE_COUNT // 2 :] != physics[None, VARIABLE_COUNT // 2 :], axis=1),
            np.mean(np.sin(np.pi * candidates / 2.0), axis=1),
            np.mean(np.cos(np.pi * candidates / 2.0), axis=1),
        )).astype(np.float32)

    def _candidate_features(
        self,
        candidates: np.ndarray,
        probabilities: np.ndarray,
        previous_code: np.ndarray,
        angle_cmd_deg: float,
    ) -> Tensor:
        one_hot = _one_hot_codes(candidates).astype(np.float32)
        scalars = self._candidate_scalar_features(candidates, probabilities, previous_code, angle_cmd_deg)
        return torch.as_tensor(np.column_stack((one_hot, scalars)), device=self.device)

    def score_candidates(
        self,
        state: TrajectoryState,
        probes: Sequence[ProbeObservation],
        candidates: np.ndarray,
        angle_cmd_deg: float,
        delta_angle_cmd_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray([validate_joint_code(code) for code in candidates])
        prediction = self.predict(state, probes, angle_cmd_deg, delta_angle_cmd_deg)
        candidate_features = self._candidate_features(
            candidates, prediction.probabilities, state.previous_code, angle_cmd_deg
        )
        inputs = self._batch_inputs([(state, probes, angle_cmd_deg, delta_angle_cmd_deg)])
        means, variances = [], []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                context = model.encode(*inputs)
                mean, log_variance = model.score(context, candidate_features)
            means.append(mean.cpu().numpy())
            variances.append(np.exp(log_variance.cpu().numpy()))
        means_array = np.asarray(means)
        total_variance = means_array.var(axis=0) + np.asarray(variances).mean(axis=0)
        return means_array.mean(axis=0), np.sqrt(np.maximum(total_variance, 0.0))

    @staticmethod
    def _transition_targets(previous: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = (target - previous) % STATE_COUNT
        result = np.full_like(delta, 3)
        result[delta == 0] = 0
        result[delta == 1] = 1
        result[delta == 3] = 2
        return result

    def fit(
        self,
        examples: Iterable[TrainingExample],
        epochs: int = 120,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
    ) -> dict[str, float]:
        examples = list(examples)
        if not examples:
            raise ValueError("at least one training example is required")
        rng = np.random.default_rng(self.seed)
        final_losses, final_maes = [], []

        for member_index, model in enumerate(self.models):
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            bootstrap = rng.choice(len(examples), size=len(examples), replace=True)
            member_examples = [examples[index] for index in bootstrap]
            inputs = self._batch_inputs([
                (example.state, example.probes, example.angle_cmd_deg, example.delta_angle_cmd_deg)
                for example in member_examples
            ])
            targets_np = np.asarray([example.target_code for example in member_examples])
            previous_np = np.asarray([example.state.previous_code for example in member_examples])
            targets = torch.as_tensor(targets_np, dtype=torch.long, device=self.device)
            transition_targets = torch.as_tensor(
                np.asarray([self._transition_targets(old, new) for old, new in zip(previous_np, targets_np)]),
                dtype=torch.long,
                device=self.device,
            )
            physics_logits = torch.as_tensor(np.asarray([
                np.log(np.maximum(self._physics_probabilities(example.angle_cmd_deg), 1e-9))
                for example in member_examples
            ]), device=self.device)

            for _ in range(int(epochs)):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                contexts = model.encode(*inputs)
                absolute_residual, transition_logits = model.code_logits(contexts)
                absolute_logits = physics_logits + absolute_residual
                absolute_loss = F.cross_entropy(
                    absolute_logits[:, FREE_VARIABLES].reshape(-1, STATE_COUNT),
                    targets[:, FREE_VARIABLES].reshape(-1),
                )
                transition_loss = F.cross_entropy(
                    transition_logits[:, FREE_VARIABLES].reshape(-1, STATE_COUNT),
                    transition_targets[:, FREE_VARIABLES].reshape(-1),
                )
                sparse_loss = torch.softmax(transition_logits[:, FREE_VARIABLES], dim=-1)[..., 1:].sum(dim=-1).mean()

                score_losses, rank_losses = [], []
                for example_index, example in enumerate(member_examples):
                    if example.candidate_codes is None or example.candidate_relative_powers_dB is None:
                        continue
                    prior_np = torch.softmax(absolute_logits[example_index], dim=-1).detach().cpu().numpy()
                    candidate_features = self._candidate_features(
                        example.candidate_codes, prior_np, example.state.previous_code, example.angle_cmd_deg
                    )
                    mean, log_variance = model.score(contexts[example_index : example_index + 1], candidate_features)
                    power_target = torch.as_tensor(
                        example.candidate_relative_powers_dB, dtype=torch.float32, device=self.device
                    )
                    robust = F.smooth_l1_loss(mean, power_target)
                    gaussian = 0.5 * (torch.exp(-log_variance) * (mean - power_target).square() + log_variance).mean()
                    score_losses.append(robust + 0.05 * gaussian)
                    high = int(torch.argmax(power_target))
                    low = int(torch.argmin(power_target))
                    rank_losses.append(F.softplus(-(mean[high] - mean[low])))
                score_loss = torch.stack(score_losses).mean() if score_losses else absolute_loss.new_zeros(())
                rank_loss = torch.stack(rank_losses).mean() if rank_losses else absolute_loss.new_zeros(())
                loss = absolute_loss + 0.4 * transition_loss + 0.5 * score_loss + 0.15 * rank_loss + 0.01 * sparse_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            final_losses.append(float(loss.detach().cpu()))
            errors = []
            model.eval()
            with torch.no_grad():
                contexts = model.encode(*inputs)
                absolute_residual, _ = model.code_logits(contexts)
                absolute_logits = physics_logits + absolute_residual
                for example_index, example in enumerate(member_examples):
                    if example.candidate_codes is None or example.candidate_relative_powers_dB is None:
                        continue
                    prior_np = torch.softmax(absolute_logits[example_index], dim=-1).cpu().numpy()
                    features = self._candidate_features(
                        example.candidate_codes, prior_np, example.state.previous_code, example.angle_cmd_deg
                    )
                    mean, _ = model.score(contexts[example_index : example_index + 1], features)
                    errors.extend(np.abs(mean.cpu().numpy() - example.candidate_relative_powers_dB).tolist())
            if errors:
                final_maes.append(float(np.median(errors)))

        self.validation_mae_dB = float(np.median(final_maes)) if final_maes else 1.0
        self.is_fitted = True
        return {
            "loss": float(np.mean(final_losses)),
            "score_mae_dB": self.validation_mae_dB,
            "training_examples": float(len(examples)),
        }

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "config": {
                "ensemble_size": self.ensemble_size,
                "seed": self.seed,
                "physics_confidence": self.physics_confidence,
                "hidden_dim": self.hidden_dim,
                "probe_dim": self.probe_dim,
                "validation_mae_dB": self.validation_mae_dB,
                "is_fitted": self.is_fitted,
            },
            "model_states": [model.state_dict() for model in self.models],
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "TorchNeuralCodePrior":
        checkpoint = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
        config = checkpoint["config"]
        model = cls(
            ensemble_size=config["ensemble_size"],
            seed=config["seed"],
            physics_confidence=config["physics_confidence"],
            hidden_dim=config["hidden_dim"],
            probe_dim=config["probe_dim"],
            device=device,
        )
        for network, state_dict in zip(model.models, checkpoint["model_states"]):
            network.load_state_dict(state_dict)
        model.validation_mae_dB = float(config["validation_mae_dB"])
        model.is_fitted = bool(config["is_fitted"])
        return model
