"""Neural-prior, measurement-budgeted CE optimization for the UAV arc experiment.

The implementation deliberately depends only on NumPy.  The neural prior uses a
small fixed-feature GRU/DeepSets encoder and trainable output heads, so the code
can run on the experiment computer without PyTorch.  A future PyTorch model can
replace :class:`NeuralCodePrior` without changing the optimizer or measurement
interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .MS_Configuration import Columns, calculate_2bit_compensation_code


STATE_COUNT = 4
VARIABLE_COUNT = 2 * Columns
FIXED_VARIABLES = (0, Columns)
FREE_VARIABLES = np.asarray(
    [index for index in range(VARIABLE_COUNT) if index not in FIXED_VARIABLES], dtype=int
)
MeasureFunction = Callable[[np.ndarray], float]


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponential = np.exp(np.clip(shifted, -60.0, 60.0))
    return exponential / exponential.sum(axis=axis, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _one_hot_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=int)
    return np.eye(STATE_COUNT, dtype=float)[codes].reshape(*codes.shape[:-1], -1)


def validate_joint_code(code: np.ndarray) -> np.ndarray:
    """Return a validated 32-state joint code with the two references fixed."""

    result = np.asarray(code, dtype=int).reshape(-1).copy()
    if result.size != VARIABLE_COUNT:
        raise ValueError(f"joint code must contain {VARIABLE_COUNT} states")
    if np.any((result < 0) | (result >= STATE_COUNT)):
        raise ValueError("all code states must be integers in [0, 3]")
    result[list(FIXED_VARIABLES)] = 0
    return result


def physics_reference_code(angle_deg: float) -> np.ndarray:
    """Return the ideal-model joint 2-bit code at a commanded arc angle."""

    clipped_angle = float(np.clip(angle_deg, -60.0, 60.0))
    single_ms = calculate_2bit_compensation_code(clipped_angle)[2]
    return validate_joint_code(np.concatenate((single_ms, single_ms)))


@dataclass
class ProbeObservation:
    code: np.ndarray
    power_dBm: float
    repeated_baseline: bool = False

    def __post_init__(self) -> None:
        self.code = validate_joint_code(self.code)
        self.power_dBm = float(self.power_dBm)


@dataclass
class PositionSummary:
    angle_cmd_deg: float
    delta_angle_cmd_deg: float
    direction: int
    best_code: np.ndarray
    best_power_dBm: float
    measurement_count: int

    def __post_init__(self) -> None:
        self.angle_cmd_deg = float(np.clip(self.angle_cmd_deg, -60.0, 60.0))
        self.delta_angle_cmd_deg = float(self.delta_angle_cmd_deg)
        self.direction = int(np.sign(self.direction))
        self.best_code = validate_joint_code(self.best_code)
        self.best_power_dBm = float(self.best_power_dBm)
        self.measurement_count = int(self.measurement_count)


@dataclass
class CalibrationRecord:
    angle_deg: float
    code: np.ndarray
    power_dBm: float
    measurement_count: int = 0
    noise_span_dB: float = 0.0

    def __post_init__(self) -> None:
        self.angle_deg = float(self.angle_deg)
        self.code = validate_joint_code(self.code)
        self.power_dBm = float(self.power_dBm)
        self.noise_span_dB = float(self.noise_span_dB)


@dataclass
class TrajectoryState:
    """History and calibration anchors for one physical experiment session."""

    angle_cmd_deg: float = 0.0
    history_length: int = 5
    history: list[PositionSummary] = field(default_factory=list)
    calibrations: dict[float, CalibrationRecord] = field(default_factory=dict)

    @property
    def previous_code(self) -> np.ndarray:
        if self.history:
            return self.history[-1].best_code.copy()
        if 0.0 in self.calibrations:
            return self.calibrations[0.0].code.copy()
        return physics_reference_code(0.0)

    @property
    def previous_power_dBm(self) -> float:
        if self.history:
            return float(self.history[-1].best_power_dBm)
        if 0.0 in self.calibrations:
            return float(self.calibrations[0.0].power_dBm)
        return -100.0

    def add_calibration(self, record: CalibrationRecord) -> None:
        key = float(np.round(record.angle_deg, 6))
        self.calibrations[key] = record
        if np.isclose(record.angle_deg, 0.0):
            self.angle_cmd_deg = 0.0

    def record_position(self, summary: PositionSummary) -> None:
        self.angle_cmd_deg = summary.angle_cmd_deg
        self.history.append(summary)
        if len(self.history) > self.history_length:
            del self.history[:-self.history_length]


@dataclass
class TrainingExample:
    state: TrajectoryState
    angle_cmd_deg: float
    delta_angle_cmd_deg: float
    target_code: np.ndarray
    probes: list[ProbeObservation] = field(default_factory=list)
    candidate_codes: np.ndarray | None = None
    candidate_relative_powers_dB: np.ndarray | None = None
    trajectory_id: int = 0

    def __post_init__(self) -> None:
        self.target_code = validate_joint_code(self.target_code)
        if self.candidate_codes is not None:
            candidate_codes = np.asarray(self.candidate_codes, dtype=int)
            if candidate_codes.ndim != 2 or candidate_codes.shape[1] != VARIABLE_COUNT:
                raise ValueError("candidate_codes must have shape (candidate_count, 32)")
            self.candidate_codes = np.asarray(
                [validate_joint_code(code) for code in candidate_codes], dtype=int
            )
        if self.candidate_relative_powers_dB is not None:
            self.candidate_relative_powers_dB = np.asarray(
                self.candidate_relative_powers_dB, dtype=float
            ).reshape(-1)
            if self.candidate_codes is None or self.candidate_codes.shape[0] != self.candidate_relative_powers_dB.size:
                raise ValueError("candidate codes and power labels must have equal lengths")


@dataclass
class PriorPrediction:
    probabilities: np.ndarray
    variable_uncertainty: np.ndarray
    member_probabilities: np.ndarray


class _EnsembleMember:
    """One randomized GRU/DeepSets feature encoder with trainable heads."""

    history_input_dim = VARIABLE_COUNT * STATE_COUNT + 6
    probe_input_dim = VARIABLE_COUNT * STATE_COUNT + 2

    def __init__(self, seed: int, hidden_dim: int = 32, probe_dim: int = 24) -> None:
        self.seed = int(seed)
        self.hidden_dim = int(hidden_dim)
        self.probe_dim = int(probe_dim)
        rng = np.random.default_rng(seed)

        def random_matrix(rows: int, columns: int) -> np.ndarray:
            return rng.normal(scale=1.0 / np.sqrt(max(rows, 1)), size=(rows, columns))

        self.Wz = random_matrix(self.history_input_dim, hidden_dim)
        self.Uz = random_matrix(hidden_dim, hidden_dim)
        self.bz = rng.normal(scale=0.03, size=hidden_dim)
        self.Wr = random_matrix(self.history_input_dim, hidden_dim)
        self.Ur = random_matrix(hidden_dim, hidden_dim)
        self.br = rng.normal(scale=0.03, size=hidden_dim)
        self.Wh = random_matrix(self.history_input_dim, hidden_dim)
        self.Uh = random_matrix(hidden_dim, hidden_dim)
        self.bh = rng.normal(scale=0.03, size=hidden_dim)
        self.probe_W = random_matrix(self.probe_input_dim, probe_dim)
        self.probe_b = rng.normal(scale=0.03, size=probe_dim)

        self.context_dim = hidden_dim + 2 * probe_dim + 4
        self.absolute_W = np.zeros((self.context_dim, VARIABLE_COUNT * STATE_COUNT))
        self.transition_W = np.zeros_like(self.absolute_W)
        self.candidate_projection = random_matrix(VARIABLE_COUNT * STATE_COUNT, 16)
        self.score_feature_dim = self.context_dim + 11 + 32 + 1
        self.score_W = np.zeros(self.score_feature_dim)

    def history_step(self, summary: PositionSummary, reference_power_dBm: float, previous_code: np.ndarray) -> np.ndarray:
        angle_rad = np.deg2rad(summary.angle_cmd_deg)
        hamming = np.mean(summary.best_code != previous_code)
        scalars = np.asarray([
            np.sin(angle_rad),
            np.cos(angle_rad),
            np.clip(summary.delta_angle_cmd_deg / 20.0, -3.0, 3.0),
            float(summary.direction),
            np.clip((summary.best_power_dBm - reference_power_dBm) / 20.0, -3.0, 3.0),
            hamming,
        ])
        return np.concatenate((_one_hot_codes(summary.best_code), scalars))

    def encode_context(
        self,
        state: TrajectoryState,
        probes: Sequence[ProbeObservation],
        angle_cmd_deg: float,
        delta_angle_cmd_deg: float,
    ) -> np.ndarray:
        hidden = np.zeros(self.hidden_dim)
        reference_power = state.calibrations.get(
            0.0, CalibrationRecord(0.0, physics_reference_code(0.0), state.previous_power_dBm)
        ).power_dBm
        previous_code = state.calibrations.get(
            0.0, CalibrationRecord(0.0, physics_reference_code(0.0), reference_power)
        ).code
        for summary in state.history[-state.history_length:]:
            step = self.history_step(summary, reference_power, previous_code)
            update = _sigmoid(step @ self.Wz + hidden @ self.Uz + self.bz)
            reset = _sigmoid(step @ self.Wr + hidden @ self.Ur + self.br)
            proposal = np.tanh(step @ self.Wh + (reset * hidden) @ self.Uh + self.bh)
            hidden = (1.0 - update) * hidden + update * proposal
            previous_code = summary.best_code

        if probes:
            baseline_power = float(np.mean([probe.power_dBm for probe in probes if probe.repeated_baseline])) \
                if any(probe.repeated_baseline for probe in probes) else state.previous_power_dBm
            duplicate_values = [probe.power_dBm for probe in probes if probe.repeated_baseline]
            duplicate_spread = float(np.ptp(duplicate_values)) if len(duplicate_values) >= 2 else 0.0
            encoded_probes = []
            for probe in probes:
                scalars = np.asarray([
                    np.clip((probe.power_dBm - baseline_power) / 20.0, -3.0, 3.0),
                    np.clip(duplicate_spread / 5.0, 0.0, 3.0),
                ])
                probe_input = np.concatenate((_one_hot_codes(probe.code), scalars))
                encoded_probes.append(np.tanh(probe_input @ self.probe_W + self.probe_b))
            probe_array = np.asarray(encoded_probes)
            probe_features = np.concatenate((probe_array.mean(axis=0), probe_array.max(axis=0)))
        else:
            probe_features = np.zeros(2 * self.probe_dim)

        angle_rad = np.deg2rad(float(np.clip(angle_cmd_deg, -60.0, 60.0)))
        current = np.asarray([
            np.sin(angle_rad),
            np.cos(angle_rad),
            np.clip(delta_angle_cmd_deg / 20.0, -3.0, 3.0),
            float(np.sign(delta_angle_cmd_deg)),
        ])
        return np.concatenate((hidden, probe_features, current))


class NeuralCodePrior:
    """Five-member neural residual prior for the joint two-metasurface code."""

    def __init__(
        self,
        ensemble_size: int = 5,
        seed: int = 20260804,
        physics_confidence: float = 0.82,
    ) -> None:
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")
        self.ensemble_size = int(ensemble_size)
        self.seed = int(seed)
        self.physics_confidence = float(physics_confidence)
        self.members = [_EnsembleMember(seed + 7919 * index) for index in range(ensemble_size)]
        self.validation_mae_dB = 1.0
        self.is_fitted = False

    def _physics_probabilities(self, angle_cmd_deg: float) -> np.ndarray:
        target = physics_reference_code(angle_cmd_deg)
        other = (1.0 - self.physics_confidence) / (STATE_COUNT - 1)
        probabilities = np.full((VARIABLE_COUNT, STATE_COUNT), other)
        probabilities[np.arange(VARIABLE_COUNT), target] = self.physics_confidence
        probabilities[list(FIXED_VARIABLES)] = [1.0, 0.0, 0.0, 0.0]
        return probabilities

    @staticmethod
    def _transition_labels(previous_code: np.ndarray, target_code: np.ndarray) -> np.ndarray:
        delta = (target_code - previous_code) % STATE_COUNT
        labels = np.full(VARIABLE_COUNT, 3, dtype=int)
        labels[delta == 0] = 0
        labels[delta == 1] = 1
        labels[delta == 3] = 2
        return labels

    @staticmethod
    def _map_transition_probabilities(previous_code: np.ndarray, transition: np.ndarray) -> np.ndarray:
        absolute = np.zeros_like(transition)
        offsets = np.asarray([0, 1, -1, 2])
        for category, offset in enumerate(offsets):
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
        physics = self._physics_probabilities(angle_cmd_deg)
        base_logits = np.log(np.maximum(physics, 1e-9))
        member_predictions = []
        for member in self.members:
            context = member.encode_context(state, probes, angle_cmd_deg, delta_angle_cmd_deg)
            absolute_residual = (context @ member.absolute_W).reshape(VARIABLE_COUNT, STATE_COUNT)
            absolute = _softmax(base_logits + absolute_residual)
            transition = _softmax((context @ member.transition_W).reshape(VARIABLE_COUNT, STATE_COUNT))
            mapped_transition = self._map_transition_probabilities(state.previous_code, transition)
            combined = 0.72 * absolute + 0.28 * mapped_transition
            combined /= combined.sum(axis=1, keepdims=True)
            combined[list(FIXED_VARIABLES)] = [1.0, 0.0, 0.0, 0.0]
            member_predictions.append(combined)
        member_array = np.asarray(member_predictions)
        mean = member_array.mean(axis=0)
        uncertainty = member_array.var(axis=0).sum(axis=1)
        return PriorPrediction(mean, uncertainty, member_array)

    def _candidate_features(
        self,
        member: _EnsembleMember,
        context: np.ndarray,
        candidates: np.ndarray,
        probabilities: np.ndarray,
        previous_code: np.ndarray,
        angle_cmd_deg: float,
    ) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=int)
        physics_code = physics_reference_code(angle_cmd_deg)
        chosen_probability = probabilities[np.arange(VARIABLE_COUNT)[None, :], candidates]
        mean_log_probability = np.log(np.maximum(chosen_probability, 1e-9)).mean(axis=1)
        delta = (candidates - previous_code[None, :]) % STATE_COUNT
        transition_fractions = np.column_stack([
            np.mean(delta == 0, axis=1),
            np.mean(delta == 1, axis=1),
            np.mean(delta == 3, axis=1),
            np.mean(delta == 2, axis=1),
        ])
        scalar_features = np.column_stack([
            mean_log_probability,
            np.mean(candidates != previous_code[None, :], axis=1),
            np.mean(candidates != physics_code[None, :], axis=1),
            transition_fractions,
            np.mean(candidates[:, :Columns] != physics_code[None, :Columns], axis=1),
            np.mean(candidates[:, Columns:] != physics_code[None, Columns:], axis=1),
            np.mean(np.sin(np.pi * candidates / 2.0), axis=1),
            np.mean(np.cos(np.pi * candidates / 2.0), axis=1),
        ])
        projected = np.tanh(_one_hot_codes(candidates) @ member.candidate_projection)
        interactions = projected * context[:16][None, :]
        repeated_context = np.repeat(context[None, :], candidates.shape[0], axis=0)
        return np.column_stack((repeated_context, scalar_features, projected, interactions, np.ones(candidates.shape[0])))

    def score_candidates(
        self,
        state: TrajectoryState,
        probes: Sequence[ProbeObservation],
        candidates: np.ndarray,
        angle_cmd_deg: float,
        delta_angle_cmd_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray([validate_joint_code(code) for code in candidates])
        prior = self.predict(state, probes, angle_cmd_deg, delta_angle_cmd_deg).probabilities
        member_scores = []
        for member in self.members:
            context = member.encode_context(state, probes, angle_cmd_deg, delta_angle_cmd_deg)
            features = self._candidate_features(
                member, context, candidates, prior, state.previous_code, angle_cmd_deg
            )
            heuristic = 8.0 * features[:, member.context_dim]
            member_scores.append(heuristic + features @ member.score_W)
        member_scores = np.asarray(member_scores)
        return member_scores.mean(axis=0), member_scores.std(axis=0)

    def fit(
        self,
        examples: Iterable[TrainingExample],
        epochs: int = 80,
        learning_rate: float = 0.04,
        l2: float = 2e-4,
    ) -> dict[str, float]:
        """Fit residual classification and robust ranking/regression heads."""

        examples = list(examples)
        if not examples:
            raise ValueError("at least one training example is required")
        losses = []
        score_errors = []
        rng = np.random.default_rng(self.seed + 17)

        for member_index, member in enumerate(self.members):
            contexts = np.asarray([
                member.encode_context(example.state, example.probes, example.angle_cmd_deg, example.delta_angle_cmd_deg)
                for example in examples
            ])
            targets = np.asarray([example.target_code for example in examples])
            previous = np.asarray([example.state.previous_code for example in examples])
            transitions = np.asarray([
                self._transition_labels(previous_code, target)
                for previous_code, target in zip(previous, targets)
            ])
            base_logits = np.asarray([
                np.log(np.maximum(self._physics_probabilities(example.angle_cmd_deg), 1e-9))
                for example in examples
            ])

            for _ in range(int(epochs)):
                residual = (contexts @ member.absolute_W).reshape(-1, VARIABLE_COUNT, STATE_COUNT)
                absolute_probability = _softmax(base_logits + residual)
                absolute_gradient = absolute_probability
                rows = np.arange(len(examples))[:, None]
                variables = np.arange(VARIABLE_COUNT)[None, :]
                absolute_gradient[rows, variables, targets] -= 1.0
                absolute_gradient[:, list(FIXED_VARIABLES)] = 0.0
                gradient_W = contexts.T @ absolute_gradient.reshape(len(examples), -1) / len(examples)
                member.absolute_W -= learning_rate * (gradient_W + l2 * member.absolute_W)

                transition_logits = (contexts @ member.transition_W).reshape(-1, VARIABLE_COUNT, STATE_COUNT)
                transition_probability = _softmax(transition_logits)
                transition_gradient = transition_probability
                transition_gradient[rows, variables, transitions] -= 1.0
                transition_gradient[:, list(FIXED_VARIABLES)] = 0.0
                gradient_transition = contexts.T @ transition_gradient.reshape(len(examples), -1) / len(examples)
                member.transition_W -= learning_rate * (gradient_transition + l2 * member.transition_W)

            final_probability = _softmax(
                base_logits + (contexts @ member.absolute_W).reshape(-1, VARIABLE_COUNT, STATE_COUNT)
            )
            selected = final_probability[rows, variables, targets]
            losses.append(float(-np.log(np.maximum(selected[:, FREE_VARIABLES], 1e-9)).mean()))

            score_features: list[np.ndarray] = []
            score_targets: list[np.ndarray] = []
            groups: list[slice] = []
            offset = 0
            for example, context in zip(examples, contexts):
                if example.candidate_codes is None or example.candidate_relative_powers_dB is None:
                    continue
                probability = self.predict(
                    example.state, example.probes, example.angle_cmd_deg, example.delta_angle_cmd_deg
                ).probabilities
                features = self._candidate_features(
                    member, context, example.candidate_codes, probability,
                    example.state.previous_code, example.angle_cmd_deg,
                )
                heuristic = 8.0 * features[:, member.context_dim]
                score_features.append(features)
                score_targets.append(example.candidate_relative_powers_dB - heuristic)
                groups.append(slice(offset, offset + features.shape[0]))
                offset += features.shape[0]
            if score_features:
                X = np.vstack(score_features)
                y = np.concatenate(score_targets)
                for _ in range(max(20, int(epochs))):
                    prediction = X @ member.score_W
                    residual = prediction - y
                    huber_gradient = np.where(np.abs(residual) <= 2.0, residual, 2.0 * np.sign(residual))
                    gradient = X.T @ huber_gradient / X.shape[0]
                    for group in groups:
                        indices = np.arange(group.start, group.stop)
                        if indices.size < 2:
                            continue
                        high = indices[int(np.argmax(y[indices]))]
                        low = indices[int(np.argmin(y[indices]))]
                        difference = X[high] - X[low]
                        margin = float(difference @ member.score_W)
                        gradient += -0.05 * _sigmoid(np.asarray([-margin]))[0] * difference
                    member.score_W -= 0.01 * (gradient + l2 * member.score_W)
                score_errors.extend(np.abs((X @ member.score_W) - y).tolist())

        self.validation_mae_dB = float(np.median(score_errors)) if score_errors else 1.0
        self.is_fitted = True
        return {
            "classification_loss": float(np.mean(losses)),
            "score_mae_dB": self.validation_mae_dB,
            "training_examples": float(len(examples)),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload: dict[str, np.ndarray] = {
            "ensemble_size": np.asarray(self.ensemble_size),
            "seed": np.asarray(self.seed),
            "physics_confidence": np.asarray(self.physics_confidence),
            "validation_mae_dB": np.asarray(self.validation_mae_dB),
            "is_fitted": np.asarray(int(self.is_fitted)),
        }
        for index, member in enumerate(self.members):
            payload[f"absolute_W_{index}"] = member.absolute_W
            payload[f"transition_W_{index}"] = member.transition_W
            payload[f"score_W_{index}"] = member.score_W
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralCodePrior":
        with np.load(Path(path), allow_pickle=False) as payload:
            model = cls(
                ensemble_size=int(payload["ensemble_size"]),
                seed=int(payload["seed"]),
                physics_confidence=float(payload["physics_confidence"]),
            )
            model.validation_mae_dB = float(payload["validation_mae_dB"])
            model.is_fitted = bool(int(payload["is_fitted"]))
            for index, member in enumerate(model.members):
                member.absolute_W = payload[f"absolute_W_{index}"].copy()
                member.transition_W = payload[f"transition_W_{index}"].copy()
                member.score_W = payload[f"score_W_{index}"].copy()
        return model


@dataclass
class BudgetedCEConfig:
    initial_probe_count: int = 8
    virtual_population: int = 256
    measured_per_generation: int = 12
    max_generations: int = 3
    elite_fraction: float = 0.20
    smoothing: float = 0.60
    minimum_probability: float = 0.01
    confidence_threshold: float = 0.90
    uncertainty_weight: float = 0.75
    prediction_error_floor_dB: float = 3.0
    minimum_local_hamming: int = 2
    maximum_local_hamming: int = 6

    @property
    def maximum_measurements(self) -> int:
        return self.initial_probe_count + self.max_generations * self.measured_per_generation


@dataclass
class OptimizationResult:
    angle_cmd_deg: float
    best_code: np.ndarray
    best_power_dBm: float
    measurement_count: int
    generations: int
    confidence: float
    prediction_mae_dB: float
    noise_span_dB: float
    calibration_required: bool
    stop_reason: str
    observations: list[ProbeObservation]


@dataclass
class CalibrationCheck:
    mean_power_dBm: float
    power_drop_dB: float
    repeated_span_dB: float
    code_hamming_fraction: float
    command_angle_error_deg: float
    calibration_required: bool


class BudgetedNeuralCEOptimizer:
    """20/32/44-measurement CE optimizer guided by a neural code prior."""

    def __init__(
        self,
        prior: NeuralCodePrior,
        config: BudgetedCEConfig | None = None,
        seed: int = 20260804,
    ) -> None:
        self.prior = prior
        self.config = config or BudgetedCEConfig()
        if self.config.maximum_measurements != 44:
            raise ValueError("default experiment contract requires a 44-measurement maximum")
        self.rng = np.random.default_rng(seed)

    def _candidate_from_ranked_alternatives(
        self,
        base: np.ndarray,
        probabilities: np.ndarray,
        ranked_variables: np.ndarray,
        change_count: int,
        variant: int,
    ) -> np.ndarray:
        candidate = base.copy()
        variables = ranked_variables[:change_count]
        for offset, variable in enumerate(variables):
            order = np.argsort(probabilities[variable])[::-1]
            alternatives = order[order != candidate[variable]]
            candidate[variable] = alternatives[(variant + offset) % alternatives.size]
        return validate_joint_code(candidate)

    def _initial_candidates(
        self,
        state: TrajectoryState,
        prediction: PriorPrediction,
    ) -> list[np.ndarray]:
        probabilities = prediction.probabilities
        base = state.previous_code
        best = validate_joint_code(np.argmax(probabilities, axis=1))
        margins = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
        ranked = FREE_VARIABLES[np.argsort(margins[FREE_VARIABLES])]
        candidates: list[np.ndarray] = []

        def add(candidate: np.ndarray) -> None:
            candidate = validate_joint_code(candidate)
            if not np.array_equal(candidate, base) and not any(np.array_equal(candidate, old) for old in candidates):
                candidates.append(candidate)

        add(best)
        second = best.copy()
        variable = int(ranked[0])
        order = np.argsort(probabilities[variable])[::-1]
        second[variable] = int(order[1])
        add(second)
        for index, change_count in enumerate((2, 3, 4, 6, 5, 2, 3, 4)):
            add(self._candidate_from_ranked_alternatives(base, probabilities, np.roll(ranked, index), change_count, index))
            if len(candidates) >= self.config.initial_probe_count - 2:
                break
        while len(candidates) < self.config.initial_probe_count - 2:
            candidate = base.copy()
            variables = self.rng.choice(
                FREE_VARIABLES,
                size=self.rng.integers(self.config.minimum_local_hamming, self.config.maximum_local_hamming + 1),
                replace=False,
            )
            for variable in variables:
                candidate[variable] = self.rng.choice(STATE_COUNT, p=probabilities[variable])
                if candidate[variable] == base[variable]:
                    candidate[variable] = (candidate[variable] + 1) % STATE_COUNT
            add(candidate)
        return candidates[: self.config.initial_probe_count - 2]

    def _sample_virtual_population(self, probability: np.ndarray) -> np.ndarray:
        samples = np.empty((self.config.virtual_population, VARIABLE_COUNT), dtype=int)
        for variable in range(VARIABLE_COUNT):
            samples[:, variable] = self.rng.choice(
                STATE_COUNT, self.config.virtual_population, p=probability[variable]
            )
        samples[:, list(FIXED_VARIABLES)] = 0
        return samples

    @staticmethod
    def _unique_rows(candidates: np.ndarray) -> np.ndarray:
        _, indices = np.unique(candidates, axis=0, return_index=True)
        return candidates[np.sort(indices)]

    def _select_diverse_candidates(
        self,
        candidates: np.ndarray,
        acquisition: np.ndarray,
        measured_codes: set[tuple[int, ...]],
    ) -> list[np.ndarray]:
        order = np.argsort(acquisition)[::-1]
        selected: list[np.ndarray] = []
        for minimum_distance in (3, 2, 1, 0):
            for index in order:
                candidate = candidates[index]
                key = tuple(int(value) for value in candidate)
                if key in measured_codes or any(np.array_equal(candidate, old) for old in selected):
                    continue
                if selected and min(np.count_nonzero(candidate != old) for old in selected) < minimum_distance:
                    continue
                selected.append(candidate.copy())
                if len(selected) == self.config.measured_per_generation:
                    return selected
        return selected

    def optimize(
        self,
        state: TrajectoryState,
        angle_cmd_deg: float,
        delta_angle_cmd_deg: float,
        measure: MeasureFunction,
    ) -> OptimizationResult:
        angle_cmd_deg = float(angle_cmd_deg)
        if not -60.0 <= angle_cmd_deg <= 60.0:
            raise ValueError("commanded angle must remain within [-60, +60] degrees")
        if state.history and not np.isclose(state.angle_cmd_deg + delta_angle_cmd_deg, angle_cmd_deg, atol=1e-6):
            raise ValueError("angle_cmd_deg must equal previous angle plus delta_angle_cmd_deg")

        observations: list[ProbeObservation] = []
        measured_codes: set[tuple[int, ...]] = set()

        def observe(code: np.ndarray, repeated: bool = False) -> ProbeObservation:
            code = validate_joint_code(code)
            observation = ProbeObservation(code, float(measure(code.copy())), repeated)
            observations.append(observation)
            measured_codes.add(tuple(int(value) for value in code))
            return observation

        previous_code = state.previous_code
        observe(previous_code, repeated=True)
        observe(previous_code, repeated=True)
        initial_prediction = self.prior.predict(state, observations, angle_cmd_deg, delta_angle_cmd_deg)
        for candidate in self._initial_candidates(state, initial_prediction):
            observe(candidate)

        if len(observations) != self.config.initial_probe_count:
            raise RuntimeError("initial probe construction did not meet the eight-measurement contract")

        prior_prediction = self.prior.predict(state, observations, angle_cmd_deg, delta_angle_cmd_deg)
        probability = (
            0.65 * prior_prediction.probabilities
            + 0.25 * np.eye(STATE_COUNT)[previous_code]
            + 0.10 / STATE_COUNT
        )
        probability = np.maximum(probability, self.config.minimum_probability)
        probability /= probability.sum(axis=1, keepdims=True)
        probability[list(FIXED_VARIABLES)] = [1.0, 0.0, 0.0, 0.0]

        baseline_values = [item.power_dBm for item in observations if item.repeated_baseline]
        noise_span = float(np.ptp(baseline_values))
        best_observation = max(observations, key=lambda item: item.power_dBm)
        previous_generation_best = best_observation.power_dBm
        stop_reason = "maximum budget reached"
        prediction_mae = np.inf
        generations = 0

        for generation in range(1, self.config.max_generations + 1):
            generations = generation
            virtual = self._sample_virtual_population(probability)
            forced = np.vstack((
                previous_code,
                best_observation.code,
                np.argmax(prior_prediction.probabilities, axis=1),
                physics_reference_code(angle_cmd_deg),
            ))
            candidates = self._unique_rows(np.vstack((forced, virtual)))
            mean_score, uncertainty = self.prior.score_candidates(
                state, observations, candidates, angle_cmd_deg, delta_angle_cmd_deg
            )
            acquisition = mean_score + self.config.uncertainty_weight * uncertainty
            selected = self._select_diverse_candidates(candidates, acquisition, measured_codes)
            while len(selected) < self.config.measured_per_generation:
                fallback = self._sample_virtual_population(probability)
                extra_scores, extra_uncertainty = self.prior.score_candidates(
                    state, observations, fallback, angle_cmd_deg, delta_angle_cmd_deg
                )
                extra = self._select_diverse_candidates(
                    fallback, extra_scores + self.config.uncertainty_weight * extra_uncertainty,
                    measured_codes | {tuple(int(value) for value in code) for code in selected},
                )
                selected.extend(extra[: self.config.measured_per_generation - len(selected)])
            generation_observations = [observe(candidate) for candidate in selected]

            measured_powers = np.asarray([item.power_dBm for item in generation_observations])
            elite_count = max(2, int(np.ceil(self.config.elite_fraction * len(generation_observations))))
            elite_indices = np.argsort(measured_powers)[-elite_count:]
            elite_codes = np.asarray([generation_observations[index].code for index in elite_indices])
            elite_probability = np.column_stack([
                np.mean(elite_codes == state_index, axis=0) for state_index in range(STATE_COUNT)
            ])
            probability = (1.0 - self.config.smoothing) * probability + self.config.smoothing * elite_probability
            probability = np.maximum(probability, self.config.minimum_probability)
            probability /= probability.sum(axis=1, keepdims=True)
            probability[list(FIXED_VARIABLES)] = [1.0, 0.0, 0.0, 0.0]

            generation_best = max(generation_observations, key=lambda item: item.power_dBm)
            if generation_best.power_dBm > best_observation.power_dBm:
                best_observation = generation_best
            predicted, _ = self.prior.score_candidates(
                state,
                observations,
                np.asarray([item.code for item in generation_observations]),
                angle_cmd_deg,
                delta_angle_cmd_deg,
            )
            relative_measured = measured_powers - np.mean(baseline_values)
            prediction_mae = float(np.median(np.abs(predicted - relative_measured)))
            confidence = float(np.max(probability[FREE_VARIABLES], axis=1).mean())
            improvement = best_observation.power_dBm - previous_generation_best
            error_limit = max(
                self.config.prediction_error_floor_dB,
                3.0 * self.prior.validation_mae_dB,
            )
            stable = improvement <= max(noise_span, 0.25)
            if confidence >= self.config.confidence_threshold and prediction_mae <= error_limit and stable:
                stop_reason = "confidence, prediction error, and power stability thresholds met"
                break
            previous_generation_best = best_observation.power_dBm
            prior_prediction = self.prior.predict(
                state, observations, angle_cmd_deg, delta_angle_cmd_deg
            )

        confidence = float(np.max(probability[FREE_VARIABLES], axis=1).mean())
        error_limit = max(self.config.prediction_error_floor_dB, 3.0 * self.prior.validation_mae_dB)
        calibration_required = bool(
            prediction_mae > error_limit
            or (generations == self.config.max_generations and confidence < self.config.confidence_threshold)
        )
        recent_full_budget = [
            summary.measurement_count == self.config.maximum_measurements
            for summary in state.history[-2:]
        ]
        if len(recent_full_budget) == 2 and all(recent_full_budget) and len(observations) == self.config.maximum_measurements:
            calibration_required = True
        result = OptimizationResult(
            angle_cmd_deg=angle_cmd_deg,
            best_code=best_observation.code.copy(),
            best_power_dBm=best_observation.power_dBm,
            measurement_count=len(observations),
            generations=generations,
            confidence=confidence,
            prediction_mae_dB=prediction_mae,
            noise_span_dB=noise_span,
            calibration_required=calibration_required,
            stop_reason=stop_reason,
            observations=observations,
        )
        state.record_position(PositionSummary(
            angle_cmd_deg=angle_cmd_deg,
            delta_angle_cmd_deg=delta_angle_cmd_deg,
            direction=int(np.sign(delta_angle_cmd_deg)),
            best_code=result.best_code,
            best_power_dBm=result.best_power_dBm,
            measurement_count=result.measurement_count,
        ))
        return result

    def check_zero_anchor(self, state: TrajectoryState, measure: MeasureFunction) -> CalibrationCheck:
        if 0.0 not in state.calibrations:
            raise ValueError("a zero-degree calibration must be recorded first")
        anchor = state.calibrations[0.0]
        powers = np.asarray([measure(anchor.code.copy()), measure(anchor.code.copy())], dtype=float)
        mean_power = float(powers.mean())
        drop = float(anchor.power_dBm - mean_power)
        prediction = self.prior.predict(state, [], 0.0, -state.angle_cmd_deg)
        predicted_code = validate_joint_code(np.argmax(prediction.probabilities, axis=1))
        hamming_fraction = float(np.mean(predicted_code[FREE_VARIABLES] != anchor.code[FREE_VARIABLES]))
        command_error = float(abs(state.angle_cmd_deg))
        code_and_power_disagree = hamming_fraction > 0.25 and drop > max(1.0, float(np.ptp(powers)))
        return CalibrationCheck(
            mean_power_dBm=mean_power,
            power_drop_dB=drop,
            repeated_span_dB=float(np.ptp(powers)),
            code_hamming_fraction=hamming_fraction,
            command_angle_error_deg=command_error,
            calibration_required=bool(drop > 3.0 or code_and_power_disagree or command_error > 1.0),
        )


@dataclass
class FullCEResult:
    best_code: np.ndarray
    best_power_dBm: float
    measurement_count: int
    measured_codes: np.ndarray | None = None
    measured_powers_dBm: np.ndarray | None = None


def full_ce_optimize(
    measure: MeasureFunction,
    seed: int = 20260804,
    population_size: int = 72,
    max_iterations: int = 25,
    elite_fraction: float = 0.15,
    smoothing: float = 0.65,
    minimum_probability: float = 0.01,
) -> FullCEResult:
    """Run the original cold-start CE contract against any scalar measurement API."""

    rng = np.random.default_rng(seed)
    probability = np.full((VARIABLE_COUNT, STATE_COUNT), 1.0 / STATE_COUNT)
    probability[list(FIXED_VARIABLES)] = [1.0, 0.0, 0.0, 0.0]
    best_code = physics_reference_code(0.0)
    best_power = -np.inf
    measurement_count = 0
    measured_code_batches = []
    measured_power_batches = []
    for _ in range(max_iterations):
        samples = np.empty((population_size, VARIABLE_COUNT), dtype=int)
        for variable in range(VARIABLE_COUNT):
            samples[:, variable] = rng.choice(STATE_COUNT, population_size, p=probability[variable])
        samples[0] = best_code
        samples[1] = np.argmax(probability, axis=1)
        samples[:, list(FIXED_VARIABLES)] = 0
        powers = np.asarray([float(measure(code.copy())) for code in samples])
        measured_code_batches.append(samples.copy())
        measured_power_batches.append(powers.copy())
        measurement_count += population_size
        best_index = int(np.argmax(powers))
        if powers[best_index] > best_power:
            best_power = float(powers[best_index])
            best_code = samples[best_index].copy()
        elite_count = max(2, int(np.ceil(elite_fraction * population_size)))
        elite = samples[np.argsort(powers)[-elite_count:]]
        elite_probability = np.column_stack([
            np.mean(elite == state_index, axis=0) for state_index in range(STATE_COUNT)
        ])
        probability = (1.0 - smoothing) * probability + smoothing * elite_probability
        probability = np.maximum(probability, minimum_probability)
        probability /= probability.sum(axis=1, keepdims=True)
        probability[list(FIXED_VARIABLES)] = [1.0, 0.0, 0.0, 0.0]
    return FullCEResult(
        validate_joint_code(best_code),
        best_power,
        measurement_count,
        measured_codes=np.vstack(measured_code_batches),
        measured_powers_dBm=np.concatenate(measured_power_batches),
    )


def initialize_zero_anchor(
    state: TrajectoryState,
    measure: MeasureFunction,
    seed: int = 20260804,
) -> FullCEResult:
    """Measure the theoretical broadside code, then run and store full CE at 0 degrees."""

    measure(physics_reference_code(0.0))
    result = full_ce_optimize(measure, seed=seed)
    verification_powers = np.asarray([
        measure(result.best_code.copy()), measure(result.best_code.copy())
    ], dtype=float)
    state.history.clear()
    state.add_calibration(CalibrationRecord(
        angle_deg=0.0,
        code=result.best_code,
        power_dBm=float(verification_powers.mean()),
        measurement_count=result.measurement_count + 3,
        noise_span_dB=float(np.ptp(verification_powers)),
    ))
    return result
