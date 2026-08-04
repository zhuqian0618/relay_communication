"""Simulation, dataset generation, and evaluation helpers for the ±60° arc."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .Channel_Modeling import Noise_Power_W, Separation_Distance_M, build_far_field_channel, link_metrics
from .MS_Configuration import Columns, Compensation_Phasors
from .Neural_CE import (
    FREE_VARIABLES,
    VARIABLE_COUNT,
    CalibrationRecord,
    PositionSummary,
    ProbeObservation,
    TrainingExample,
    TrajectoryState,
    full_ce_optimize,
    physics_reference_code,
    validate_joint_code,
)


@dataclass
class DomainRandomization:
    command_angle_error_std_deg: float = 0.35
    radius_error_std_fraction: float = 0.01
    phase_error_std_deg: float = 7.0
    gain_error_std_dB: float = 0.45
    receiver_drift_std_dB: float = 0.35
    measurement_noise_std_dB: float = 0.18
    failed_element_probability: float = 0.005


class ArcExperimentSimulator:
    """Spectrum-analyzer-like scalar power measurement on the horizontal arc."""

    def __init__(
        self,
        seed: int = 20260804,
        randomization: DomainRandomization | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.randomization = randomization or DomainRandomization()
        self.commanded_angle_deg = 0.0
        self.true_angle_deg = 0.0
        self.measurement_count = 0
        self._initialize_hardware_errors()

    def _initialize_hardware_errors(self) -> None:
        settings = self.randomization
        self.phase_offsets_rad = np.deg2rad(
            self.rng.normal(0.0, settings.phase_error_std_deg, size=VARIABLE_COUNT)
        )
        self.gain_scales = 10.0 ** (
            self.rng.normal(0.0, settings.gain_error_std_dB, size=VARIABLE_COUNT) / 20.0
        )
        failed = self.rng.random(VARIABLE_COUNT) < settings.failed_element_probability
        self.gain_scales[failed] *= 0.05
        self.receiver_drift_dB = float(self.rng.normal(0.0, settings.receiver_drift_std_dB))
        self.separation_distance_M = float(
            Separation_Distance_M * (1.0 + self.rng.normal(0.0, settings.radius_error_std_fraction))
        )

    def reset_hardware(self) -> None:
        self._initialize_hardware_errors()

    def move_to(self, commanded_angle_deg: float, true_angle_deg: float | None = None) -> None:
        if not -60.0 <= commanded_angle_deg <= 60.0:
            raise ValueError("UAV2 commanded angle must remain in [-60, +60] degrees")
        self.commanded_angle_deg = float(commanded_angle_deg)
        if true_angle_deg is None:
            true_angle_deg = commanded_angle_deg + self.rng.normal(
                0.0, self.randomization.command_angle_error_std_deg
            )
        self.true_angle_deg = float(np.clip(true_angle_deg, -60.0, 60.0))

    def clean_power_dBm(self, joint_code: np.ndarray) -> float:
        code = validate_joint_code(joint_code)
        angle_rad = np.deg2rad(self.true_angle_deg)
        h12, _, _, _ = build_far_field_channel(angle_rad, self.separation_distance_M)
        phasors = Compensation_Phasors[code] * self.gain_scales * np.exp(1j * self.phase_offsets_rad)
        v1 = phasors[:Columns]
        v2 = phasors[Columns:]
        _, signal_power_W, _ = link_metrics(v1, v2, angle_rad, h12)
        total_power_W = signal_power_W + Noise_Power_W
        return float(10.0 * np.log10(max(total_power_W, 1e-30)) + 30.0 + self.receiver_drift_dB)

    def measure(self, joint_code: np.ndarray) -> float:
        self.measurement_count += 1
        return float(
            self.clean_power_dBm(joint_code)
            + self.rng.normal(0.0, self.randomization.measurement_noise_std_dB)
        )


def canonical_training_paths(step_deg: float = 10.0) -> list[np.ndarray]:
    if step_deg <= 0.0:
        raise ValueError("step_deg must be positive")

    def segment(start: float, stop: float) -> np.ndarray:
        direction = 1.0 if stop >= start else -1.0
        values = np.arange(start, stop + direction * 1e-9, direction * step_deg)
        if not np.isclose(values[-1], stop):
            values = np.append(values, stop)
        return values

    return [
        segment(0.0, 60.0),
        segment(60.0, 0.0),
        segment(0.0, -60.0),
        segment(-60.0, 0.0),
        segment(-60.0, 60.0),
        segment(60.0, -60.0),
    ]


def experiment_path(step_deg: float = 10.0) -> np.ndarray:
    """Return 0→+60→0→-60→0 without duplicate turning points."""

    paths = canonical_training_paths(step_deg)
    return np.concatenate((paths[0], paths[1][1:], paths[2][1:], paths[3][1:]))


def _select_training_candidates(
    codes: np.ndarray,
    powers: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    count = min(int(count), codes.shape[0])
    elite_count = max(2, count // 3)
    elite = np.argsort(powers)[-elite_count:]
    remaining = np.setdiff1d(np.arange(codes.shape[0]), elite, assume_unique=False)
    random_count = count - elite.size
    random_indices = rng.choice(remaining, size=random_count, replace=False)
    indices = np.concatenate((elite, random_indices))
    return codes[indices].copy(), powers[indices].copy()


def generate_training_examples(
    trajectory_count: int = 6,
    step_deg: float = 10.0,
    seed: int = 20260804,
    full_ce_labels: bool = True,
    candidates_per_position: int = 24,
) -> list[TrainingExample]:
    """Generate domain-randomized trajectory examples and full-CE labels.

    ``full_ce_labels=False`` is intended only for quick smoke tests.  Formal
    training and all reported comparisons should retain the default full CE.
    """

    rng = np.random.default_rng(seed)
    paths = canonical_training_paths(step_deg)
    examples: list[TrainingExample] = []
    for trajectory_index in range(int(trajectory_count)):
        simulator = ArcExperimentSimulator(seed + 1009 * trajectory_index)
        path = paths[trajectory_index % len(paths)].copy()
        if trajectory_index >= len(paths):
            jitter = rng.normal(0.0, min(1.0, step_deg * 0.1), size=path.size)
            jitter[0] = 0.0
            path = np.clip(path + jitter, -60.0, 60.0)
        state = TrajectoryState()
        simulator.move_to(float(path[0]), true_angle_deg=float(path[0]))
        initial_code = physics_reference_code(float(path[0]))
        initial_power = simulator.clean_power_dBm(initial_code)
        if np.isclose(path[0], 0.0):
            state.add_calibration(CalibrationRecord(0.0, initial_code, initial_power))
        else:
            state.record_position(PositionSummary(
                float(path[0]), 0.0, 0, initial_code, initial_power, 0
            ))

        for angle in path[1:]:
            previous_angle = state.angle_cmd_deg
            delta = float(angle - previous_angle)
            true_angle = float(np.clip(angle + rng.normal(0.0, 0.35), -60.0, 60.0))
            simulator.move_to(float(angle), true_angle_deg=true_angle)
            baseline_power = simulator.clean_power_dBm(state.previous_code)
            baseline_probes = [
                ProbeObservation(state.previous_code, baseline_power, True),
                ProbeObservation(state.previous_code, baseline_power + rng.normal(0.0, 0.12), True),
            ]

            if full_ce_labels:
                ce_result = full_ce_optimize(
                    simulator.measure,
                    seed=seed + trajectory_index * 10000 + int(round((angle + 60.0) * 10)),
                )
                target_code = ce_result.best_code
                target_power = simulator.clean_power_dBm(target_code)
                candidate_codes, candidate_powers = _select_training_candidates(
                    ce_result.measured_codes,
                    np.asarray([simulator.clean_power_dBm(code) for code in ce_result.measured_codes]),
                    candidates_per_position,
                    rng,
                )
            else:
                target_code = physics_reference_code(float(angle))
                target_power = simulator.clean_power_dBm(target_code)
                candidates = [target_code, state.previous_code]
                for _ in range(max(2, candidates_per_position - 2)):
                    candidate = target_code.copy()
                    change_count = int(rng.integers(2, 7))
                    variables = rng.choice(FREE_VARIABLES, size=change_count, replace=False)
                    candidate[variables] = rng.integers(0, 4, size=change_count)
                    candidates.append(validate_joint_code(candidate))
                candidate_codes = np.asarray(candidates)
                candidate_powers = np.asarray([
                    simulator.clean_power_dBm(code) for code in candidate_codes
                ])

            relative_powers = candidate_powers - baseline_power
            examples.append(TrainingExample(
                state=deepcopy(state),
                angle_cmd_deg=float(angle),
                delta_angle_cmd_deg=delta,
                target_code=target_code,
                probes=baseline_probes,
                candidate_codes=candidate_codes,
                candidate_relative_powers_dB=relative_powers,
                trajectory_id=trajectory_index,
            ))
            state.record_position(PositionSummary(
                angle_cmd_deg=float(angle),
                delta_angle_cmd_deg=delta,
                direction=int(np.sign(delta)),
                best_code=target_code,
                best_power_dBm=target_power,
                measurement_count=1800 if full_ce_labels else 0,
            ))
    return examples


def split_examples_by_trajectory(
    examples: Sequence[TrainingExample],
    validation_fraction: float = 0.2,
    seed: int = 20260804,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split whole trajectories to prevent adjacent-position data leakage."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie between zero and one")
    examples = list(examples)
    trajectory_ids = np.unique([example.trajectory_id for example in examples])
    if trajectory_ids.size < 2:
        return examples, []
    rng = np.random.default_rng(seed)
    shuffled = trajectory_ids.copy()
    rng.shuffle(shuffled)
    validation_count = max(1, int(np.ceil(validation_fraction * shuffled.size)))
    validation_ids = set(int(value) for value in shuffled[:validation_count])
    training = [example for example in examples if example.trajectory_id not in validation_ids]
    validation = [example for example in examples if example.trajectory_id in validation_ids]
    return training, validation


def evaluate_prior(prior, examples: Iterable[TrainingExample]) -> dict[str, float]:
    """Evaluate code accuracy and candidate-power MAE on held-out trajectories."""

    examples = list(examples)
    if not examples:
        return {"code_accuracy": float("nan"), "score_mae_dB": float("nan"), "examples": 0.0}
    correct, total, errors = 0, 0, []
    for example in examples:
        prediction = prior.predict(
            example.state, example.probes, example.angle_cmd_deg, example.delta_angle_cmd_deg
        )
        predicted_code = np.argmax(prediction.probabilities, axis=1)
        correct += int(np.count_nonzero(predicted_code[FREE_VARIABLES] == example.target_code[FREE_VARIABLES]))
        total += FREE_VARIABLES.size
        if example.candidate_codes is not None and example.candidate_relative_powers_dB is not None:
            scores, _ = prior.score_candidates(
                example.state,
                example.probes,
                example.candidate_codes,
                example.angle_cmd_deg,
                example.delta_angle_cmd_deg,
            )
            errors.extend(np.abs(scores - example.candidate_relative_powers_dB).tolist())
    metrics = {
        "code_accuracy": float(correct / total),
        "score_mae_dB": float(np.median(errors)) if errors else float("nan"),
        "examples": float(len(examples)),
    }
    if errors:
        prior.validation_mae_dB = metrics["score_mae_dB"]
    return metrics
