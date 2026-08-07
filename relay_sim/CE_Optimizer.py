"""与 Main_dual_ris_channel_simulation.py 一致的共享CE优化核心。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .Channel_Modeling import Fixed_Link_Field_Gain, Noise_Power_W, Transmit_Power_W
from .MS_Configuration import Columns, Compensation_Phasors, Element_Field_Exponent


STATE_COUNT = 4
COLUMN_COUNT = 2 * Columns
FIXED_COLUMNS = (0, Columns)
FREE_COLUMNS = np.asarray(
    [index for index in range(COLUMN_COUNT) if index not in FIXED_COLUMNS], dtype=int
)


@dataclass(frozen=True)
class CEConfig:
    population_size: int = 50
    max_iterations: int = 25
    elite_fraction: float = 0.20
    smoothing: float = 0.60
    minimum_probability: float = 0.02
    pilot_symbols_L: int = 4
    convergence_probability: float = 0.90

    def __post_init__(self) -> None:
        if self.population_size < 2 or self.max_iterations < 1:
            raise ValueError("population_size must be >=2 and max_iterations must be positive")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if not 0.0 <= self.smoothing <= 1.0:
            raise ValueError("smoothing must be in [0, 1]")
        if self.minimum_probability < 0.0 or self.pilot_symbols_L < 1:
            raise ValueError("minimum_probability must be nonnegative and pilot_symbols_L positive")
        if not 0.0 < self.convergence_probability <= 1.0:
            raise ValueError("convergence_probability must be in (0, 1]")


@dataclass
class CEIterationSnapshot:
    iteration: int
    probability: np.ndarray
    incumbent_code: np.ndarray
    incumbent_score: float
    incumbent_power_dBm: float
    mean_max_probability: float


@dataclass
class CEResult:
    best_code: np.ndarray
    best_score: float
    best_power_dBm: float
    iteration_count: int
    candidate_measurement_count: int
    raw_read_count: int
    final_probability: np.ndarray
    mean_max_probability: float
    termination_reason: str
    reached_probability_threshold: bool
    reached_max_iterations: bool
    best_score_history: np.ndarray
    best_power_history_dBm: np.ndarray
    mean_max_probability_history: np.ndarray
    measured_codes: np.ndarray
    measured_scores: np.ndarray
    snapshots: list[CEIterationSnapshot]


# evaluator返回“用于排序的分数”和“用于显示的功率dBm”。
PopulationEvaluator = Callable[
    [np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]
]


def uniform_probability() -> np.ndarray:
    probability = np.full((COLUMN_COUNT, STATE_COUNT), 1.0 / STATE_COUNT)
    probability[list(FIXED_COLUMNS)] = [1.0, 0.0, 0.0, 0.0]
    return probability


def normalize_probability(probability: np.ndarray, minimum_probability: float) -> np.ndarray:
    probability = np.asarray(probability, dtype=float).copy()
    if probability.shape != (COLUMN_COUNT, STATE_COUNT):
        raise ValueError(f"probability must have shape ({COLUMN_COUNT}, {STATE_COUNT})")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("probability must contain finite nonnegative values")
    probability = np.maximum(probability, minimum_probability)
    probability /= probability.sum(axis=1, keepdims=True)
    probability[list(FIXED_COLUMNS)] = [1.0, 0.0, 0.0, 0.0]
    return probability


def run_ce(
    evaluator: PopulationEvaluator,
    initial_probability: np.ndarray,
    config: CEConfig | None = None,
    seed: int = 20260724,
    rng: np.random.Generator | None = None,
    capture_snapshots: bool = False,
) -> CEResult:
    """运行冷/热启动共用的CE循环；二者只传入不同初始概率。"""

    config = config or CEConfig()
    rng = rng or np.random.default_rng(seed)
    probability = normalize_probability(initial_probability, config.minimum_probability)
    incumbent = None
    incumbent_score = -np.inf
    incumbent_power_dBm = -np.inf
    score_history, power_history, probability_history = [], [], []
    all_codes, all_scores, snapshots = [], [], []

    for iteration_count in range(1, config.max_iterations + 1):
        candidates = np.empty((config.population_size, COLUMN_COUNT), dtype=int)
        for column_index in range(COLUMN_COUNT):
            candidates[:, column_index] = rng.choice(
                STATE_COUNT, config.population_size, p=probability[column_index]
            )

        # 与原主程序一致：保留历史最优和当前概率众数。
        if incumbent is not None:
            candidates[0] = incumbent
        candidates[1] = np.argmax(probability, axis=1)
        candidates[:, list(FIXED_COLUMNS)] = 0

        scores, powers_dBm = evaluator(candidates.copy(), rng)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        powers_dBm = np.asarray(powers_dBm, dtype=float).reshape(-1)
        if scores.size != config.population_size or powers_dBm.size != config.population_size:
            raise ValueError("evaluator must return one score and one power for every candidate")
        if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(powers_dBm)):
            raise ValueError("CE evaluator returned a non-finite measurement")
        all_codes.append(candidates.copy())
        all_scores.append(scores.copy())

        best_index = int(np.argmax(scores))
        if incumbent is None or scores[best_index] > incumbent_score:
            incumbent = candidates[best_index].copy()
            incumbent_score = float(scores[best_index])
            incumbent_power_dBm = float(powers_dBm[best_index])

        elite_count = max(2, int(np.ceil(config.elite_fraction * config.population_size)))
        elite = candidates[np.argsort(scores)[-elite_count:]]
        elite_probability = np.column_stack(
            [(elite == state).mean(axis=0) for state in range(STATE_COUNT)]
        )
        probability = (
            (1.0 - config.smoothing) * probability
            + config.smoothing * elite_probability
        )
        probability = normalize_probability(probability, config.minimum_probability)

        mean_max_probability = float(
            np.max(probability[FREE_COLUMNS], axis=1).mean()
        )
        score_history.append(incumbent_score)
        power_history.append(incumbent_power_dBm)
        probability_history.append(mean_max_probability)
        if capture_snapshots:
            snapshots.append(
                CEIterationSnapshot(
                    iteration=iteration_count,
                    probability=probability.copy(),
                    incumbent_code=incumbent.copy(),
                    incumbent_score=incumbent_score,
                    incumbent_power_dBm=incumbent_power_dBm,
                    mean_max_probability=mean_max_probability,
                )
            )

        # 终止条件严格保持为“概率达到阈值 或 达到最大迭代次数”。
        if (
            mean_max_probability >= config.convergence_probability
            or iteration_count >= config.max_iterations
        ):
            break

    reached_probability = mean_max_probability >= config.convergence_probability
    reached_maximum = iteration_count >= config.max_iterations
    if reached_probability and reached_maximum:
        reason = "both_conditions_reached"
    elif reached_probability:
        reason = "probability_converged"
    else:
        reason = "max_iterations_reached"

    measured_codes = np.vstack(all_codes)
    measured_scores = np.concatenate(all_scores)
    return CEResult(
        best_code=incumbent.copy(),
        best_score=incumbent_score,
        best_power_dBm=incumbent_power_dBm,
        iteration_count=iteration_count,
        candidate_measurement_count=config.population_size * iteration_count,
        raw_read_count=config.population_size * iteration_count * config.pilot_symbols_L,
        final_probability=probability.copy(),
        mean_max_probability=mean_max_probability,
        termination_reason=reason,
        reached_probability_threshold=reached_probability,
        reached_max_iterations=reached_maximum,
        best_score_history=np.asarray(score_history),
        best_power_history_dBm=np.asarray(power_history),
        mean_max_probability_history=np.asarray(probability_history),
        measured_codes=measured_codes,
        measured_scores=measured_scores,
        snapshots=snapshots,
    )


def make_original_channel_evaluator(
    angle_rad: float,
    h12: np.ndarray,
    config: CEConfig | None = None,
) -> PopulationEvaluator:
    """完全复现原主程序的复导频、总功率和SNR估计。"""

    config = config or CEConfig()
    scan_field_factor = np.cos(angle_rad) ** (2 * Element_Field_Exponent)

    def evaluate(candidates: np.ndarray, rng: np.random.Generator):
        v1_batch = Compensation_Phasors[candidates[:, :Columns]]
        v2_batch = Compensation_Phasors[candidates[:, Columns:]]
        air_channels = np.einsum(
            "mi,ij,mj->m", np.conj(v2_batch), h12, v1_batch, optimize=True
        )
        h_eff_batch = (
            Fixed_Link_Field_Gain * air_channels / Columns**2 * scan_field_factor
        )
        noise = np.sqrt(Noise_Power_W / 2) * (
            rng.normal(size=(config.pilot_symbols_L, config.population_size))
            + 1j * rng.normal(size=(config.pilot_symbols_L, config.population_size))
        )
        received_pilots = np.sqrt(Transmit_Power_W) * h_eff_batch[None, :] + noise
        estimated_total_power_W = np.mean(np.abs(received_pilots) ** 2, axis=0)
        estimated_snr = (estimated_total_power_W - Noise_Power_W) / Noise_Power_W
        power_dBm = 10.0 * np.log10(estimated_total_power_W) + 30.0
        return estimated_snr, power_dBm

    return evaluate


def make_scalar_measurement_evaluator(
    measure: Callable[[np.ndarray], float],
    config: CEConfig | None = None,
    noise_power_W: float = Noise_Power_W,
) -> PopulationEvaluator:
    """把真实频谱仪 ``measure(code)->dBm`` 适配为原始L导频SNR评价。

    每个候选实际读取 ``pilot_symbols_L`` 次。dBm必须先转为线性功率再求平均，
    不能直接平均dBm；这样 ``raw_read_count`` 才与真实仪器调用次数一致。
    """

    config = config or CEConfig()
    if not np.isfinite(noise_power_W) or noise_power_W <= 0.0:
        raise ValueError("noise_power_W must be finite and positive")

    def evaluate(candidates: np.ndarray, _rng: np.random.Generator):
        readings_dBm = np.asarray(
            [
                [float(measure(code.copy())) for code in candidates]
                for _ in range(config.pilot_symbols_L)
            ]
        )
        if not np.all(np.isfinite(readings_dBm)):
            raise ValueError("spectrum analyzer returned a non-finite power")
        readings_W = 10.0 ** ((readings_dBm - 30.0) / 10.0)
        total_power_W = readings_W.mean(axis=0)
        estimated_snr = (total_power_W - noise_power_W) / noise_power_W
        averaged_power_dBm = 10.0 * np.log10(total_power_W) + 30.0
        return estimated_snr, averaged_power_dBm

    return evaluate
