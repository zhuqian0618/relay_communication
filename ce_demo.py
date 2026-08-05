"""使用频谱仪探针和MLP完成36次预算的编码更新。

真实实验接入时，保留 probe_assisted_ce()，把本文件末尾仿真的
``measure(code)`` 替换成频谱仪通信函数即可。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from model import (
    FIXED_COLUMNS,
    FREE_COLUMNS,
    STATE_COUNT,
    COLUMN_COUNT,
    SimpleCodeNet,
    build_probe_codes,
    build_probe_features,
    load_model,
    make_simulated_measurement,
    predict_probabilities,
    reference_joint_code,
    validate_joint_code,
)


# ============================== 1. 可调整参数 ==============================

MODEL_PATH = Path("probe_code_net.pth")
TEST_STEP_DEG = 10.0
TRANSMIT_POWER_DBM = 0.0
RANDOM_SEED = 20260805
DEMO_PLOT_PATH = Path("probe_ce_comparison.png")

TOTAL_MEASUREMENTS = 36
FIRST_PROBE_MEASUREMENTS = 8
CANDIDATES_PER_GENERATION = 12
ELITE_COUNT = 3
SMOOTHING = 0.60
MINIMUM_PROBABILITY = 0.01

MeasureFunction = Callable[[np.ndarray], float]


# ============================== 2. CE公共工具 ==============================

@dataclass
class CEResult:
    best_code: np.ndarray
    best_power_dBm: float
    measurement_count: int
    measured_codes: np.ndarray
    measured_powers_dBm: np.ndarray
    final_probability: np.ndarray
    used_network_weight: float


def normalize_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.maximum(np.asarray(probability, dtype=float), MINIMUM_PROBABILITY)
    probability /= probability.sum(axis=1, keepdims=True)
    probability[list(FIXED_COLUMNS)] = [1.0, 0.0, 0.0, 0.0]
    return probability


def sample_codes(
    probability: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    Candidate_Coding_Matrices = np.empty((count, COLUMN_COUNT), dtype=int)
    for column_index in range(COLUMN_COUNT):
        Candidate_Coding_Matrices[:, column_index] = rng.choice(
            STATE_COUNT, size=count, p=probability[column_index])
    Candidate_Coding_Matrices[:, list(FIXED_COLUMNS)] = 0
    return Candidate_Coding_Matrices


def update_probability(
    probability: np.ndarray,
    codes: np.ndarray,
    powers_dBm: np.ndarray,
) -> np.ndarray:
    finite_indices = np.flatnonzero(np.isfinite(powers_dBm))
    if finite_indices.size == 0:
        return probability
    elite_count = min(ELITE_COUNT, finite_indices.size)
    ranked = finite_indices[np.argsort(powers_dBm[finite_indices])]
    elite = codes[ranked[-elite_count:]]
    elite_probability = np.column_stack(
        [np.mean(elite == state, axis=0) for state in range(STATE_COUNT)]
    )
    updated = (1.0 - SMOOTHING) * probability + SMOOTHING * elite_probability
    return normalize_probability(updated)


def coherent_transition_candidates(
    model: SimpleCodeNet,
    previous_code: np.ndarray,
    network_probability: np.ndarray,
    count: int = 3,
) -> np.ndarray:
    """选出网络认为最可能的物理连续编码，避免各列状态被随意拼接。"""

    previous_code = validate_joint_code(previous_code)
    # 模板来自训练数据中出现过的完整编码变化，只包含30个离散状态差，
    # 不包含角度、位置或方向。
    unique_deltas = np.asarray(model.transition_deltas, dtype=int).reshape(-1, 30)
    if unique_deltas.shape[0] == 0:
        return np.tile(previous_code, (count, 1))
    candidates = np.tile(previous_code, (unique_deltas.shape[0], 1))
    candidates[:, FREE_COLUMNS] = (
        candidates[:, FREE_COLUMNS] + unique_deltas
    ) % STATE_COUNT
    candidates[:, list(FIXED_COLUMNS)] = 0

    column_indices = np.arange(COLUMN_COUNT)[None, :]
    log_probability = np.log(np.maximum(network_probability, 1e-8))
    scores = np.sum(log_probability[column_indices, candidates], axis=1)
    selected = np.argsort(scores)[-count:][::-1]
    return candidates[selected]


def finish_result(
    measured_codes: list[np.ndarray],
    measured_powers: list[float],
    probability: np.ndarray,
    network_weight: float,
) -> CEResult:
    codes = np.asarray(measured_codes, dtype=int)
    powers = np.asarray(measured_powers, dtype=float)
    if codes.shape != (TOTAL_MEASUREMENTS, COLUMN_COUNT):
        raise RuntimeError("CE must contain exactly 36 measured codes")
    if not np.any(np.isfinite(powers)):
        raise RuntimeError("all spectrum-analyzer readings are invalid")
    best_index = int(np.nanargmax(powers))
    return CEResult(
        best_code=codes[best_index].copy(),
        best_power_dBm=float(powers[best_index]),
        measurement_count=int(powers.size),
        measured_codes=codes,
        measured_powers_dBm=powers,
        final_probability=probability,
        used_network_weight=network_weight,
    )


# ============================== 3. 36次冷启动CE ==============================

def cold_start_ce(
    measure: MeasureFunction,
    seed: int = RANDOM_SEED,
) -> CEResult:
    """不使用上一编码和神经网络，执行3代×12次测量。"""

    rng = np.random.default_rng(seed)
    probability = normalize_probability(
        np.full((COLUMN_COUNT, STATE_COUNT), 1.0 / STATE_COUNT)
    )
    measured_codes: list[np.ndarray] = []
    measured_powers: list[float] = []

    for _ in range(3):
        generation_codes = sample_codes(probability, CANDIDATES_PER_GENERATION, rng)
        generation_powers = np.asarray([float(measure(code.copy())) for code in generation_codes])
        measured_codes.extend(generation_codes)
        measured_powers.extend(generation_powers)
        probability = update_probability(probability, generation_codes, generation_powers)

    return finish_result(measured_codes, measured_powers, probability, network_weight=0.0)


# ============================== 4. 36次探针辅助CE ==============================

def _safe_feature_powers(
    previous_best_power_dBm: float,
    baseline: np.ndarray,
    probes: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """无效或异常探针不让程序崩溃，而是降低神经网络权重。"""

    raw = np.concatenate(([previous_best_power_dBm], baseline, probes)).astype(float)
    finite = np.isfinite(raw)
    if not np.any(finite):
        return 0.0, np.zeros(2), np.zeros(6), 0.0

    fallback = float(np.median(raw[finite]))
    safe = np.where(finite, raw, fallback)
    safe_previous = float(safe[0])
    safe_baseline = safe[1:3]
    safe_probes = safe[3:]
    baseline_mean = float(np.mean(safe_baseline))
    is_outlier = (
        np.any(np.abs(safe_probes - baseline_mean) > 12.0)
        or abs(safe_baseline[0] - safe_baseline[1]) > 3.0
    )
    if not np.all(finite):
        network_weight = 0.0
    elif is_outlier:
        network_weight = 0.20
    else:
        network_weight = 0.70
    return safe_previous, safe_baseline, safe_probes, network_weight


def probe_assisted_ce(
    model: SimpleCodeNet,
    previous_code: np.ndarray,
    previous_best_power_dBm: float,
    measure: MeasureFunction,
    seed: int = RANDOM_SEED,
) -> CEResult:
    """先做8次信道探测，再在总预算36次内完成两代后续CE。"""

    rng = np.random.default_rng(seed)
    previous_code = validate_joint_code(previous_code)
    probe_codes = build_probe_codes(previous_code)

    # 第1--8次：上一编码重复两次，随后测量六个固定探针。
    first_codes = np.vstack((previous_code, previous_code, probe_codes))
    first_powers = np.asarray([float(measure(code.copy())) for code in first_codes])
    safe_previous, safe_baseline, safe_probes, network_weight = _safe_feature_powers(
        float(previous_best_power_dBm), first_powers[:2], first_powers[2:]
    )
    features = build_probe_features(
        previous_code, safe_previous, safe_baseline, safe_probes
    )
    network_probability = predict_probabilities(model, features)
    previous_probability = np.eye(STATE_COUNT)[previous_code]

    if network_weight == 0.70:
        previous_weight, uniform_weight = 0.20, 0.10
    elif network_weight == 0.20:
        previous_weight, uniform_weight = 0.60, 0.20
    else:
        previous_weight, uniform_weight = 0.75, 0.25
    probability = normalize_probability(
        network_weight * network_probability
        + previous_weight * previous_probability
        + uniform_weight / STATE_COUNT
    )

    # 第9--12次：网络逐列最大概率编码 + 三个物理连续候选。
    # 后三者把训练轨迹中可能出现的“整组编码变化”作为模板，可避免30个
    # 各列独立采样后可能拼成方向图不连贯的编码。
    network_code = validate_joint_code(np.argmax(network_probability, axis=1))
    network_candidates = np.vstack(
        (network_code, coherent_transition_candidates(model, previous_code, network_probability, 3))
    )
    network_candidates[0] = network_code
    network_powers = np.asarray([float(measure(code.copy())) for code in network_candidates])

    measured_codes = [code.copy() for code in first_codes]
    measured_codes.extend(code.copy() for code in network_candidates)
    measured_powers = first_powers.tolist() + network_powers.tolist()
    first_generation_codes = np.vstack((first_codes, network_candidates))
    first_generation_powers = np.concatenate((first_powers, network_powers))
    probability = update_probability(
        probability, first_generation_codes, first_generation_powers
    )

    # 第13--24次和第25--36次：两代标准CE，每代12个候选。
    for _ in range(2):
        generation_codes = sample_codes(probability, CANDIDATES_PER_GENERATION, rng)
        # 强制保留截至目前的实测最佳编码，防止CE概率更新后遗忘好结果。
        finite_powers = np.asarray(measured_powers, dtype=float)
        if np.any(np.isfinite(finite_powers)):
            generation_codes[0] = measured_codes[int(np.nanargmax(finite_powers))]
        generation_powers = np.asarray([float(measure(code.copy())) for code in generation_codes])
        measured_codes.extend(code.copy() for code in generation_codes)
        measured_powers.extend(generation_powers.tolist())
        probability = update_probability(probability, generation_codes, generation_powers)

    return finish_result(measured_codes, measured_powers, probability, network_weight)


# ============================== 5. 仿真对照实验 ==============================

def build_test_path(step_deg: float = TEST_STEP_DEG) -> np.ndarray:
    def segment(start: float, stop: float) -> np.ndarray:
        direction = 1.0 if stop >= start else -1.0
        values = np.arange(start, stop + direction * 1e-9, direction * step_deg)
        if not np.isclose(values[-1], stop):
            values = np.append(values, stop)
        return values

    return np.concatenate(
        (
            segment(0.0, 60.0),
            segment(60.0, 0.0)[1:],
            segment(0.0, -60.0)[1:],
            segment(-60.0, 0.0)[1:],
        )
    )


def main() -> None:
    model = load_model(MODEL_PATH)
    path = build_test_path()
    scenarios = (("High SNR", -100.0), ("Default SNR", -90.0), ("Low SNR", -75.0))
    plot_data = []

    for scenario_index, (name, noise_power_dBm) in enumerate(scenarios):
        previous_code = reference_joint_code(float(path[0]))
        initial_measure = make_simulated_measurement(
            float(path[0]), TRANSMIT_POWER_DBM, noise_power_dBm, seed=RANDOM_SEED
        )
        previous_best_power = initial_measure(previous_code)
        warm_powers, cold_powers, reused_powers, reference_powers = [], [], [], []

        for index, angle in enumerate(path[1:], start=1):
            seed = RANDOM_SEED + scenario_index * 100000 + index
            warm_measure = make_simulated_measurement(
                float(angle), TRANSMIT_POWER_DBM, noise_power_dBm, seed=seed
            )
            cold_measure = make_simulated_measurement(
                float(angle), TRANSMIT_POWER_DBM, noise_power_dBm, seed=seed
            )
            comparison_measure = make_simulated_measurement(
                float(angle), TRANSMIT_POWER_DBM, noise_power_dBm, seed=seed + 1
            )
            reused_power = comparison_measure(previous_code)
            reference_power = comparison_measure(reference_joint_code(float(angle)))
            warm = probe_assisted_ce(
                model, previous_code, previous_best_power, warm_measure, seed=seed
            )
            cold = cold_start_ce(cold_measure, seed=seed)
            if warm.measurement_count != 36 or cold.measurement_count != 36:
                raise RuntimeError("both CE methods must use exactly 36 measurements")
            previous_code = warm.best_code
            previous_best_power = warm.best_power_dBm
            warm_powers.append(warm.best_power_dBm)
            cold_powers.append(cold.best_power_dBm)
            reused_powers.append(reused_power)
            reference_powers.append(reference_power)

        warm_powers = np.asarray(warm_powers)
        cold_powers = np.asarray(cold_powers)
        reused_powers = np.asarray(reused_powers)
        reference_powers = np.asarray(reference_powers)
        plot_data.append((name, warm_powers, cold_powers, reused_powers, reference_powers))
        print(
            f"{name:11s} | warm-cold={np.mean(warm_powers-cold_powers):+.3f} dB | "
            f"warm-reference median={np.median(warm_powers-reference_powers):+.3f} dB | "
            "reads/location=36"
        )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10.0, 8.5), sharex=True)
    x_angles = path[1:]
    for axis, (name, warm, cold, reused, reference) in zip(axes, plot_data):
        axis.plot(x_angles, warm, "-o", ms=3, label="Probe-MLP + CE")
        axis.plot(x_angles, cold, "--s", ms=3, label="Cold-start CE")
        axis.plot(x_angles, reused, ":", label="Reuse previous code")
        axis.plot(x_angles, reference, "-.", label="Reference code")
        axis.set(title=name, ylabel="Power (dBm)")
        axis.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Angle used only by the simulator (deg)")
    figure.tight_layout()
    figure.savefig(DEMO_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print("Saved:", DEMO_PLOT_PATH)


if __name__ == "__main__":
    main()
