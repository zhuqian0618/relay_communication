"""初始纯CE、2 dB功率触发和NN热启动CE的完整演示。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from model import (
    FIXED_COLUMNS,
    STATE_COUNT,
    SimpleCodeNet,
    build_code_history_features,
    load_model,
    make_simulated_measurement,
    predict_probabilities,
    validate_joint_code,
)
from relay_sim.CE_Optimizer import (
    CEConfig,
    CEResult,
    make_original_channel_evaluator,
    normalize_probability,
    run_ce,
    uniform_probability,
)
from relay_sim.Channel_Modeling import build_far_field_channel


# ============================== 1. 可调整参数 ==============================

MODEL_PATH = Path("code_evolution_net.pth")
TRIGGER_DROP_DB = 2.0
MONITOR_WINDOW = 5
ANCHOR_REPEATS = 5
SIMULATION_ANGLE_STEP_DEG = 0.5
RANDOM_SEED = 20260807
DEMO_PLOT_PATH = Path("cold_warm_ce_comparison.png")

# 冷启动和热启动必须共享同一个配置对象。
CE_CONFIG = CEConfig(
    population_size=50,
    max_iterations=25,
    elite_fraction=0.20,
    smoothing=0.60,
    minimum_probability=0.02,
    pilot_symbols_L=4,
    convergence_probability=0.90,
)


# ============================== 2. 2 dB触发监测器 ==============================

@dataclass
class PowerDropMonitor:
    threshold_dB: float = TRIGGER_DROP_DB
    window_size: int = MONITOR_WINDOW

    def __post_init__(self) -> None:
        if self.threshold_dB <= 0.0 or self.window_size < 1:
            raise ValueError("threshold_dB and window_size must be positive")
        self.anchor_power_dBm: float | None = None
        self.recent_powers: deque[float] = deque(maxlen=self.window_size)
        self.last_drop_dB = 0.0

    def calibrate(self, repeated_powers_dBm: np.ndarray) -> float:
        powers = np.asarray(repeated_powers_dBm, dtype=float).reshape(-1)
        if powers.size != ANCHOR_REPEATS or not np.all(np.isfinite(powers)):
            raise ValueError(f"anchor calibration requires {ANCHOR_REPEATS} finite readings")
        self.anchor_power_dBm = float(np.median(powers))
        self.recent_powers.clear()
        self.last_drop_dB = 0.0
        return self.anchor_power_dBm

    def update(self, power_dBm: float) -> bool:
        if self.anchor_power_dBm is None:
            raise RuntimeError("call calibrate() before monitoring")
        if not np.isfinite(power_dBm):
            raise ValueError("spectrum-analyzer power must be finite")
        self.recent_powers.append(float(power_dBm))
        if len(self.recent_powers) < self.window_size:
            return False
        monitored_power = float(np.median(self.recent_powers))
        self.last_drop_dB = self.anchor_power_dBm - monitored_power
        return self.last_drop_dB >= self.threshold_dB


# ============================== 3. NN初始概率 ==============================

def neural_initial_probability(
    model: SimpleCodeNet,
    latest_best_code: np.ndarray,
    previous_best_code: np.ndarray | None,
    config: CEConfig = CE_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """在第一代CE前产生热启动概率；返回概率和241维网络输入。"""

    latest = validate_joint_code(latest_best_code)
    features = build_code_history_features(latest, previous_best_code)
    neural_probability = predict_probabilities(model, features)
    previous_one_hot = np.eye(STATE_COUNT)[latest]
    uniform = uniform_probability()

    if features[-1] == 0.0:
        initial = 0.30 * neural_probability + 0.40 * previous_one_hot + 0.30 * uniform
    else:
        initial = 0.60 * neural_probability + 0.30 * previous_one_hot + 0.10 * uniform
    initial[list(FIXED_COLUMNS)] = [1.0, 0.0, 0.0, 0.0]
    return normalize_probability(initial, config.minimum_probability), features


# ============================== 4. 冷/热启动调用 ==============================

def run_cold_ce_at_angle(
    angle_deg: float,
    seed: int,
    config: CEConfig = CE_CONFIG,
) -> CEResult:
    angle_rad = np.deg2rad(float(angle_deg))
    h12, _, _, _ = build_far_field_channel(angle_rad)
    evaluator = make_original_channel_evaluator(angle_rad, h12, config)
    return run_ce(evaluator, uniform_probability(), config=config, seed=seed)


def run_warm_ce_at_angle(
    model: SimpleCodeNet,
    angle_deg: float,
    latest_best_code: np.ndarray,
    previous_best_code: np.ndarray | None,
    seed: int,
    config: CEConfig = CE_CONFIG,
) -> tuple[CEResult, np.ndarray]:
    initial_probability, features = neural_initial_probability(
        model, latest_best_code, previous_best_code, config
    )
    angle_rad = np.deg2rad(float(angle_deg))
    h12, _, _, _ = build_far_field_channel(angle_rad)
    evaluator = make_original_channel_evaluator(angle_rad, h12, config)
    result = run_ce(evaluator, initial_probability, config=config, seed=seed)
    return result, features


# ============================== 5. 连续移动轨迹 ==============================

def _segment(start: float, stop: float, step_deg: float) -> np.ndarray:
    direction = 1.0 if stop >= start else -1.0
    values = np.arange(start, stop + direction * 1e-10, direction * step_deg)
    if not np.isclose(values[-1], stop):
        values = np.append(values, stop)
    return np.round(values, 10)


def build_test_path(step_deg: float = SIMULATION_ANGLE_STEP_DEG) -> np.ndarray:
    """0→+60→0→-60→0，转折点不重复。"""

    return np.concatenate(
        (
            _segment(0.0, 60.0, step_deg),
            _segment(60.0, 0.0, step_deg)[1:],
            _segment(0.0, -60.0, step_deg)[1:],
            _segment(-60.0, 0.0, step_deg)[1:],
        )
    )


def calibrate_anchor(angle_deg: float, code: np.ndarray, seed: int) -> np.ndarray:
    measure = make_simulated_measurement(angle_deg, seed=seed)
    return np.asarray([measure(code) for _ in range(ANCHOR_REPEATS)])


# ============================== 6. 完整演示 ==============================

def main() -> None:
    model = load_model(MODEL_PATH)
    path = build_test_path()

    # 实验最开始只运行均匀概率纯CE，不调用神经网络。
    initial = run_cold_ce_at_angle(float(path[0]), RANDOM_SEED)
    latest_best_code = initial.best_code.copy()
    previous_best_code = None
    monitor = PowerDropMonitor()
    monitor.calibrate(
        calibrate_anchor(float(path[0]), latest_best_code, RANDOM_SEED + 1)
    )
    print(
        f"Initial cold CE: iterations={initial.iteration_count}, "
        f"power={initial.best_power_dBm:.2f} dBm, reason={initial.termination_reason}"
    )

    events = []
    for position_index, angle_deg in enumerate(path[1:], start=1):
        measure = make_simulated_measurement(
            float(angle_deg), seed=RANDOM_SEED + 1000 + position_index
        )
        if not monitor.update(measure(latest_best_code)):
            continue

        event_seed = RANDOM_SEED + 100000 + len(events)
        start_time = perf_counter()
        cold = run_cold_ce_at_angle(float(angle_deg), event_seed)
        cold_runtime_s = perf_counter() - start_time
        start_time = perf_counter()
        warm, _ = run_warm_ce_at_angle(
            model,
            float(angle_deg),
            latest_best_code,
            previous_best_code,
            event_seed,
        )
        warm_runtime_s = perf_counter() - start_time
        events.append(
            {
                "angle_deg": float(angle_deg),
                "drop_dB": monitor.last_drop_dB,
                "cold_iterations": cold.iteration_count,
                "warm_iterations": warm.iteration_count,
                "cold_measurements": cold.candidate_measurement_count,
                "warm_measurements": warm.candidate_measurement_count,
                "cold_raw_reads": cold.raw_read_count,
                "warm_raw_reads": warm.raw_read_count,
                "cold_runtime_s": cold_runtime_s,
                "warm_runtime_s": warm_runtime_s,
                "cold_power_dBm": cold.best_power_dBm,
                "warm_power_dBm": warm.best_power_dBm,
                "cold_reason": cold.termination_reason,
                "warm_reason": warm.termination_reason,
            }
        )
        print(
            f"Trigger {len(events):2d}: angle={angle_deg:+5.1f} deg, "
            f"drop={monitor.last_drop_dB:.2f} dB, "
            f"cold/warm iterations={cold.iteration_count}/{warm.iteration_count}, "
            f"warm-cold={warm.best_power_dBm-cold.best_power_dBm:+.2f} dB"
        )

        # 实际闭环采用热启动CE结果，然后重建局部功率锚点。
        previous_best_code = latest_best_code.copy()
        latest_best_code = warm.best_code.copy()
        monitor.calibrate(
            calibrate_anchor(
                float(angle_deg), latest_best_code, RANDOM_SEED + 200000 + len(events)
            )
        )

    if not events:
        raise RuntimeError("test trajectory produced no 2 dB trigger event")

    angles = np.asarray([event["angle_deg"] for event in events])
    cold_iterations = np.asarray([event["cold_iterations"] for event in events])
    warm_iterations = np.asarray([event["warm_iterations"] for event in events])
    cold_measurements = np.asarray([event["cold_measurements"] for event in events])
    warm_measurements = np.asarray([event["warm_measurements"] for event in events])
    cold_raw_reads = np.asarray([event["cold_raw_reads"] for event in events])
    warm_raw_reads = np.asarray([event["warm_raw_reads"] for event in events])
    cold_runtime = np.asarray([event["cold_runtime_s"] for event in events])
    warm_runtime = np.asarray([event["warm_runtime_s"] for event in events])
    power_difference = np.asarray(
        [event["warm_power_dBm"] - event["cold_power_dBm"] for event in events]
    )
    within_one_dB = float(np.mean(power_difference >= -1.0))
    median_quality_ok = bool(np.median(power_difference) >= -0.5)
    tail_quality_ok = bool(within_one_dB >= 0.95)
    cold_max_ratio = float(np.mean(cold_iterations >= CE_CONFIG.max_iterations))
    warm_max_ratio = float(np.mean(warm_iterations >= CE_CONFIG.max_iterations))
    max_iteration_ok = bool(warm_max_ratio <= cold_max_ratio)
    quality_passed = median_quality_ok and tail_quality_ok and max_iteration_ok
    print(
        f"Summary: events={len(events)}, median iteration reduction="
        f"{np.median(cold_iterations-warm_iterations):+.1f}, "
        f"median warm-cold power={np.median(power_difference):+.2f} dB, "
        f"within 1 dB ratio={within_one_dB:.3f}"
    )
    print(
        "Median cold/warm cost: "
        f"candidates={np.median(cold_measurements):.0f}/{np.median(warm_measurements):.0f}, "
        f"raw reads={np.median(cold_raw_reads):.0f}/{np.median(warm_raw_reads):.0f}, "
        f"runtime={np.median(cold_runtime):.4f}/{np.median(warm_runtime):.4f} s"
    )
    print(
        "Quality gate: " + ("PASS" if quality_passed else "FAIL")
        + f" | median>=-0.5 dB: {median_quality_ok}"
        + f" | 95% within 1 dB: {tail_quality_ok}"
        + f" | max-iteration ratio warm/cold="
        + f"{warm_max_ratio:.3f}/{cold_max_ratio:.3f}: {max_iteration_ok}"
    )
    if not quality_passed:
        print("Do not claim acceleration: final-power quality requirements were not met.")

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
    event_index = np.arange(1, len(events) + 1)
    axes[0].plot(event_index, cold_iterations, "-o", label="Cold-start CE")
    axes[0].plot(event_index, warm_iterations, "-s", label="NN warm-start CE")
    axes[0].set(ylabel="Iterations", title="CE convergence iterations")
    axes[0].legend()
    axes[1].bar(event_index, power_difference, color="#4C78A8")
    axes[1].axhline(-0.5, color="#E76F51", linestyle="--", label="Median quality target")
    axes[1].set(xlabel="Trigger event", ylabel="Warm - cold power (dB)", title="Power quality")
    axes[1].legend()
    for index, angle in zip(event_index, angles):
        axes[1].text(index, power_difference[index - 1], f"{angle:+.1f}°", fontsize=7)
    figure.tight_layout()
    figure.savefig(DEMO_PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print("Saved:", DEMO_PLOT_PATH)


if __name__ == "__main__":
    main()
