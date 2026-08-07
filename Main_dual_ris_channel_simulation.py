"""双无人机双超表面仿真主程序：角度扫描与盲CE全过程都在main()中。"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from relay_sim.CE_Optimizer import (
    CEConfig,
    make_original_channel_evaluator,
    run_ce,
    uniform_probability,
)
from relay_sim.Channel_Modeling import (
    Noise_Power_dBm,
    build_far_field_channel,
    plot_link_results,
)
from relay_sim.MS_Configuration import (
    Columns,
    calculate_2bit_compensation_code,
    plot_ce_iteration_evolution,
)


def main() -> None:
    """按规则方位角采样完成双超表面链路仿真并显示三幅结果图。"""

    # ---------- 1. 实验扫描参数：只在本主程序中使用 ----------
    angle_start_deg = -60.0
    angle_stop_deg = 60.0
    angle_step_deg = 10.0
    test_angle_deg = 60.0

    # 角度数组只包含由起点、终点和步长形成的规则采样点，测试角度必须是其中一个采样点。
    angles_deg = np.arange(angle_start_deg, angle_stop_deg + 0.5 * angle_step_deg, angle_step_deg)
    if not np.any(np.isclose(angles_deg, test_angle_deg)):
        raise ValueError(
            f"test_angle_deg={test_angle_deg:g}°不在规则采样点中；"
            f"请从{angles_deg.tolist()}中选择，或修改angle_start_deg、angle_stop_deg和angle_step_deg。")

    # ---------- 2. CE参数：与冷启动、NN热启动共享同一个配置结构 ----------
    ce_config = CEConfig(
        population_size=50,
        max_iterations=25,
        elite_fraction=0.2,
        smoothing=0.6,
        minimum_probability=0.02,
        pilot_symbols_L=4,
        convergence_probability=0.9,
    )
    observation_iteration = 5
    rng = np.random.default_rng(20260724)

    # Figure 3固定展示第1次、用户指定的观察时刻和实际终止时刻；观察时刻必须先小于最大迭代次数。
    if not 1 < observation_iteration < ce_config.max_iterations:
        raise ValueError("observation_iteration必须大于1且小于max_iterations。")

    # ---------- 3. 为角度扫描结果分配列表 ----------
    noisy_power_ce_dBm, estimated_snr_ce_dB = [], []
    CE_Optimal_Matrices_MS1, CE_Optimal_Matrices_MS2 = [], []
    test_data = None

    # ---------- 4. 逐个方位角建立信道并运行CE；只在测试角度额外计算已知角度参考码本 ----------
    for angle_deg in angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        is_test_angle = np.isclose(angle_deg, test_angle_deg)
        ce_iteration_snapshots = []
        h12, _, _, _ = build_far_field_channel(angle_rad)

        if is_test_angle:
            # 已知角度参考：搜索公共相位，使2-bit量化后的目标方向相干叠加功率最大。
            (ideal_compensation_phase1, quantized_compensation_phase1, Known_Angle_Matrix_MS1,
             optimal_common_phase_rad, Direct_Coding_Matrix) = calculate_2bit_compensation_code(angle_deg)

            # 两块MS的列编号和直流偏置设计一致，因此已知角度时直接使用相同编码矩阵。
            ideal_compensation_phase2 = ideal_compensation_phase1.copy()
            quantized_compensation_phase2 = quantized_compensation_phase1.copy()
            Known_Angle_Matrix_MS2 = Known_Angle_Matrix_MS1.copy()

        # ---------- 5. 未知CSI盲CE：调用冷、热启动共用的原始优化核心 ----------
        evaluator = make_original_channel_evaluator(angle_rad, h12, ce_config)
        ce_result = run_ce(
            evaluator,
            uniform_probability(),
            config=ce_config,
            rng=rng,
            capture_snapshots=is_test_angle,
        )
        probability = ce_result.final_probability
        incumbent = ce_result.best_code
        incumbent_estimated_snr_linear = ce_result.best_score
        iteration_count = ce_result.iteration_count
        estimated_snr_history_dB = 10 * np.log10(ce_result.best_score_history)
        mean_max_probability_history = ce_result.mean_max_probability_history
        ce_iteration_snapshots = [
            {
                "iteration": snapshot.iteration,
                "probability": snapshot.probability.copy(),
                "Coding_Matrix_MS1": snapshot.incumbent_code[:Columns].copy(),
                "Coding_Matrix_MS2": snapshot.incumbent_code[Columns:].copy(),
                "estimated_snr_dB": float(10 * np.log10(snapshot.incumbent_score)),
            }
            for snapshot in ce_result.snapshots
        ]
        print(
            f"Angle {angle_deg:g} deg: CE stopped at iteration {iteration_count}; "
            f"{ce_result.termination_reason}."
        )

        # ---------- 7. 最终选择：直接采用全部CE迭代中由L个导频测得的历史最优码本 ----------
        CE_Optimal_Matrix_MS1 = incumbent[:Columns].copy()
        CE_Optimal_Matrix_MS2 = incumbent[Columns:].copy()

        # ---------- 8. 保存CE实际使用的L导频含噪功率和对应标准SNR估计 ----------
        noisy_power_ce_dBm.append(ce_result.best_power_dBm)
        estimated_snr_ce_dB.append(10 * np.log10(incumbent_estimated_snr_linear))
        CE_Optimal_Matrices_MS1.append(CE_Optimal_Matrix_MS1.copy())
        CE_Optimal_Matrices_MS2.append(CE_Optimal_Matrix_MS2.copy())

        # 保存测试角度的码本与CE过程，供最终方向图、编码图和迭代演化图使用。
        if is_test_angle:
            terminal_iteration = len(estimated_snr_history_dB)
            if not 1 < observation_iteration < terminal_iteration:
                raise ValueError(
                    f"observation_iteration={observation_iteration}必须位于第1次与实际终止的"
                    f"第{terminal_iteration}次迭代之间；请减小observation_iteration，或调整CE收敛参数。")
            test_data = {
                "angle_deg": float(angle_deg),
                "Known_Angle_Matrix_MS1": Known_Angle_Matrix_MS1.copy(),
                "Known_Angle_Matrix_MS2": Known_Angle_Matrix_MS2.copy(),
                "ideal_compensation_phase1": ideal_compensation_phase1.copy(),
                "ideal_compensation_phase2": ideal_compensation_phase2.copy(),
                "quantized_compensation_phase1": quantized_compensation_phase1.copy(),
                "quantized_compensation_phase2": quantized_compensation_phase2.copy(),
                "optimal_common_phase_deg": float(np.rad2deg(optimal_common_phase_rad)),
                "Direct_Coding_Matrix": Direct_Coding_Matrix.copy(),
                "CE_Optimal_Matrix_MS1": CE_Optimal_Matrix_MS1.copy(),
                "CE_Optimal_Matrix_MS2": CE_Optimal_Matrix_MS2.copy(),
                "estimated_snr_history_dB": np.asarray(estimated_snr_history_dB),
                "mean_max_probability_history": np.asarray(mean_max_probability_history),
                "final_probability": probability.copy(),
                "ce_iteration_snapshots": ce_iteration_snapshots,
                "selected_iterations": (1, observation_iteration, terminal_iteration),
                "terminal_iteration": terminal_iteration,
                "noise_power_dBm": Noise_Power_dBm,
                "pilot_symbols_L": ce_config.pilot_symbols_L,
            }

    # ---------- 9. 整理绘图需要的数组 ----------
    noisy_power_ce_dBm = np.asarray(noisy_power_ce_dBm)
    estimated_snr_ce_dB = np.asarray(estimated_snr_ce_dB)
    results = {
        "angles_deg": angles_deg,
        "noisy_power_ce_dBm": noisy_power_ce_dBm,
        "estimated_snr_ce_dB": estimated_snr_ce_dB,
        "noise_power_dBm": Noise_Power_dBm,
        "pilot_symbols_L": ce_config.pilot_symbols_L,
        "test": test_data,
    }

    # ---------- 10. 各模块直接绘制自己负责的数据 ----------
    # 统一采用清新、低饱和度的论文配色和浅色背景，避免纯黑或高饱和色块抢占视觉注意力。
    plt.rcParams.update({
        "figure.dpi": 100, "figure.facecolor": "white", "axes.facecolor": "#FBFCFD",
        "axes.edgecolor": "#AAB7C4", "axes.labelcolor": "#243447", "font.family": "Arial", "font.size": 9.5,
        "xtick.color": "#34495E", "ytick.color": "#34495E", "axes.grid": True,
        "grid.color": "#DCE6EE", "grid.alpha": 0.55, "grid.linewidth": 0.6,
        "legend.frameon": True, "legend.framealpha": 0.88, "legend.edgecolor": "#D6E0E8",
    })
    plot_link_results(results, np.asarray(CE_Optimal_Matrices_MS1), np.asarray(CE_Optimal_Matrices_MS2))

    # Figure 2展示测试角度下的估计SNR和平均最大概率历史，并标出Figure 3选取的三次迭代。
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    iteration = np.arange(1, test_data["estimated_snr_history_dB"].size + 1)
    axes[0].plot(iteration, test_data["estimated_snr_history_dB"], "-o", ms=3.5, color="#E76F51")
    axes[0].set(xlabel="CE iteration", ylabel="Estimated SNR (dB)",
                title=f"(a) Pilot-based CE history (L={ce_config.pilot_symbols_L})")
    axes[1].plot(iteration, test_data["mean_max_probability_history"], "-o", ms=3.5, color="#2A9D8F")
    axes[1].axhline(
        ce_config.convergence_probability,
        color="#6B7280", linestyle="--", label="Threshold"
    )
    axes[1].set(xlabel="CE iteration", ylabel="Mean maximum probability", ylim=(0.2, 1.02),
                title="(b) Mean maximum probability")
    # CE迭代次数只能是整数；整数刻度定位器会根据实际终止次数自动选择合适的整数间隔。
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].legend(loc="best")
    for selected_iteration in test_data["selected_iterations"]:
        axes[0].axvline(selected_iteration, color="#AAB7C4", linestyle="--", linewidth=0.8)
        axes[1].axvline(selected_iteration, color="#AAB7C4", linestyle="--", linewidth=0.8)
    figure.suptitle(f"Part II: CE convergence at selected angle theta={test_angle_deg:.0f}°", fontsize=12)
    figure.tight_layout()

    # Figure 3逐列展示第1次、用户指定的观察时刻和满足双重终止条件时的内部状态。
    plot_ce_iteration_evolution(test_data)

    # 阻塞式显示会一直保留全部图窗，手动关闭所有图窗后程序结束。
    plt.show()


if __name__ == "__main__":
    main()
