"""双无人机双超表面仿真主程序：角度扫描与盲CE全过程都在main()中。"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from relay_sim.Channel_Modeling import (
    Fixed_Link_Field_Gain,
    Noise_Power_W,
    Noise_Power_dBm,
    Transmit_Power_W,
    build_far_field_channel,
    plot_link_results,
)
from relay_sim.MS_Configuration import (
    Columns,
    Element_Field_Exponent,
    Compensation_Phasors,
    Compensation_Phase_States_Rad,
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

    # ---------- 2. CE参数：算法是本文件核心，因此不再放入其他模块 ----------
    population_size = 50
    max_iterations = 25
    observation_iteration = 5
    elite_fraction = 0.2
    smoothing = 0.6
    minimum_probability = 0.02
    pilot_symbols_L = 4
    convergence_probability = 0.9
    rng = np.random.default_rng(20260724)

    # Figure 3固定展示第1次、用户指定的观察时刻和实际终止时刻；观察时刻必须先小于最大迭代次数。
    if not 1 < observation_iteration < max_iterations:
        raise ValueError("observation_iteration必须大于1且小于max_iterations。")

    # 两块MS各有16列、每列4种状态，所以联合概率矩阵大小为32×4。
    variable_count, state_count = 2 * Columns, Compensation_Phase_States_Rad.size

    # 第一列补偿相位固定为0°，可以消除每块MS公共补偿相位不唯一的问题。
    fixed_variables = (0, Columns)

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

        # ---------- 5. 未知CSI盲CE：从均匀概率开始 ----------
        # 初始化均匀概率，矩阵维度(variable_count*state_count)每个元素都是1/4。
        probability = np.full((variable_count, state_count), 1 / state_count)

        # MS1和MS2第一列固定为状态0
        for fixed in fixed_variables:
            probability[fixed] = [1.0, 0.0, 0.0, 0.0]

        # incumbent保存“L个含噪导频估计SNR”意义下，截至当前最好的联合码本。
        incumbent = None
        incumbent_estimated_snr_linear = None
        incumbent_total_power_W = 0.0
        estimated_snr_history_dB, mean_max_probability_history = [], []

        # ---------- 6. CE迭代：生成候选编码 → 测量 → Elite → 更新概率 → 判断双重终止条件 ----------
        for iteration_count in range(1, max_iterations + 1):

            # 每行是一组双MS联合编码，每列是一个可控列的2-bit状态，矩阵尺寸为population_size×variable_count。
            Candidate_Coding_Matrices = np.empty((population_size, variable_count), dtype=int)
            for variable in range(variable_count):
                Candidate_Coding_Matrices[:, variable] = rng.choice(
                    state_count, population_size, p=probability[variable])

            # 保留历史最优解和当前概率众数，避免优秀码本在随机采样中丢失。
            if incumbent is not None:
                Candidate_Coding_Matrices[0] = incumbent
            Candidate_Coding_Matrices[1] = np.argmax(probability, axis=1)
            Candidate_Coding_Matrices[:, fixed_variables] = 0

            # 将每组32维状态拆成MS1和MS2各16列的复补偿相位向量。
            v1_batch = Compensation_Phasors[Candidate_Coding_Matrices[:, :Columns]]
            v2_batch = Compensation_Phasors[Candidate_Coding_Matrices[:, Columns:]]

            # v2^H*H12*v1给出空中复信道；除以N1*N2后，再加入固定链路场增益与双端扫描场因子。
            Air_Channels = np.einsum("mi,ij,mj->m", np.conj(v2_batch), h12, v1_batch, optimize=True)
            Scan_Field_Factor = np.cos(angle_rad) ** (2 * Element_Field_Exponent)
            h_eff_batch = Fixed_Link_Field_Gain * Air_Channels / Columns**2 * Scan_Field_Factor

            # 对每个候选发送L个单位导频：y_l=sqrt(Pt)*h_eff+n_l，n_l~CN(0,sigma²)。
            # 复高斯噪声的实部和虚部方差均为sigma²/2，因此E[|n_l|²]=sigma²。
            noise = np.sqrt(Noise_Power_W / 2) * (
                rng.normal(size=(pilot_symbols_L, population_size))
                + 1j * rng.normal(size=(pilot_symbols_L, population_size)))
            received_pilots = np.sqrt(Transmit_Power_W) * h_eff_batch[None, :] + noise

            # 标准SNR估计为(mean(|y_l|²)-sigma²)/sigma²，当前参数范围内该估计值默认为正数。
            estimated_total_power_W = np.mean(np.abs(received_pilots) ** 2, axis=0)
            estimated_snr_scores_linear = (estimated_total_power_W - Noise_Power_W) / Noise_Power_W

            # 如果本代出现更高的导频估计SNR，就更新历史最优联合码本。
            best_index = int(np.argmax(estimated_snr_scores_linear))
            if incumbent is None or estimated_snr_scores_linear[best_index] > incumbent_estimated_snr_linear:
                incumbent_estimated_snr_linear = float(estimated_snr_scores_linear[best_index])
                incumbent_total_power_W = float(estimated_total_power_W[best_index])
                incumbent = Candidate_Coding_Matrices[best_index].copy()

            # 取导频估计SNR最高的一组候选联合编码，并统计每个变量中四种补偿相位的出现频率。
            elite_count = max(2, int(np.ceil(elite_fraction * population_size)))
            Elite_Coding_Matrices = Candidate_Coding_Matrices[
                np.argsort(estimated_snr_scores_linear)[-elite_count:]]
            elite_probability = np.column_stack([
                (Elite_Coding_Matrices == state).mean(axis=0) for state in range(state_count)])

            # 用平滑系数更新概率，同时保留最小探索概率，防止过早锁死。
            probability = (1 - smoothing) * probability + smoothing * elite_probability
            probability = np.maximum(probability, minimum_probability)
            probability /= probability.sum(axis=1, keepdims=True)
            for fixed in fixed_variables:
                probability[fixed] = [1.0, 0.0, 0.0, 0.0]

            estimated_snr_history_dB.append(10 * np.log10(incumbent_estimated_snr_linear))
            # 两个固定参考列的最大概率恒为1，因此只对其余30个实际搜索变量计算平均最大概率。
            mask = np.ones(variable_count, dtype=bool)
            mask[list(fixed_variables)] = False
            mean_max_probability = float(np.max(probability[mask], axis=1).mean())
            mean_max_probability_history.append(mean_max_probability)

            # 只记录测试角度的CE内部状态，供后续展示概率、相位和方向图如何随迭代演化。
            if is_test_angle:
                ce_iteration_snapshots.append({
                    "iteration": iteration_count,
                    "probability": probability.copy(),
                    "Coding_Matrix_MS1": incumbent[:Columns].copy(),
                    "Coding_Matrix_MS2": incumbent[Columns:].copy(),
                    "estimated_snr_dB": float(estimated_snr_history_dB[-1]),
                })

            # 达到平均最大概率阈值时提前收敛；否则最多运行到max_iterations后终止。
            if mean_max_probability >= convergence_probability or iteration_count >= max_iterations:
                break

        # 每个扫描角度的CE结束后，明确打印本次优化由哪一个终止条件触发。
        reached_probability_threshold = mean_max_probability >= convergence_probability
        reached_maximum_iterations = iteration_count >= max_iterations
        if reached_probability_threshold and reached_maximum_iterations:
            termination_reason = (
                f"both conditions were met: mean_max_probability={mean_max_probability:.4f} "
                f">= convergence_probability={convergence_probability:.4f}, and "
                f"iteration_count reached max_iterations={max_iterations}")
        elif reached_probability_threshold:
            termination_reason = (
                f"mean_max_probability={mean_max_probability:.4f} "
                f">= convergence_probability={convergence_probability:.4f}")
        else:
            termination_reason = f"iteration_count reached max_iterations={max_iterations}"
        print(f"Angle {angle_deg:g} deg: CE stopped at iteration {iteration_count}; {termination_reason}.")

        # ---------- 7. 最终选择：直接采用全部CE迭代中由L个导频测得的历史最优码本 ----------
        CE_Optimal_Matrix_MS1 = incumbent[:Columns].copy()
        CE_Optimal_Matrix_MS2 = incumbent[Columns:].copy()

        # ---------- 8. 保存CE实际使用的L导频含噪功率和对应标准SNR估计 ----------
        noisy_power_ce_dBm.append(10 * np.log10(incumbent_total_power_W) + 30)
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
                "pilot_symbols_L": pilot_symbols_L,
            }

    # ---------- 9. 整理绘图需要的数组 ----------
    noisy_power_ce_dBm = np.asarray(noisy_power_ce_dBm)
    estimated_snr_ce_dB = np.asarray(estimated_snr_ce_dB)
    results = {
        "angles_deg": angles_deg,
        "noisy_power_ce_dBm": noisy_power_ce_dBm,
        "estimated_snr_ce_dB": estimated_snr_ce_dB,
        "noise_power_dBm": Noise_Power_dBm,
        "pilot_symbols_L": pilot_symbols_L,
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
                title=f"(a) Pilot-based CE history (L={pilot_symbols_L})")
    axes[1].plot(iteration, test_data["mean_max_probability_history"], "-o", ms=3.5, color="#2A9D8F")
    axes[1].axhline(convergence_probability, color="#6B7280", linestyle="--", label="Threshold")
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
