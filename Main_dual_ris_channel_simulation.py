"""双无人机双超表面仿真主程序：角度扫描与盲CE全过程都在main()中。"""

import matplotlib.pyplot as plt
import numpy as np

from relay_sim.Channel_Modeling import (
    Aperture_Width_MS,
    Far_Field_Distance_M,
    Fixed_Link_Field_Gain,
    Noise_Power_W,
    Noise_Power_dBm,
    Separation_Distance_M,
    Transmit_Power_W,
    Transmit_Power_dBm,
    build_far_field_channel,
    plot_link_results,
)
from relay_sim.MS_Configuration import (
    Columns,
    Element_Field_Exponent,
    Lambda,
    Period_MS,
    Compensation_Phasors,
    Compensation_Phase_States_Rad,
    calculate_2bit_compensation_code,
    plot_ce_iteration_evolution,
)


def main() -> None:
    """按25个方位角完成双超表面链路仿真并显示全部结果。"""

    # ---------- 1. 实验扫描参数：只在本主程序中使用 ----------
    angle_start_deg = -60.0
    angle_stop_deg = 60.0
    angle_step_deg = 10.0
    test_angle_deg = 20.0

    # 角度步长可以任意指定；终点和测试角度若不在规则采样点中，就自动补入并按升序排列。
    if angle_step_deg <= 0:
        raise ValueError("angle_step_deg必须大于0。")
    angles_deg = np.arange(angle_start_deg, angle_stop_deg + 1e-12, angle_step_deg)
    if not np.any(np.isclose(angles_deg, angle_stop_deg)):
        angles_deg = np.append(angles_deg, angle_stop_deg)
    if not np.any(np.isclose(angles_deg, test_angle_deg)):
        angles_deg = np.append(angles_deg, test_angle_deg)
    angles_deg = np.sort(angles_deg)

    # ---------- 2. CE参数：算法是本文件核心，因此不再放入其他模块 ----------
    population_size = 72
    max_iterations = 25
    ce_middle_iteration = 5
    elite_fraction = 0.15
    smoothing = 0.65
    minimum_probability = 0.01
    pilot_symbols_L = 16
    convergence_probability = 0.95
    rng = np.random.default_rng(20260724)

    # 迭代演化图固定展示第1次、用户指定的中间一次和最后一次；这里只需修改这个中间编号。
    if not 1 < ce_middle_iteration < max_iterations:
        raise ValueError("ce_middle_iteration必须位于第1次与最后一次迭代之间。")

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
        h12, _, _, alpha = build_far_field_channel(angle_rad)

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

        # incumbent保存“L个含噪导频估计SNR”意义下截至当前最好的联合码本。
        incumbent = np.zeros(variable_count, dtype=int)
        incumbent_estimated_snr_linear = -np.inf
        incumbent_total_power_W = 0.0
        estimated_snr_history_dB, confidence_history = [], []

        # ---------- 6. CE迭代：采样 → 测量 → Elite → 更新概率 ----------
        for iteration in range(max_iterations):

            # 每个变量都按照自己的一行概率独立抽取补偿相位状态，矩阵维度(population_size*variable_count)。
            samples = np.empty((population_size, variable_count), dtype=int)
            for variable in range(variable_count):
                samples[:, variable] = rng.choice(state_count, population_size, p=probability[variable])

            # 保留历史最优解和当前概率众数，避免优秀码本在随机采样中丢失。
            if np.isfinite(incumbent_estimated_snr_linear):
                samples[0] = incumbent
            samples[1] = np.argmax(probability, axis=1)
            samples[:, fixed_variables] = 0

            # 将每组32维状态拆成MS1和MS2各16列的复补偿相位向量。
            v1_batch = Compensation_Phasors[samples[:, :Columns]]
            v2_batch = Compensation_Phasors[samples[:, Columns:]]

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

            # 标准SNR估计为(mean(|y_l|²)-sigma²)/sigma²；有限L时估计值可能暂时为负。
            # CE在线性域直接比较这些估计值，不提前截断，因此完整保留低SNR测量的相对大小。
            estimated_total_power_W = np.mean(np.abs(received_pilots) ** 2, axis=0)
            estimated_snr_scores_linear = (estimated_total_power_W - Noise_Power_W) / Noise_Power_W

            # 如果本代出现更高的导频估计SNR，就更新历史最优联合码本。
            best_index = int(np.argmax(estimated_snr_scores_linear))
            if estimated_snr_scores_linear[best_index] > incumbent_estimated_snr_linear:
                incumbent_estimated_snr_linear = float(estimated_snr_scores_linear[best_index])
                incumbent_total_power_W = float(estimated_total_power_W[best_index])
                incumbent = samples[best_index].copy()

            # 取导频估计SNR最高的前15%样本，并统计每个变量中四种补偿相位的出现频率。
            elite_count = max(2, int(np.ceil(elite_fraction * population_size)))
            elite_samples = samples[np.argsort(estimated_snr_scores_linear)[-elite_count:]]
            elite_probability = np.column_stack([(elite_samples == state).mean(axis=0) for state in range(state_count)])

            # 用平滑系数更新概率，同时保留最小探索概率，防止过早锁死。
            probability = (1 - smoothing) * probability + smoothing * elite_probability
            probability = np.maximum(probability, minimum_probability)
            probability /= probability.sum(axis=1, keepdims=True)
            for fixed in fixed_variables:
                probability[fixed] = [1.0, 0.0, 0.0, 0.0]

            # dB无法表示非正SNR估计，因此只在绘图转换时使用极小正数保护log10；CE排序不受影响。
            estimated_snr_history_dB.append(10 * np.log10(max(incumbent_estimated_snr_linear, 1e-30)))
            mask = np.ones(variable_count, dtype=bool)
            mask[list(fixed_variables)] = False
            confidence_history.append(np.max(probability[mask], axis=1).mean())

            # 只记录测试角度的CE内部状态，供后续展示概率、相位和方向图如何随迭代演化。
            if is_test_angle:
                ce_iteration_snapshots.append({
                    "iteration": iteration + 1,
                    "probability": probability.copy(),
                    "Coding_Matrix_MS1": incumbent[:Columns].copy(),
                    "Coding_Matrix_MS2": incumbent[Columns:].copy(),
                    "estimated_snr_dB": float(estimated_snr_history_dB[-1]),
                })

        # ---------- 7. 最终选择：直接采用全部CE迭代中由L个导频测得的历史最优码本 ----------
        CE_Optimal_Matrix_MS1 = incumbent[:Columns].copy()
        CE_Optimal_Matrix_MS2 = incumbent[Columns:].copy()

        # 最后一帧采用历史最优编码，保证演化图终点与其余结果图完全一致。
        if is_test_angle:
            final_snapshot = {
                "iteration": len(estimated_snr_history_dB),
                "probability": probability.copy(),
                "Coding_Matrix_MS1": CE_Optimal_Matrix_MS1.copy(),
                "Coding_Matrix_MS2": CE_Optimal_Matrix_MS2.copy(),
                "estimated_snr_dB": float(10 * np.log10(max(incumbent_estimated_snr_linear, 1e-30))),
            }
            if ce_iteration_snapshots:
                ce_iteration_snapshots[-1] = final_snapshot
            else:
                ce_iteration_snapshots.append(final_snapshot)

        # ---------- 8. 保存CE实际使用的L导频含噪功率和对应标准SNR估计 ----------
        noisy_power_ce_dBm.append(10 * np.log10(max(incumbent_total_power_W, 1e-30)) + 30)
        estimated_snr_ce_dB.append(10 * np.log10(max(incumbent_estimated_snr_linear, 1e-30)))
        CE_Optimal_Matrices_MS1.append(CE_Optimal_Matrix_MS1.copy())
        CE_Optimal_Matrices_MS2.append(CE_Optimal_Matrix_MS2.copy())

        # 保存测试角度的码本与CE过程，供最终方向图、编码图和迭代演化图使用。
        if is_test_angle:
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
                "confidence_history": np.asarray(confidence_history),
                "final_probability": probability.copy(),
                "ce_iteration_snapshots": ce_iteration_snapshots,
                "selected_iterations": (1, ce_middle_iteration, len(estimated_snr_history_dB)),
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

    # ---------- 10. 打印最重要的派生量，便于核对实验条件 ----------
    print(f"Wavelength: {Lambda * 1e3:.3f} mm")
    print(f"Unit period / wavelength: {Period_MS / Lambda:.3f}")
    print(f"Aperture width: {Aperture_Width_MS:.3f} m")
    print(f"Fraunhofer distance: {Far_Field_Distance_M:.3f} m")
    print(f"UAV separation: {Separation_Distance_M:.3f} m")
    Broadside_h_eff = Fixed_Link_Field_Gain * alpha
    Broadside_Signal_Power_W = Transmit_Power_W * np.abs(Broadside_h_eff) ** 2
    Broadside_SNR_Linear = Broadside_Signal_Power_W / Noise_Power_W
    print(f"Transmit power: {Transmit_Power_dBm:.1f} dBm")
    print(f"Directly specified noise power sigma^2: {Noise_Power_dBm:.3f} dBm")
    print(f"Broadside clean received power: {10 * np.log10(Broadside_Signal_Power_W) + 30:.3f} dBm")
    print(f"Broadside theoretical SNR: {10 * np.log10(Broadside_SNR_Linear):.3f} dB")
    print(f"CE pilot symbols per candidate L: {pilot_symbols_L}")
    print("CE objective: pilot-estimated SNR = (mean(|y_l|^2) - sigma^2) / sigma^2")

    # ---------- 11. 各模块直接绘制自己负责的数据 ----------
    # 统一采用清新、低饱和度的论文配色和浅色背景，避免纯黑或高饱和色块抢占视觉注意力。
    plt.rcParams.update({
        "figure.dpi": 100, "figure.facecolor": "white", "axes.facecolor": "#FBFCFD",
        "axes.edgecolor": "#AAB7C4", "axes.labelcolor": "#243447", "font.family": "Arial", "font.size": 9.5,
        "xtick.color": "#34495E", "ytick.color": "#34495E", "axes.grid": True,
        "grid.color": "#DCE6EE", "grid.alpha": 0.55, "grid.linewidth": 0.6,
        "legend.frameon": True, "legend.framealpha": 0.88, "legend.edgecolor": "#D6E0E8",
    })
    plot_link_results(results, np.asarray(CE_Optimal_Matrices_MS1), np.asarray(CE_Optimal_Matrices_MS2))

    # Figure 3属于第二部分：展示测试角度下完整的估计SNR历史与概率置信度，并标出Figure 4选择的四次迭代。
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    iteration = np.arange(1, test_data["estimated_snr_history_dB"].size + 1)
    axes[0].plot(iteration, test_data["estimated_snr_history_dB"], "-o", ms=3.5, color="#E76F51")
    axes[0].set(xlabel="CE iteration", ylabel="Estimated SNR (dB)",
                title=f"(a) Pilot-based CE history (L={pilot_symbols_L})")
    axes[1].plot(iteration, test_data["confidence_history"], "-o", ms=3.5, color="#2A9D8F")
    axes[1].axhline(convergence_probability, color="#6B7280", linestyle="--", label="Threshold")
    axes[1].set(xlabel="CE iteration", ylabel="Mean maximum probability", ylim=(0.2, 1.02),
                title="(b) CE probability confidence")
    axes[1].legend(loc="best")
    for selected_iteration in test_data["selected_iterations"]:
        axes[0].axvline(selected_iteration, color="#AAB7C4", linestyle="--", linewidth=0.8)
        axes[1].axvline(selected_iteration, color="#AAB7C4", linestyle="--", linewidth=0.8)
    figure.suptitle(f"CE convergence at selected angle theta={test_angle_deg:.0f}°", fontsize=12)
    figure.tight_layout()

    # 迭代演化图逐列展示第1次、用户指定的中间时刻和最后一次迭代的内部状态。
    plot_ce_iteration_evolution(test_data)

    # 阻塞式显示会一直保留全部图窗，手动关闭所有图窗后程序结束。
    plt.show()


if __name__ == "__main__":
    main()
