"""双无人机双超表面仿真主程序：角度扫描与盲CE全过程都在main()中。"""

import matplotlib.pyplot as plt
import numpy as np

from relay_sim.channel import (
    Aperture_Width_MS,
    Far_Field_Distance_M,
    Separation_Distance_M,
    build_far_field_channel,
    plot_channel_at_test_angle,
)
from relay_sim.link_budget import Base_Power_dBm, Noise_Power_dBm, plot_link_results, received_power_dBm
from relay_sim.metasurface import (
    Columns,
    Element_Field_Exponent,
    Lambda,
    Period_MS,
    Compensation_Phasors,
    Compensation_Phase_States_Rad,
    calculate_2bit_compensation_code,
    plot_ce_coding_matrices,
    plot_patterns,
)


def main() -> None:
    """按19个方位角完成双超表面链路仿真并显示全部结果。"""

    # ---------- 1. 实验扫描参数：只在本主程序中使用 ----------
    angles_deg = np.arange(-45.0, 45.01, 5.0)
    test_angle_deg = 25.0

    # ---------- 2. CE参数：算法是本文件核心，因此不再放入其他模块 ----------
    population_size = 72
    max_iterations = 22
    elite_fraction = 0.15
    smoothing = 0.65
    minimum_probability = 0.01
    SNR_dB = 30.0
    final_verification_repeats = 8
    convergence_probability = 0.985
    rng = np.random.default_rng(20260724)

    # 两块MS各有16列、每列4种状态，所以联合概率矩阵大小为32×4。
    variable_count, state_count = 2 * Columns, Compensation_Phase_States_Rad.size

    # 第一列补偿相位固定为0°，可以消除每块MS公共补偿相位不唯一的问题。
    fixed_variables = (0, Columns)

    # ---------- 3. 为角度扫描结果分配列表 ----------
    power_known_dBm, power_ce_dBm = [], []
    CE_Optimal_Matrices_MS1, CE_Optimal_Matrices_MS2 = [], []
    test_data = None
    test_h12 = None
    controlled_noise_power_dBm = None

    # 以“理想正侧向、双端完全匹配”的接收信号功率为1，按照指定SNR设置固定复AWGN功率。
    noise_power_normalized = 10 ** (-SNR_dB / 10)

    # ---------- 4. 逐个方位角建立信道并运行两种方案 ----------
    for angle_deg in angles_deg:
        angle_rad = np.deg2rad(angle_deg)
        h12, a1, a2, alpha = build_far_field_channel(angle_rad)

        # 距离固定时alpha不随角度改变；该噪声底使理想0°链路的SNR恰好等于SNR_dB。
        if controlled_noise_power_dBm is None:
            controlled_noise_power_dBm = Base_Power_dBm + 10 * np.log10(np.abs(alpha) ** 2) - SNR_dB

        # 已知角度方案：按“理想补偿相位→归一化→2-bit量化”的顺序计算两块板的16列编码。
        ideal_compensation_phase1, quantized_compensation_phase1, Known_Angle_Matrix_MS1 = calculate_2bit_compensation_code(angle_deg)

        # 两块MS的列编号和直流偏置设计一致，因此已知角度时直接使用相同编码矩阵。
        ideal_compensation_phase2 = ideal_compensation_phase1.copy()
        quantized_compensation_phase2 = quantized_compensation_phase1.copy()
        Known_Angle_Matrix_MS2 = Known_Angle_Matrix_MS1.copy()
        Known_Angle_v1 = Compensation_Phasors[Known_Angle_Matrix_MS1]
        Known_Angle_v2 = Compensation_Phasors[Known_Angle_Matrix_MS2]

        # ---------- 5. 未知CSI盲CE：从均匀概率开始 ----------
        probability = np.full((variable_count, state_count), 1 / state_count)
        for fixed in fixed_variables:
            probability[fixed] = [1.0, 0.0, 0.0, 0.0]

        # incumbent保存含噪测量意义下截至当前最好的联合码本。
        incumbent = np.zeros(variable_count, dtype=int)
        incumbent_score = -np.inf
        measured_history, confidence_history = [], []

        # ---------- 6. CE迭代：采样 → 测量 → Elite → 更新概率 ----------
        for iteration in range(max_iterations):
            # 每个变量都按照自己的一行概率独立抽取补偿相位状态。
            samples = np.empty((population_size, variable_count), dtype=int)
            for variable in range(variable_count):
                samples[:, variable] = rng.choice(state_count, population_size, p=probability[variable])

            # 保留历史最优解和当前概率众数，避免优秀码本在随机采样中丢失。
            if np.isfinite(incumbent_score):
                samples[0] = incumbent
            samples[1] = np.argmax(probability, axis=1)
            samples[:, fixed_variables] = 0

            # 将每组32维状态拆成MS1和MS2各16列的复补偿相位向量。
            v1_batch = Compensation_Phasors[samples[:, :Columns]]
            v2_batch = Compensation_Phasors[samples[:, Columns:]]

            # 一次性计算72个候选的v2^H*H12*v1，不单独定义评价函数。
            h_eff = np.einsum("mi,ij,mj->m", np.conj(v2_batch), h12, v1_batch, optimize=True)
            ideal_power = np.abs(alpha) ** 2 * Columns**4
            beam_matching = np.clip(np.abs(h_eff) ** 2 / ideal_power, 1e-5, 1.0)

            # 先计算无噪理论功率，再将复AWGN加到归一化接收场上，模拟y=hx+n后测得的总功率。
            # Tian等公式(4)把cos(theta)^0.8定义为单元场方向图；两端均取模平方后，总功率因子为cos(theta)^(4*0.8)。
            scan_product = np.cos(angle_rad) ** (4 * Element_Field_Exponent)
            theoretical_scores = Base_Power_dBm + 10 * np.log10(np.abs(alpha) ** 2 * beam_matching * scan_product)
            normalized_signal = h_eff / np.sqrt(ideal_power) * np.sqrt(scan_product)
            complex_noise = np.sqrt(noise_power_normalized / 2) * (
                rng.normal(size=population_size) + 1j * rng.normal(size=population_size)
            )
            noisy_power_factor = np.abs(normalized_signal + complex_noise) ** 2
            measured_scores = Base_Power_dBm + 10 * np.log10(np.abs(alpha) ** 2 * np.maximum(noisy_power_factor, 1e-12))

            # 如果本代出现更高的含噪测量值，就更新历史最优码本。
            best_index = int(np.argmax(measured_scores))
            if measured_scores[best_index] > incumbent_score:
                incumbent_score = float(measured_scores[best_index])
                incumbent = samples[best_index].copy()

            # 取功率最高的前15%样本，并统计每个变量中四种补偿相位的出现频率。
            elite_count = max(2, int(np.ceil(elite_fraction * population_size)))
            elite_samples = samples[np.argsort(measured_scores)[-elite_count:]]
            elite_probability = np.column_stack([(elite_samples == state).mean(axis=0) for state in range(state_count)])

            # 用平滑系数更新概率，同时保留最小探索概率，防止过早锁死。
            probability = (1 - smoothing) * probability + smoothing * elite_probability
            probability = np.maximum(probability, minimum_probability)
            probability /= probability.sum(axis=1, keepdims=True)
            for fixed in fixed_variables:
                probability[fixed] = [1.0, 0.0, 0.0, 0.0]

            # 记录CE真正看到的含噪历史最优与当前平均置信度。
            measured_history.append(incumbent_score)
            mask = np.ones(variable_count, dtype=bool)
            mask[list(fixed_variables)] = False
            confidence_history.append(np.max(probability[mask], axis=1).mean())

            # 当全部非固定变量的最大概率都超过阈值时提前停止。
            if np.all(np.max(probability[mask], axis=1) >= convergence_probability):
                break

        # ---------- 7. 最终复测：比较历史最优与概率众数 ----------
        mode = np.argmax(probability, axis=1)
        mode[list(fixed_variables)] = 0
        final_candidates = np.stack([incumbent, mode])
        final_means = np.zeros(2)

        # 重复测量8次后取平均，降低单次AWGN实现造成的“虚假最优”。
        for _ in range(final_verification_repeats):
            candidate_v1 = Compensation_Phasors[final_candidates[:, :Columns]]
            candidate_v2 = Compensation_Phasors[final_candidates[:, Columns:]]
            candidate_h = np.einsum("mi,ij,mj->m", np.conj(candidate_v2), h12, candidate_v1, optimize=True)
            candidate_signal = candidate_h / np.sqrt(ideal_power) * np.sqrt(scan_product)
            candidate_noise = np.sqrt(noise_power_normalized / 2) * (
                rng.normal(size=2) + 1j * rng.normal(size=2)
            )
            candidate_noisy_power = np.abs(candidate_signal + candidate_noise) ** 2
            final_means += Base_Power_dBm + 10 * np.log10(np.abs(alpha) ** 2 * np.maximum(candidate_noisy_power, 1e-12))

        # 选择重复测量均值更高的最终码本。
        best_indices = final_candidates[int(np.argmax(final_means / final_verification_repeats))]
        CE_Optimal_Matrix_MS1, CE_Optimal_Matrix_MS2 = best_indices[:Columns], best_indices[Columns:]
        CE_v1 = Compensation_Phasors[CE_Optimal_Matrix_MS1]
        CE_v2 = Compensation_Phasors[CE_Optimal_Matrix_MS2]

        # ---------- 8. 用统一链路预算计算两种方案的最终理论接收功率 ----------
        power_known_dBm.append(received_power_dBm(Known_Angle_v1, Known_Angle_v2, angle_rad, h12, alpha))
        power_ce_dBm.append(received_power_dBm(CE_v1, CE_v2, angle_rad, h12, alpha))
        CE_Optimal_Matrices_MS1.append(CE_Optimal_Matrix_MS1.copy())
        CE_Optimal_Matrices_MS2.append(CE_Optimal_Matrix_MS2.copy())

        # 保存测试角度的信道、码本与CE过程，供后续三张图使用。
        if np.isclose(angle_deg, test_angle_deg):
            test_h12 = h12.copy()
            test_data = {
                "angle_deg": float(angle_deg),
                "Known_Angle_Matrix_MS1": Known_Angle_Matrix_MS1.copy(),
                "Known_Angle_Matrix_MS2": Known_Angle_Matrix_MS2.copy(),
                "ideal_compensation_phase1": ideal_compensation_phase1.copy(),
                "ideal_compensation_phase2": ideal_compensation_phase2.copy(),
                "quantized_compensation_phase1": quantized_compensation_phase1.copy(),
                "quantized_compensation_phase2": quantized_compensation_phase2.copy(),
                "CE_Optimal_Matrix_MS1": CE_Optimal_Matrix_MS1.copy(),
                "CE_Optimal_Matrix_MS2": CE_Optimal_Matrix_MS2.copy(),
                "measured_history_dBm": np.asarray(measured_history),
                "confidence_history": np.asarray(confidence_history),
                "final_probability": probability.copy(),
                "SNR_dB": SNR_dB,
            }

    # ---------- 9. 整理绘图需要的数组 ----------
    power_known_dBm = np.asarray(power_known_dBm)
    power_ce_dBm = np.asarray(power_ce_dBm)
    results = {
        "angles_deg": angles_deg,
        "power_known_dBm": power_known_dBm,
        "power_ce_dBm": power_ce_dBm,
        "snr_known_dB": power_known_dBm - controlled_noise_power_dBm,
        "snr_ce_dB": power_ce_dBm - controlled_noise_power_dBm,
        "controlled_noise_power_dBm": controlled_noise_power_dBm,
        "SNR_dB": SNR_dB,
        "test": test_data,
    }

    # ---------- 10. 打印最重要的派生量，便于核对实验条件 ----------
    print(f"Wavelength: {Lambda * 1e3:.3f} mm")
    print(f"Unit period / wavelength: {Period_MS / Lambda:.3f}")
    print(f"Aperture width: {Aperture_Width_MS:.3f} m")
    print(f"Fraunhofer distance: {Far_Field_Distance_M:.3f} m")
    print(f"UAV separation: {Separation_Distance_M:.3f} m")
    print(f"Configured broadside-reference SNR: {SNR_dB:.1f} dB")
    print(f"Controlled AWGN floor: {controlled_noise_power_dBm:.3f} dBm")
    print(f"Thermal-noise reference from bandwidth and NF: {Noise_Power_dBm:.3f} dBm")

    # ---------- 11. 各模块直接绘制自己负责的数据 ----------
    plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    plot_link_results(results)
    plot_channel_at_test_angle(test_h12, test_angle_deg)
    plot_ce_coding_matrices(angles_deg, np.asarray(CE_Optimal_Matrices_MS1), np.asarray(CE_Optimal_Matrices_MS2))
    plot_patterns(test_data)

    # CE是主程序核心，因此最后一张CE概率图也直接在main()中绘制。
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    iteration = np.arange(1, test_data["measured_history_dBm"].size + 1)
    axes[0].plot(iteration, test_data["measured_history_dBm"], "-o", ms=3.5, color="#d95f02")
    axes[0].set(xlabel="CE iteration", ylabel="Noisy incumbent power (dBm)", title="(a) Measured CE history")
    axes[1].plot(iteration, test_data["confidence_history"], "-o", ms=3.5, color="#7570b3")
    axes[1].axhline(convergence_probability, color="0.35", linestyle="--", label="Threshold")
    axes[1].set(xlabel="CE iteration", ylabel="Mean maximum probability", ylim=(0.2, 1.02),
                title="(b) CE probability confidence")
    axes[1].legend(loc="best")
    figure.tight_layout()

    # 阻塞式显示会一直保留全部图窗，手动关闭所有图窗后程序结束。
    plt.show()


if __name__ == "__main__":
    main()
