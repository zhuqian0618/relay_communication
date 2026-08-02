"""RoF链路、接收功率、噪声以及总体结果绘图。"""

import matplotlib.pyplot as plt
import numpy as np

from .channel import Separation_Distance_M
from .metasurface import Columns, Element_Field_Exponent

# 本文件只保存链路预算所需参数；dB量在计算时直接转成线性值，不再调用单位转换函数。
Transmit_Power_dBm = 0
MS1_Broadside_Gain_dBi = 15.0
MS2_Broadside_Gain_dBi = 15.0
Total_RoF_Gain_dB = -20.0
Receiver_Misc_Gain_dB = -3
Base_Power_dBm = Transmit_Power_dBm + MS1_Broadside_Gain_dBi + MS2_Broadside_Gain_dBi + Total_RoF_Gain_dB + Receiver_Misc_Gain_dB

# 接收机噪声由Pn=-174+10log10(B)+NF计算；SNR不再作为输入，而由接收功率减去噪声功率得到。
Receiver_Bandwidth_Hz = 20e6
Receiver_Noise_Figure_dB = 7.0
Noise_Power_dBm = -174.0 + 10 * np.log10(Receiver_Bandwidth_Hz) + Receiver_Noise_Figure_dB


def received_power_dBm(v1: np.ndarray, v2: np.ndarray, angle_rad: float,
                       h12: np.ndarray, alpha: complex) -> float:
    """根据双端补偿相位匹配、扫描损耗、Friis损耗和RoF损耗计算接收功率。"""

    # v2^H*H12*v1是两块超表面之间的等效复信道。
    h_eff = np.vdot(v2, h12 @ v1)

    # 连续补偿相位完全匹配时，等效功率为|alpha|²*N1²*N2²；二者之比限制在0至1。
    ideal_power = np.abs(alpha) ** 2 * Columns**4
    beam_matching = np.clip(np.abs(h_eff) ** 2 / ideal_power, 1e-5, 1.0)

    # 只考虑-90°至90°。每块超表面的场因子为cos(theta)^q，转换成功率后为cos(theta)^(2q)；
    # 收发两块超表面的功率因子相乘，因此双端链路总扫描因子为cos(theta)^(4q)。
    scan_product = np.cos(angle_rad) ** (4 * Element_Field_Exponent)

    # dB增益直接相加等价于线性增益相乘；最后仍以dBm输出。
    return float(Base_Power_dBm + 10 * np.log10(np.abs(alpha) ** 2 * beam_matching * scan_product))


def plot_link_results(results: dict) -> None:
    """绘制轨迹、接收功率、SNR和含噪CE历史最优。"""

    angles = results["angles_deg"]
    radius = Separation_Distance_M
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 6.0))

    # 图(a)：UAV2在以UAV1为圆心、半径10 m的圆弧上运动。
    full_circle = np.linspace(0, 2 * np.pi, 400)
    axes[0, 0].plot(radius * np.cos(full_circle), radius * np.sin(full_circle), "--", color="0.75")
    axes[0, 0].plot(radius * np.cos(np.deg2rad(angles)), radius * np.sin(np.deg2rad(angles)),
                    "o-", color="#d95f02", label="UAV2 trajectory")
    axes[0, 0].scatter([0], [0], s=80, marker="s", color="#1b9e77", label="UAV1")
    axes[0, 0].set(xlabel="x position (m)", ylabel="y position (m)",
                   title="(a) Constant-distance trajectory", aspect="equal")
    axes[0, 0].legend(loc="best")

    # 图(b)：两条曲线分别对应已知角度2-bit码本和未知CSI盲CE。
    axes[0, 1].plot(angles, results["power_known_dBm"], "--o", ms=3.5,
                    color="#1b9e77", label="Known-angle optimized 2-bit")
    axes[0, 1].plot(angles, results["power_ce_dBm"], "-o", ms=3.5,
                    color="#7570b3", label="Unknown-CSI blind CE")
    axes[0, 1].set(xlabel="UAV2 azimuth psi (deg)", ylabel="Received power (dBm)",
                   title="(b) Aerial-link received power")
    axes[0, 1].legend(loc="best")

    # 图(c)：SNR等于无噪接收信号功率减去固定的物理接收机噪声功率。
    axes[1, 0].plot(angles, results["snr_known_dB"], "--o", ms=3.5, color="#1b9e77")
    axes[1, 0].plot(angles, results["snr_ce_dB"], "-o", ms=3.5, color="#7570b3")
    axes[1, 0].set(xlabel="UAV2 azimuth psi (deg)", ylabel="SNR (dB)",
                   title=f"(c) Derived SNR (physical noise={results['noise_power_dBm']:.1f} dBm)")

    # 图(d)：只显示CE实际使用的含噪历史最优，并用三点移动平均帮助观察趋势。
    measured = np.asarray(results["test"]["measured_history_dBm"])
    iteration = np.arange(1, measured.size + 1)
    moving_average = np.convolve(measured, np.ones(3) / 3, mode="valid")
    axes[1, 1].plot(iteration, measured, "-o", ms=3.5, color="#d95f02", label="Noisy incumbent")
    axes[1, 1].plot(iteration[2:], moving_average, color="#1b9e77", lw=2, label="3-point average")
    axes[1, 1].set(xlabel="CE iteration", ylabel="Measured power (dBm)",
                   title=f"(d) Noisy CE history at test angle psi={results['test']['angle_deg']:.0f}°")
    axes[1, 1].legend(loc="best")

    figure.suptitle("Dual UAV-borne 2-bit metasurface far-field link", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
