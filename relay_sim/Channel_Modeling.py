"""统一建立远场空中信道、等效信道、链路预算、接收功率和理论SNR。"""

import matplotlib.pyplot as plt
import numpy as np

from .MS_Configuration import Beta0, Column_Positions_MS, Columns, Element_Field_Exponent, Lambda, Period_MS

# 两块超表面中心间距固定为6.5 m；孔径宽度和夫琅禾费距离用于检查远场条件。
Separation_Distance_M = 6.5
Aperture_Width_MS = Columns * Period_MS
Far_Field_Distance_M = 2 * Aperture_Width_MS**2 / Lambda


def build_far_field_channel(angle_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, complex]:
    """构造同一水平面、固定距离、单径LoS条件下的16×16远场信道矩阵。"""

    # 两块MS列编号与朝向定义一致，因此给定同一局部角度时使用相同空间响应向量。
    Space_Phase_Rad = Beta0 * Column_Positions_MS * np.sin(angle_rad)
    a1 = np.exp(-1j * Space_Phase_Rad)
    a2 = np.exp(-1j * Space_Phase_Rad)

    # alpha包含Friis场衰减lambda/(4*pi*R)以及传播距离引起的公共相位。
    alpha = Lambda / (4 * np.pi * Separation_Distance_M)
    alpha *= np.exp(-1j * Beta0 * Separation_Distance_M)

    # 远场单径LoS信道为接收空间响应和发射空间响应共轭转置的外积，因此理论秩为1。
    h12 = alpha * np.outer(a2, np.conj(a1))
    return h12, a1, a2, complex(alpha)

# 发射功率只在统一模型y=sqrt(Pt)*h_eff*s+n中出现一次。
Transmit_Power_dBm = 15
Transmit_Power_W = 10 ** ((Transmit_Power_dBm - 30) / 10)
MS1_Broadside_Gain_dBi = 15.0
MS2_Broadside_Gain_dBi = 15.0
Total_RoF_Gain_dB = -20.0
Receiver_Misc_Gain_dB = -3

# 固定增益不包含发射功率和Friis传播系数；功率增益开平方后才能乘到复信道场上。
Fixed_Link_Power_Gain_dB = MS1_Broadside_Gain_dBi + MS2_Broadside_Gain_dBi + Total_RoF_Gain_dB + Receiver_Misc_Gain_dB
Fixed_Link_Field_Gain = np.sqrt(10 ** (Fixed_Link_Power_Gain_dB / 10))

# 直接给定接收端总噪声功率sigma²；后续生成训练数据时只需修改该变量。
Noise_Power_dBm = -90.0
Noise_Power_W = 10 ** ((Noise_Power_dBm - 30) / 10)


def link_metrics(v1: np.ndarray, v2: np.ndarray, angle_rad: float,
                 h12: np.ndarray) -> tuple[complex, float, float]:
    """按照y=sqrt(Pt)*h_eff*s+n返回等效信道、无噪接收功率和理论SNR。"""

    # v2^H*H12*v1包含双端列阵相干叠加和H12中的Friis复系数。
    Air_Channel = np.vdot(v2, h12 @ v1)

    # 除以N1*N2，把列阵相干增益归一化；阵面的实际宽边增益已由上面的dBi参数给出。
    Normalized_Air_Channel = Air_Channel / Columns**2

    # 单块MS的单元场方向图为cos(theta)^q；双端场系数相乘后为cos(theta)^(2q)。
    Scan_Field_Factor = np.cos(angle_rad) ** (2 * Element_Field_Exponent)

    # 统一等效信道包含固定RoF/天线增益、空中Friis传播、双端相位匹配和扫描损耗。
    h_eff = Fixed_Link_Field_Gain * Normalized_Air_Channel * Scan_Field_Factor

    # s已归一化为E[|s|²]=1，因此无噪信号功率为Pt*|h_eff|²，理论SNR为该功率除以噪声方差。
    Signal_Power_W = Transmit_Power_W * np.abs(h_eff) ** 2
    Theoretical_SNR_Linear = Signal_Power_W / Noise_Power_W
    return complex(h_eff), float(Signal_Power_W), float(Theoretical_SNR_Linear)


def plot_link_results(results: dict) -> None:
    """绘制轨迹、无噪接收信号功率、理论SNR和CE估计SNR历史。"""

    angles = results["angles_deg"]
    radius = Separation_Distance_M
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 6.0))

    # 图(a)：UAV2在以UAV1为圆心、半径6.5 m的圆弧上运动。
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

    # 图(c)：理论SNR严格按Pt*|h_eff|²/sigma²计算，而不是由单次含噪观测计算。
    axes[1, 0].plot(angles, results["snr_known_dB"], "--o", ms=3.5, color="#1b9e77")
    axes[1, 0].plot(angles, results["snr_ce_dB"], "-o", ms=3.5, color="#7570b3")
    axes[1, 0].set(xlabel="UAV2 azimuth psi (deg)", ylabel="SNR (dB)",
                   title=f"(c) Theoretical SNR (noise={results['noise_power_dBm']:.1f} dBm)")

    # 图(d)：CE用多枚含噪导频估计SNR；移动平均只帮助观察趋势，不参与优化。
    measured = np.asarray(results["test"]["estimated_snr_history_dB"])
    iteration = np.arange(1, measured.size + 1)
    moving_average = np.convolve(measured, np.ones(3) / 3, mode="valid")
    axes[1, 1].plot(iteration, measured, "-o", ms=3.5, color="#d95f02", label="Estimated-SNR incumbent")
    axes[1, 1].plot(iteration[2:], moving_average, color="#1b9e77", lw=2, label="3-point average")
    axes[1, 1].set(xlabel="CE iteration", ylabel="Estimated SNR (dB)",
                   title=f"(d) Pilot-based CE history at test angle psi={results['test']['angle_deg']:.0f}°")
    axes[1, 1].legend(loc="best")

    figure.suptitle("Dual UAV-borne 2-bit metasurface far-field link", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
