"""统一建立远场空中信道、等效信道、链路预算、接收功率和理论SNR。"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
except ImportError:  # Core channel calculations remain usable without plotting.
    plt = None
    MultipleLocator = None

from .MS_Configuration import Beta0, Column_Positions_MS, Columns, Element_Field_Exponent, Lambda, Period_MS, Phase_State_Cmap

# 两块超表面中心间距固定为6.5 m；孔径宽度和夫琅禾费距离用于检查远场条件。
Separation_Distance_M = 6.5
Aperture_Width_MS = Columns * Period_MS
Far_Field_Distance_M = 2 * Aperture_Width_MS**2 / Lambda

# 发射功率只在统一模型y=sqrt(Pt)*h_eff*s+n中出现一次。
Transmit_Power_dBm = 0.0
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

# 图1(b)双纵轴的显示范围和主刻度间隔；修改这四个参数即可自行控制坐标轴。
Received_Power_Y_Limits_dBm = (-70.0, -45.0)
Received_Power_Y_Interval_dB = 5.0
SNR_Y_Limits_dB = (10.0, 35.0)
SNR_Y_Interval_dB = 5.0


def build_far_field_channel(
    angle_rad: float,
    separation_distance_M: float = Separation_Distance_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, complex]:
    """构造同一水平面、固定距离、单径LoS条件下的16×16远场信道矩阵。"""

    if separation_distance_M <= 0.0:
        raise ValueError("separation_distance_M must be positive")

    # 两块MS列编号与朝向定义一致，因此给定同一局部角度时使用相同空间响应向量。
    Space_Phase_Rad = Beta0 * Column_Positions_MS * np.sin(angle_rad)
    a1 = np.exp(-1j * Space_Phase_Rad)
    a2 = np.exp(-1j * Space_Phase_Rad)

    # alpha包含Friis场衰减lambda/(4*pi*R)以及传播距离引起的公共相位。
    alpha = Lambda / (4 * np.pi * separation_distance_M)
    alpha *= np.exp(-1j * Beta0 * separation_distance_M)

    # 远场单径LoS信道为接收空间响应和发射空间响应共轭转置的外积，因此理论秩为1。
    h12 = alpha * np.outer(a2, np.conj(a1))
    return h12, a1, a2, complex(alpha)


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


def plot_link_results(results: dict, CE_Matrices_MS1: np.ndarray, CE_Matrices_MS2: np.ndarray) -> None:
    """Figure 1：用2×2大图汇总轨迹、含噪链路测量和两块MS的CE编码矩阵。"""

    if plt is None or MultipleLocator is None:
        raise RuntimeError("plot_link_results requires matplotlib")

    angles = results["angles_deg"]
    radius = Separation_Distance_M
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))

    # 图(a)：UAV2在以UAV1为圆心、半径6.5 m的圆弧上运动。
    full_circle = np.linspace(0, 2 * np.pi, 400)
    axes[0, 0].plot(radius * np.cos(full_circle), radius * np.sin(full_circle), "--", color="#C7D3DD")
    axes[0, 0].plot(radius * np.cos(np.deg2rad(angles)), radius * np.sin(np.deg2rad(angles)),
                    "o-", color="#E07A5F", label="UAV2 trajectory")
    axes[0, 0].scatter([0], [0], s=80, marker="s", color="#2A9D8F", label="UAV1")
    axes[0, 0].set(xlabel="x position (m)", ylabel="y position (m)",
                   title="(a) Constant-distance trajectory", aspect="equal")
    axes[0, 0].legend(loc="upper left")

    # 图(b)：左轴为L导频测得的含噪总接收功率，右轴为由同一批导频计算的标准SNR估计。
    snr_axis = axes[0, 1].twinx()
    power_line = axes[0, 1].plot(angles, results["noisy_power_ce_dBm"], "-o", ms=4,
                                 color="#4C78A8", label="Noisy received power")[0]
    snr_line = snr_axis.plot(angles, results["estimated_snr_ce_dB"], "-s", ms=3.5,
                             color="#E76F51", label="Estimated SNR")[0]
    axes[0, 1].set(xlabel="UAV2 azimuth angle (deg)", ylabel="Noisy received power (dBm)",
                   title=f"(b) CE-optimized link measurements (L={results['pilot_symbols_L']})")
    snr_axis.set_ylabel("Estimated SNR (dB)", color="#E76F51")
    snr_axis.tick_params(axis="y", labelcolor="#E76F51")

    # 双纵轴范围和间隔完全由文件开头的四个参数控制，不再根据当前数据自动改变。
    axes[0, 1].set_ylim(*Received_Power_Y_Limits_dBm)
    snr_axis.set_ylim(*SNR_Y_Limits_dB)
    axes[0, 1].yaxis.set_major_locator(MultipleLocator(Received_Power_Y_Interval_dB))
    snr_axis.yaxis.set_major_locator(MultipleLocator(SNR_Y_Interval_dB))
    axes[0, 1].legend([power_line, snr_line], [power_line.get_label(), snr_line.get_label()], loc="lower right")

    # 下排分别显示MS1和MS2沿整条UAV轨迹得到的CE最优2-bit编码矩阵。
    Column_Edges = np.arange(0.5, Columns + 1.5)
    Angle_Edges = np.empty(angles.size + 1)
    Angle_Edges[1:-1] = (angles[:-1] + angles[1:]) / 2
    Angle_Edges[0] = angles[0] - (Angle_Edges[1] - angles[0])
    Angle_Edges[-1] = angles[-1] + (angles[-1] - Angle_Edges[-2])
    for ax, Coding_Matrices, title in [
        (axes[1, 0], CE_Matrices_MS1, "(c) MS1 compensation-phase map"),
        (axes[1, 1], CE_Matrices_MS2, "(d) MS2 compensation-phase map"),
    ]:
        image = ax.pcolormesh(Column_Edges, Angle_Edges, Coding_Matrices, cmap=Phase_State_Cmap,
                              vmin=-0.5, vmax=3.5, shading="flat", edgecolors="#C9CED6", linewidth=0.45)
        ax.set(xlabel="Column index", ylabel="UAV2 azimuth angle (deg)", title=title)
        ax.set_xticks([1, 4, 7, 10, 13, 16])
        ax.tick_params(axis="both", which="major", length=0)
        ax.grid(False, which="major")
        colorbar = figure.colorbar(image, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.03)
        colorbar.ax.set_yticklabels(["0°", "90°", "180°", "270°"])

    figure.suptitle("Part I:Overall CE-optimized aerial-link results", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
