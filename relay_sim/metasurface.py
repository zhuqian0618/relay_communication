"""计算2-bit列控超表面的相位编码，按照叠加定理计算远场方向图。"""

import matplotlib.pyplot as plt
import numpy as np

# 每块超表面由2行、16列物理单元组成；同一列的两个单元使用相同补偿相位。
Rows, Columns = 2, 16

# 单元周期固定为25 mm，既是单元尺寸，也是相邻受控列的中心间距。
Period_MS = 25e-3

# 以阵面中心为坐标原点，建立16列物理单元的中心坐标。
Column_Positions_MS = (np.arange(Columns) - (Columns - 1) / 2) * Period_MS

# 2-bit补偿相位共有四种状态；复数形式依次为1、j、-1、-j。
Compensation_Phase_States_Rad = np.deg2rad([0.0, 90.0, 180.0, 270.0])
Compensation_Phasors = np.exp(1j * Compensation_Phase_States_Rad)

# 当前用cos(theta)^q近似“功率”方向图，因此q=1表示功率近似cos(theta)。
Element_Pattern_Exponent = 0.8

# 方向图只显示水平面，绘制范围为-90°至90°。
Pattern_Angles_Deg = np.arange(-90.0, 90.01, 0.1)
Pattern_Floor_dB = -50.0


def calculate_2bit_compensation_code(Target_Angle_Deg: float, Lambda: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """给定偏折角，依次计算16列理想补偿相位、2-bit补偿相位和编码编号。"""

    # 自由空间相位常数β0=2π/λ；目标角度先由度转换为弧度。
    Beta0 = 2 * np.pi / Lambda
    Target_Angle_Rad = np.deg2rad(Target_Angle_Deg)

    # 采用e^(jωt)约定，第n列抵消空间相位所需的理想补偿相位为φn=-β0*xn*sin(θ0)。
    Ideal_Compensation_Phase_Rad = -Beta0 * Column_Positions_MS * np.sin(Target_Angle_Rad)

    # 与MATLAB的mod(phi_comp,2*pi)一致，把理想补偿相位归一化到[0,2π)。
    Normalized_Compensation_Phase_Rad = np.mod(Ideal_Compensation_Phase_Rad, 2 * np.pi)

    # 以45°、135°、225°、315°为分界，量化到最近的0°、90°、180°、270°。Code_Indices(0,...,3)对应rad(0,...,3pi/2)
    Code_Indices = np.floor((Normalized_Compensation_Phase_Rad + np.pi / 4) / (np.pi / 2)).astype(int) % 4
    Quantized_Compensation_Phase_Rad = Compensation_Phase_States_Rad[Code_Indices]

    return Ideal_Compensation_Phase_Rad, Quantized_Compensation_Phase_Rad, Code_Indices


def direction_pattern_dB(Code_Indices: np.ndarray, Lambda: float) -> np.ndarray:
    """按照叠加定理逐单元累加远场，并以理想0°波束为0 dB参考。"""

    # 同一列两行使用相同量化补偿相位；该矩阵与实际2×16直流偏置分布对应。
    Quantized_Compensation_Phase_Rad = Compensation_Phase_States_Rad[np.asarray(Code_Indices, dtype=int)]
    Compensation_Phase_Matrix_Rad = np.tile(Quantized_Compensation_Phase_Rad, (Rows, 1))

    # 在-90°至90°逐角度计算远场；Fields保存每个方向的复电场。
    Beta0 = 2 * np.pi / Lambda
    Fields = np.zeros(Pattern_Angles_Deg.size, dtype=complex)

    # 逐个观察角、逐行、逐列累加exp[j(空间传播相位+补偿相位)]，直接体现叠加定理。
    for Angle_Index, Angle_Deg in enumerate(Pattern_Angles_Deg):
        Angle_Rad = np.deg2rad(Angle_Deg)
        for Row_Index in range(Rows):
            for Column_Index in range(Columns):
                Space_Phase_Rad = Beta0 * Column_Positions_MS[Column_Index] * np.sin(Angle_Rad)
                Compensation_Phase = Compensation_Phase_Matrix_Rad[Row_Index, Column_Index]
                Fields[Angle_Index] += np.exp(1j * (Compensation_Phase + Space_Phase_Rad))

    # 只计算-90°至90°前向空间，该范围内cos(θ)非负，单元功率方向图直接取cos(θ)^q。
    Element_Power = np.cos(np.deg2rad(Pattern_Angles_Deg)) ** Element_Pattern_Exponent
    Relative_Power = np.abs(Fields) ** 2 / (Rows * Columns) ** 2 * Element_Power
    return np.maximum(10 * np.log10(np.maximum(Relative_Power, 1e-300)), Pattern_Floor_dB)


def plot_codebooks(angles_deg: np.ndarray, codes1: np.ndarray, codes2: np.ndarray) -> None:
    """绘制两块超表面随UAV2方位角变化的几何2-bit码本。"""

    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    for ax, codes, title in [
        (axes[0], codes1, "RIS1 geometry-based 2-bit compensation codebook"),
        (axes[1], codes2, "RIS2 geometry-based 2-bit compensation codebook"),
    ]:
        image = ax.imshow(codes, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=3,
                          extent=[0.5, Columns + 0.5, angles_deg[0], angles_deg[-1]])
        ax.set_xlabel("Column index")
        ax.set_title(title)
        colorbar = figure.colorbar(image, ax=ax, ticks=[0, 1, 2, 3])
        colorbar.ax.set_yticklabels(["0°", "90°", "180°", "270°"])

    axes[0].set_ylabel("UAV2 azimuth angle psi (deg)")
    figure.tight_layout()


def plot_patterns(diagnostic: dict, Lambda: float) -> None:
    """绘制诊断角度下的两端方向图和CE最终2×16补偿相位热力图。"""

    # 分别计算已知角度码本与盲CE码本的方向图。
    p1_known = direction_pattern_dB(diagnostic["geometric_indices1"], Lambda)
    p2_known = direction_pattern_dB(diagnostic["geometric_indices2"], Lambda)
    p1_ce = direction_pattern_dB(diagnostic["ce_indices1"], Lambda)
    p2_ce = direction_pattern_dB(diagnostic["ce_indices2"], Lambda)

    figure, axes = plt.subplots(2, 2, figsize=(8.6, 5.6), gridspec_kw={"height_ratios": [1.0, 0.55]})

    # 上排分别显示RIS1与RIS2的水平面方向图。
    for ax, known, ce, title in [
        (axes[0, 0], p1_known, p1_ce, "(a) RIS1 transmit pattern"),
        (axes[0, 1], p2_known, p2_ce, "(b) RIS2 receive pattern"),
    ]:
        ax.plot(Pattern_Angles_Deg, known, "--", color="#1b9e77", label="Known-angle 2-bit")
        ax.plot(Pattern_Angles_Deg, ce, "-", color="#7570b3", label="Blind CE")
        ax.axvline(diagnostic["angle_deg"], color="#d95f02", linestyle="-.", label="Target direction")
        ax.set(xlim=(-90, 90), ylim=(Pattern_Floor_dB, 1), xlabel="Local azimuth (deg)",
               ylabel="Gain relative to broadside (dB)", title=title)
        ax.legend(loc="upper right")

    # 下排把16列状态复制为2行，直观显示实际2×16列控偏置分布。
    for ax, indices, title in [
        (axes[1, 0], diagnostic["ce_indices1"], "(c) RIS1 CE compensation-phase map"),
        (axes[1, 1], diagnostic["ce_indices2"], "(d) RIS2 CE compensation-phase map"),
    ]:
        image = ax.imshow(np.tile(indices, (Rows, 1)), aspect="auto", cmap="viridis",
                          vmin=-0.5, vmax=3.5, interpolation="nearest")
        ax.set(xlabel="Column index", ylabel="Row index", title=title)
        ax.set_xticks(np.arange(Columns), np.arange(1, Columns + 1))
        ax.set_yticks(np.arange(Rows), np.arange(1, Rows + 1))
        colorbar = figure.colorbar(image, ax=ax, ticks=[0, 1, 2, 3], fraction=0.045, pad=0.03)
        colorbar.ax.set_yticklabels(["0°", "90°", "180°", "270°"])

    figure.suptitle(f"Patterns and CE compensation-phase maps at psi={diagnostic['angle_deg']:.0f}°", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
