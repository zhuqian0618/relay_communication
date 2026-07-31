"""计算2-bit列控超表面的相位编码，按照叠加定理计算远场方向图。"""

import matplotlib.pyplot as plt
import numpy as np

# 超表面工作频率为5.8 GHz；波长和自由空间相位常数在本模块统一定义，其他模块直接引用。
Speed_Of_Light_M_S = 299_792_458.0
Carrier_Frequency_Hz = 5.8e9
Lambda = Speed_Of_Light_M_S / Carrier_Frequency_Hz
Beta0 = 2 * np.pi / Lambda

# 每块超表面由2行、16列物理单元组成；同一列的两个单元使用相同补偿相位。
Rows, Columns = 2, 16

# 单元周期固定为25 mm，既是单元尺寸，也是相邻受控列的中心间距。
Period_MS = 25e-3

# 以阵面中心为坐标原点，建立16列物理单元的中心坐标。
Column_Positions_MS = (np.arange(Columns) - (Columns - 1) / 2) * Period_MS

# 2-bit补偿相位共有四种状态；复数形式依次为1、j、-1、-j。
Compensation_Phase_States_Rad = np.deg2rad([0.0, 90.0, 180.0, 270.0])
Compensation_Phasors = np.exp(1j * Compensation_Phase_States_Rad)

# cos(theta)^q近似单元远场“场幅度”；本文取q=0.8。
Element_Field_Exponent = 0.8

# 方向图只显示水平面，绘制范围为-90°至90°。
Pattern_Angles_Deg = np.arange(-90.0, 90.01, 0.1)
Pattern_Floor_dB = -50.0


def calculate_2bit_compensation_code(Target_Angle_Deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """给定偏折角，依次计算16列理想补偿相位、2-bit补偿相位和编码矩阵。"""

    # 目标角度由度转换为弧度；Beta0已根据5.8 GHz工作频率在文件开头计算。
    Target_Angle_Rad = np.deg2rad(Target_Angle_Deg)

    # 采用e^(jωt)约定，第n列抵消空间相位所需的理想补偿相位为φn=-β0*xn*sin(θ0)。
    Ideal_Compensation_Phase_Rad = -Beta0 * Column_Positions_MS * np.sin(Target_Angle_Rad)

    # 与MATLAB的mod(phi_comp,2*pi)一致，把理想补偿相位归一化到[0,2π)。
    Normalized_Compensation_Phase_Rad = np.mod(Ideal_Compensation_Phase_Rad, 2 * np.pi)

    # 以45°、135°、225°、315°为分界，量化到最近的0°、90°、180°、270°。
    # Coding_Matrix中的状态0、1、2、3分别对应0°、90°、180°、270°补偿相位。
    Coding_Matrix = np.floor((Normalized_Compensation_Phase_Rad + np.pi / 4) / (np.pi / 2)).astype(int) % 4
    Quantized_Compensation_Phase_Rad = Compensation_Phase_States_Rad[Coding_Matrix]

    return Ideal_Compensation_Phase_Rad, Quantized_Compensation_Phase_Rad, Coding_Matrix


def direction_pattern_dB(Coding_Matrix: np.ndarray) -> np.ndarray:
    """按照叠加定理逐单元累加远场，并以理想0°波束为0 dB参考。"""

    # 同一列两行使用相同量化补偿相位；该矩阵与实际2×16直流偏置分布对应。
    Quantized_Compensation_Phase_Rad = Compensation_Phase_States_Rad[np.asarray(Coding_Matrix, dtype=int)]
    Compensation_Phase_Matrix_Rad = np.tile(Quantized_Compensation_Phase_Rad, (Rows, 1))

    # 在-90°至90°逐角度计算远场；Fields保存每个方向的复电场。
    Fields = np.zeros(Pattern_Angles_Deg.size, dtype=complex)

    # 逐个观察角、逐行、逐列累加exp[j(空间传播相位+补偿相位)]，直接体现叠加定理。
    for Angle_Index, Angle_Deg in enumerate(Pattern_Angles_Deg):
        Angle_Rad = np.deg2rad(Angle_Deg)
        for Row_Index in range(Rows):
            for Column_Index in range(Columns):
                Space_Phase_Rad = Beta0 * Column_Positions_MS[Column_Index] * np.sin(Angle_Rad)
                Compensation_Phase = Compensation_Phase_Matrix_Rad[Row_Index, Column_Index]
                Fields[Angle_Index] += np.exp(1j * (Compensation_Phase + Space_Phase_Rad))

    # 只计算-90°至90°前向空间，该范围内cos(θ)非负；先将单元场方向图乘到阵列复电场上。
    # 随后取模平方，因此单块超表面的单元功率因子自然成为cos(theta)^(2q)=cos(theta)^1.6。
    Element_Field = np.cos(np.deg2rad(Pattern_Angles_Deg)) ** Element_Field_Exponent
    Total_Fields = Fields * Element_Field
    Relative_Power = np.abs(Total_Fields) ** 2 / (Rows * Columns) ** 2
    Minimum_Power = 10 ** (Pattern_Floor_dB / 10)
    return 10 * np.log10(np.maximum(Relative_Power, Minimum_Power))


def plot_ce_coding_matrices(angles_deg: np.ndarray, CE_Optimal_Matrices_MS1: np.ndarray,
                            CE_Optimal_Matrices_MS2: np.ndarray) -> None:
    """绘制各UAV2方位角下由CE优化得到的两块超表面2-bit编码矩阵。"""

    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    for ax, Coding_Matrices, title in [
        (axes[0], CE_Optimal_Matrices_MS1, "(a) MS1 CE-optimal coding matrix"),
        (axes[1], CE_Optimal_Matrices_MS2, "(b) MS2 CE-optimal coding matrix"),
    ]:
        image = ax.imshow(Coding_Matrices, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=3,
                          extent=[0.5, Columns + 0.5, angles_deg[0], angles_deg[-1]])
        ax.set_xlabel("Column index")
        ax.set_title(title)
        colorbar = figure.colorbar(image, ax=ax, ticks=[0, 1, 2, 3])
        colorbar.ax.set_yticklabels(["0°", "90°", "180°", "270°"])

    axes[0].set_ylabel("UAV2 azimuth angle psi (deg)")
    figure.tight_layout()


def plot_patterns(test_data: dict) -> None:
    """绘制测试角度下的两端方向图和CE最终2×16补偿相位热力图。"""

    # 分别计算测试角度下已知角度码本与盲CE码本的方向图。
    MS1_Known_Pattern = direction_pattern_dB(test_data["Known_Angle_Matrix_MS1"])
    MS2_Known_Pattern = direction_pattern_dB(test_data["Known_Angle_Matrix_MS2"])
    MS1_CE_Pattern = direction_pattern_dB(test_data["CE_Optimal_Matrix_MS1"])
    MS2_CE_Pattern = direction_pattern_dB(test_data["CE_Optimal_Matrix_MS2"])

    figure, axes = plt.subplots(2, 2, figsize=(8.6, 5.6), gridspec_kw={"height_ratios": [1.0, 0.55]})

    # 上排分别显示MS1与MS2的水平面方向图。
    for ax, known, ce, title in [
        (axes[0, 0], MS1_Known_Pattern, MS1_CE_Pattern, "(a) MS1 transmit pattern"),
        (axes[0, 1], MS2_Known_Pattern, MS2_CE_Pattern, "(b) MS2 receive pattern"),
    ]:
        ax.plot(Pattern_Angles_Deg, known, "--", color="#1b9e77", label="Known-angle 2-bit")
        ax.plot(Pattern_Angles_Deg, ce, "-", color="#7570b3", label="Blind CE")
        ax.axvline(test_data["angle_deg"], color="#d95f02", linestyle="-.", label="Test direction")
        ax.set(xlim=(-90, 90), ylim=(Pattern_Floor_dB, 1), xlabel="Local azimuth (deg)",
               ylabel="Gain relative to broadside (dB)", title=title)
        ax.legend(loc="upper right")

    # 下排把16列状态复制为2行，直观显示实际2×16列控偏置分布。
    for ax, Coding_Matrix, title in [
        (axes[1, 0], test_data["CE_Optimal_Matrix_MS1"], "(c) MS1 CE compensation-phase map"),
        (axes[1, 1], test_data["CE_Optimal_Matrix_MS2"], "(d) MS2 CE compensation-phase map"),
    ]:
        image = ax.imshow(np.tile(Coding_Matrix, (Rows, 1)), aspect="auto", cmap="viridis",
                          vmin=-0.5, vmax=3.5, interpolation="nearest")
        ax.set(xlabel="Column index", ylabel="Row index", title=title)
        ax.set_xticks(np.arange(Columns), np.arange(1, Columns + 1))
        ax.set_yticks(np.arange(Rows), np.arange(1, Rows + 1))
        colorbar = figure.colorbar(image, ax=ax, ticks=[0, 1, 2, 3], fraction=0.045, pad=0.03)
        colorbar.ax.set_yticklabels(["0°", "90°", "180°", "270°"])

    figure.suptitle(f"Patterns and CE compensation-phase maps at test angle psi={test_data['angle_deg']:.0f}°", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
