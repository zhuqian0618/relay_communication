"""计算2-bit列控超表面的相位编码，按照叠加定理计算远场方向图。"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:  # Plotting is optional for headless experiment control.
    plt = None
    ListedColormap = None

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

# 四种2-bit相位采用清新的离散颜色；所有编码热力图和概率柱状图统一使用该配色。
Phase_State_Colors = ["#2A82C5", "#079D63", "#E46964", "#F8CC04"]
Phase_State_Cmap = (
    ListedColormap(Phase_State_Colors, name="Viridis_2bit_Phases")
    if ListedColormap is not None else None
)


def calculate_2bit_compensation_code(Target_Angle_Deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """给定偏折角，搜索最优公共相位并返回量化后的16列2-bit编码。"""

    # 目标角度由度转换为弧度；Beta0已根据5.8 GHz工作频率在文件开头计算。
    Target_Angle_Rad = np.deg2rad(Target_Angle_Deg)

    # 采用e^(jωt)约定，第n列抵消空间相位所需的理想补偿相位为φn=-β0*xn*sin(θ0)。
    Ideal_Compensation_Phase_Rad = -Beta0 * Column_Positions_MS * np.sin(Target_Angle_Rad)

    # 先保存公共相位为0时的直接量化编码，用于和公共相位优化结果比较。
    Direct_Normalized_Phase_Rad = np.mod(Ideal_Compensation_Phase_Rad, 2 * np.pi)
    Direct_Coding_Matrix = np.floor((Direct_Normalized_Phase_Rad + np.pi / 4) / (np.pi / 2)).astype(int) % 4

    # 连续相位整体增加同一个公共相位不会改变波束方向，但会改变2-bit量化误差。
    # 在一个90°量化周期内以0.25°间隔搜索公共相位，并同时生成所有候选编码矩阵。
    Common_Phase_Candidates_Rad = np.deg2rad(np.arange(0.0, 90.0, 0.25))
    Candidate_Phases_Rad = Ideal_Compensation_Phase_Rad[None, :] + Common_Phase_Candidates_Rad[:, None]
    Candidate_Coding_Matrices = np.floor((np.mod(Candidate_Phases_Rad, 2 * np.pi) + np.pi / 4) / (np.pi / 2)).astype(int) % 4

    # 对每个候选编码计算目标方向的相干叠加功率；误差相位越集中，叠加功率越大。
    Candidate_Phasors = Compensation_Phasors[Candidate_Coding_Matrices]
    Ideal_Phasors = np.exp(1j * Ideal_Compensation_Phase_Rad)
    Coherent_Powers = np.abs(np.sum(np.conj(Ideal_Phasors)[None, :] * Candidate_Phasors, axis=1)) ** 2
    Best_Common_Phase_Index = int(np.argmax(Coherent_Powers))
    Optimal_Common_Phase_Rad = float(Common_Phase_Candidates_Rad[Best_Common_Phase_Index])
    Coding_Matrix = Candidate_Coding_Matrices[Best_Common_Phase_Index]
    Quantized_Compensation_Phase_Rad = Compensation_Phase_States_Rad[Coding_Matrix]

    return Ideal_Compensation_Phase_Rad, Quantized_Compensation_Phase_Rad, Coding_Matrix, Optimal_Common_Phase_Rad, Direct_Coding_Matrix


def direction_pattern_dB(Coding_Matrix: np.ndarray) -> np.ndarray:
    """按照叠加定理累加2行16列单元的远场，并把当前曲线自身最大值归一化为0 dB。"""

    # 同一列两行使用相同量化补偿相位；该矩阵与实际2×16直流偏置分布对应。
    Quantized_Compensation_Phase_Rad = Compensation_Phase_States_Rad[np.asarray(Coding_Matrix, dtype=int)]
    Compensation_Phase_Matrix_Rad = np.tile(Quantized_Compensation_Phase_Rad, (Rows, 1))

    # 在-90°至90°逐角度计算远场；Fields保存每个方向的复电场。
    Fields = np.zeros(Pattern_Angles_Deg.size, dtype=complex)

    # 对每个观察角、每行和每列累加exp[j(空间传播相位+补偿相位)]。
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
    Pattern_Power = np.abs(Total_Fields) ** 2
    Relative_Power = Pattern_Power / np.max(Pattern_Power)
    Minimum_Power = 10 ** (Pattern_Floor_dB / 10)
    return 10 * np.log10(np.maximum(Relative_Power, Minimum_Power))


def plot_ce_iteration_evolution(test_data: dict) -> None:
    """每列显示一次CE迭代，并从上到下排列概率、双MS相位和方向图。"""

    if plt is None:
        raise RuntimeError("plot_ce_iteration_evolution requires matplotlib")

    snapshots = test_data["ce_iteration_snapshots"]
    if not snapshots:
        return

    # 三列依次对应第1次、用户指定的观察时刻和满足双重终止条件时的实际终止时刻。
    Snapshot_By_Iteration = {snapshot["iteration"]: snapshot for snapshot in snapshots}
    Selected_Snapshots = [Snapshot_By_Iteration[i] for i in test_data["selected_iterations"]]
    figure = plt.figure(figsize=(15.6, 8.3))
    grid = figure.add_gridspec(4, 3, height_ratios=[1.05, 0.36, 0.36, 1.0])
    figure.subplots_adjust(left=0.045, right=0.94, bottom=0.075, top=0.90, wspace=0.18, hspace=0.54)
    Column_Tick_Positions = np.array([0, 3, 6, 9, 12, 15])
    Column_Tick_Labels = ["1", "4", "7", "10", "13", "16"]
    Joint_Ticks = [1, 5, 9, 13, 17, 21, 25, 29, 32]
    Joint_Tick_Labels = [str(index) for index in Joint_Ticks]
    Panel_Letters = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]
    Known_Angle_Pattern = direction_pattern_dB(test_data["Known_Angle_Matrix_MS1"])
    Last_Phase_Image = None

    for column, snapshot in enumerate(Selected_Snapshots):
        iteration = snapshot["iteration"]
        probability = snapshot["probability"]
        Coding_MS1 = snapshot["Coding_Matrix_MS1"]
        Coding_MS2 = snapshot["Coding_Matrix_MS2"]
        Probability_Letter, Phase_Letter, Pattern_Letter = Panel_Letters[column]

        # 每列顶部：用3D柱状图展示32个联合变量选择四种2-bit相位状态的概率。
        Probability_Axis = figure.add_subplot(grid[0, column], projection="3d")
        X = np.repeat(np.arange(1, 2 * Columns + 1), 4)
        Y = np.tile(np.arange(4), 2 * Columns)
        Heights = probability.reshape(-1)
        Probability_Bars = Probability_Axis.bar3d(
            X - 0.34, Y - 0.21, np.zeros_like(Heights), 0.68, 0.42, Heights,
            color=np.tile(Phase_State_Colors, 2 * Columns), edgecolor="#C9CED6",
            linewidth=0.25, shade=False, alpha=0.96)
        # 放大三维坐标盒时，Matplotlib默认会按子图矩形裁掉边缘柱体；关闭柱体集合裁剪以完整显示首尾变量。
        Probability_Bars.set_clip_on(False)
        # 三维坐标范围固定覆盖32个联合变量、4种相位状态和0至1的概率。
        Probability_Axis.set(xlim=(0.3, 32.7), ylim=(-0.4, 3.7), zlim=(0, 1.02),
                             xlabel="", ylabel="",
                             title=f"({Probability_Letter}) Iteration {iteration}: probability distribution\n"
                                   f"Best estimated SNR={snapshot['estimated_snr_dB']:.2f} dB")
        Probability_Axis.set_zlabel("")
        Probability_Axis.tick_params(pad=0)
        Probability_Axis.set_xticks(Joint_Ticks, Joint_Tick_Labels, fontsize=10)
        Probability_Axis.set_yticks(np.arange(4), ["0°", "90°", "180°", "270°"], fontsize=8)
        Probability_Axis.set_zticks([0.0, 0.5, 1.0], ["0", "0.5", "1"], fontsize=10)
        Probability_Axis.view_init(elev=26, azim=-62)
        Probability_Axis.set_box_aspect((5, 1.0, 1.0), zoom=2)

        # 去掉3D坐标轴默认的灰色面板，只保留坐标轴和浅色网格线。
        Probability_Axis.set_facecolor("white")
        Probability_Axis.xaxis.pane.fill = False
        Probability_Axis.yaxis.pane.fill = False
        Probability_Axis.zaxis.pane.fill = False
        Probability_Axis.xaxis.pane.set_edgecolor((1, 1, 1, 0))
        Probability_Axis.yaxis.pane.set_edgecolor((1, 1, 1, 0))
        Probability_Axis.zaxis.pane.set_edgecolor((1, 1, 1, 0))

        # 中间两层：MS1和MS2分别使用一张较薄的2×16相位热力图。
        for phase_row, Coding_Matrix, MS_Name in [(1, Coding_MS1, "MS1"), (2, Coding_MS2, "MS2")]:
            Phase_Axis = figure.add_subplot(grid[phase_row, column])
            Last_Phase_Image = Phase_Axis.imshow(np.tile(Coding_Matrix, (Rows, 1)), aspect="equal", cmap=Phase_State_Cmap,
                                                 vmin=-0.5, vmax=3.5, interpolation="nearest")
            Phase_Title = f"({Phase_Letter}) {MS_Name} compensation-phase map" if phase_row == 1 else f"{MS_Name} compensation-phase map"
            Phase_Axis.set_title(Phase_Title, pad=5)
            Phase_Axis.set_xlabel("Column index")
            Phase_Axis.set_ylabel("Row")
            Phase_Axis.set_xticks(Column_Tick_Positions, Column_Tick_Labels)
            Phase_Axis.set_yticks(np.arange(Rows), np.arange(1, Rows + 1))

            # 网格线位于单元边界而不是单元中心，使每个Column index明确对应一整格相位状态。
            Phase_Axis.grid(False)
            Phase_Axis.set_xticks(np.arange(-0.5, Columns, 1.0), minor=True)
            Phase_Axis.set_yticks(np.arange(-0.5, Rows, 1.0), minor=True)
            Phase_Axis.grid(which="minor", color="#C9CED6", linewidth=0.45)
            Phase_Axis.tick_params(which="minor", bottom=False, left=False)

        # 每列底部绘制方向图；只有实际终止时刻额外加入已知角度参考结果。
        Pattern_Axis = figure.add_subplot(grid[3, column])
        Pattern_Axis.plot(Pattern_Angles_Deg, direction_pattern_dB(Coding_MS1), color="#4C78A8", label="CE MS1")
        Pattern_Axis.plot(Pattern_Angles_Deg, direction_pattern_dB(Coding_MS2), "--", color="#E76F51", label="CE MS2")
        if column == len(Selected_Snapshots) - 1:
            Pattern_Axis.plot(Pattern_Angles_Deg, Known_Angle_Pattern, "-.", color="#1B9E77",
                              label="Known-angle 2-bit")
        Pattern_Axis.axvline(test_data["angle_deg"], color="#D95F02", linestyle=":", label="Test direction")
        Pattern_Axis.set(xlim=(-90, 90), ylim=(Pattern_Floor_dB, 1), xlabel="theta (deg)",
                         ylabel="Normalized pattern (dB)" if column == 0 else "",
                         title=f"({Pattern_Letter}) Normalized direction pattern")
        Pattern_Axis.set_xticks(np.arange(-90, 91, 30))
        Pattern_Axis.legend(loc="lower right", fontsize=8)

    Colorbar_Axis = figure.add_axes([0.955, 0.39, 0.012, 0.20])
    Colorbar = figure.colorbar(Last_Phase_Image, cax=Colorbar_Axis, ticks=[0, 1, 2, 3])
    Colorbar.ax.set_yticklabels(["0°", "90°", "180°", "270°"])
    figure.suptitle(f"Part II: Blind-CE evolution at selected angle theta={test_data['angle_deg']:.0f}°", fontsize=14)
