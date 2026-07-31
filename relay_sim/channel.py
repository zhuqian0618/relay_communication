"""5.8 GHz远场空中信道参数、信道矩阵计算与测试角度绘图。"""

import matplotlib.pyplot as plt
import numpy as np

from .metasurface import Column_Positions_MS, Columns, Period_MS

# 本文件只保存信道自身需要的参数，不再建立集中式config类。
Speed_Of_Light_M_S = 299_792_458.0
Carrier_Frequency_Hz = 5.8e9
Lambda = Speed_Of_Light_M_S / Carrier_Frequency_Hz
Separation_Distance_M = 6.5

# 两块阵面的最大尺寸为16×25 mm；Fraunhofer距离用于检查10 m是否满足远场条件。
Aperture_Width_MS = Columns * Period_MS
Far_Field_Distance_M = 2 * Aperture_Width_MS**2 / Lambda

# 从两块超表面各自的辐射面观察时，列编号一致，因此相同编码指向相同本地方位。
MS2_Local_Angle_Sign = 1.0


def build_far_field_channel(angle_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, complex]:
    """构造H12=alpha*a2*a1^H，并同时返回两端阵列响应和Friis复系数。"""

    # 按MATLAB相位公式显式计算每一列的空间相位β0*xn*sin(θ)。
    Beta0 = 2 * np.pi / Lambda
    Phase1_Rad = Beta0 * Column_Positions_MS * np.sin(angle_rad)
    Phase2_Rad = Beta0 * Column_Positions_MS * np.sin(MS2_Local_Angle_Sign * angle_rad)

    # 采用e^(jωt)约定，导向矢量使用负指数；列号一致时两块板在同一角度具有相同响应。
    a1 = np.exp(-1j * Phase1_Rad)
    a2 = np.exp(-1j * Phase2_Rad)

    # alpha的模长是Friis幅度λ/(4πR)，复指数表示两块阵面中心之间的传播相位。
    alpha = Lambda / (4 * np.pi * Separation_Distance_M)
    alpha *= np.exp(-1j * Beta0 * Separation_Distance_M)

    # 外积a2*a1^H得到16×16、rank-one的远场LoS信道矩阵。
    h12 = alpha * np.outer(a2, np.conj(a1))
    return h12, a1, a2, alpha


def plot_channel_at_test_angle(h12: np.ndarray, angle_deg: float) -> None:
    """用信道相位矩阵和奇异值检查远场rank-one模型。"""

    # 奇异值除以最大值后，理想rank-one信道只有第一个等于1。
    singular_values = np.linalg.svd(h12, compute_uv=False)
    singular_values /= singular_values[0]

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))

    # 左图显示MS1每列到MS2每列的传播相位。
    image = axes[0].imshow(np.angle(h12, deg=True), origin="lower", aspect="auto",
                           cmap="twilight", vmin=-180, vmax=180)
    axes[0].set(xlabel="MS1 column index", ylabel="MS2 column index",
                title=f"Channel phase at psi={angle_deg:.0f}°")
    figure.colorbar(image, ax=axes[0], label="Phase (deg)")

    # 右图采用对数纵轴，便于看出除第一奇异值外的数值接近机器精度。
    axes[1].stem(np.arange(1, singular_values.size + 1), singular_values, basefmt=" ")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Singular-value index", ylabel="Normalized singular value",
                ylim=(1e-18, 2), title="Rank-one far-field check")
    figure.tight_layout()
