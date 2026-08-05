"""频谱仪探针辅助编码网络。

这个文件只放三类最基础的内容：

1. 生成上一编码附近的六个固定探针编码；
2. 把“上一编码 + 八次功率测量”整理成128维网络输入；
3. 定义一个普通的全连接神经网络（MLP）。

实际在线调用不需要角度。角度只在仿真器生成训练数据时使用。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from relay_sim.Channel_Modeling import build_far_field_channel, link_metrics
from relay_sim.MS_Configuration import Columns, Compensation_Phasors, calculate_2bit_compensation_code


# ============================== 1. 基本参数 ==============================

STATE_COUNT = 4
COLUMN_COUNT = 2 * Columns
FIXED_COLUMNS = (0, Columns)
FREE_COLUMNS = np.asarray(
    [column_index for column_index in range(COLUMN_COUNT) if column_index not in FIXED_COLUMNS],
    dtype=int,
)
FREE_COLUMN_COUNT = FREE_COLUMNS.size

PROBE_COUNT = 6
POWER_CLIP_DB = 12.0
REPEAT_CLIP_DB = 3.0
NETWORK_INPUT_DIM = FREE_COLUMN_COUNT * STATE_COUNT + PROBE_COUNT + 2
MODEL_FORMAT_VERSION = 3


# ============================== 2. 编码工具 ==============================

def validate_joint_code(code: np.ndarray) -> np.ndarray:
    """检查32个列控2-bit状态，并强制两块MS的参考列为状态0。"""

    result = np.asarray(code, dtype=int).reshape(-1).copy()
    if result.size != COLUMN_COUNT:
        raise ValueError(f"joint code must contain {COLUMN_COUNT} column states")
    if np.any((result < 0) | (result >= STATE_COUNT)):
        raise ValueError("all code states must be integers from 0 to 3")
    result[list(FIXED_COLUMNS)] = 0
    return result


def extract_free_column_code(joint_code: np.ndarray) -> np.ndarray:
    """从32列联合编码中取出30个参与优化的列状态。"""

    return validate_joint_code(joint_code)[FREE_COLUMNS]


def restore_joint_column_code(free_column_states: np.ndarray) -> np.ndarray:
    """把30个预测列状态放回32列联合编码。"""

    free_column_states = np.asarray(free_column_states, dtype=int).reshape(-1)
    if free_column_states.size != FREE_COLUMN_COUNT:
        raise ValueError(f"free code must contain {FREE_COLUMN_COUNT} column states")
    if np.any((free_column_states < 0) | (free_column_states >= STATE_COUNT)):
        raise ValueError("all free states must be integers from 0 to 3")
    joint_column_code = np.zeros(COLUMN_COUNT, dtype=int)
    joint_column_code[FREE_COLUMNS] = free_column_states
    return joint_column_code


@lru_cache(maxsize=512)
def _cached_reference_code(angle_deg: float) -> tuple[int, ...]:
    if not -60.0 <= angle_deg <= 60.0:
        raise ValueError("simulation angle must remain within [-60, +60] degrees")
    single_ms = np.asarray(calculate_2bit_compensation_code(angle_deg)[2], dtype=int)
    # 公共相位不改变波束方向。减去首列状态后，两块超表面的首列都固定为0。
    single_ms = (single_ms - single_ms[0]) % STATE_COUNT
    return tuple(validate_joint_code(np.concatenate((single_ms, single_ms))).tolist())


def reference_joint_code(angle_deg: float, state_offsets: np.ndarray | None = None) -> np.ndarray:
    """生成仿真标签；state_offsets用于模拟单元离散相位偏差。"""

    ideal = np.asarray(_cached_reference_code(float(angle_deg)), dtype=int)
    if state_offsets is None:
        return ideal.copy()
    offsets = validate_joint_code(state_offsets)
    result = ideal.copy()
    result[FREE_COLUMNS] = (result[FREE_COLUMNS] - offsets[FREE_COLUMNS]) % STATE_COUNT
    return validate_joint_code(result)


def build_probe_codes(previous_code: np.ndarray) -> np.ndarray:
    """将可调列分成三组，生成“增加一级/减少一级”六个固定探针。"""

    previous_code = validate_joint_code(previous_code)
    probes = []
    # 30个参与优化的列按0,1,2,0,1,2...轮流分为三组。
    for group_index in range(3):
        column_group = FREE_COLUMNS[np.arange(FREE_COLUMN_COUNT) % 3 == group_index]
        for change in (+1, -1):
            probe = previous_code.copy()
            probe[column_group] = (probe[column_group] + change) % STATE_COUNT
            probes.append(probe)
    return np.asarray(probes, dtype=int)


# ============================== 3. 128维输入 ==============================

def build_probe_features(
    previous_code: np.ndarray,
    previous_best_power_dBm: float,
    baseline_powers_dBm: np.ndarray,
    probe_powers_dBm: np.ndarray,
) -> np.ndarray:
    """把已知编码和频谱仪读数转换成不含角度的128维输入。"""

    previous_code = validate_joint_code(previous_code)
    baseline = np.asarray(baseline_powers_dBm, dtype=float).reshape(-1)
    probes = np.asarray(probe_powers_dBm, dtype=float).reshape(-1)
    if baseline.size != 2:
        raise ValueError("exactly two baseline powers are required")
    if probes.size != PROBE_COUNT:
        raise ValueError(f"exactly {PROBE_COUNT} probe powers are required")
    all_powers = np.concatenate(([float(previous_best_power_dBm)], baseline, probes))
    if not np.all(np.isfinite(all_powers)):
        raise ValueError("all spectrum-analyzer powers must be finite")

    # 每个参与优化的列用四维one-hot表示，例如状态2写成[0, 0, 1, 0]。
    previous_one_hot = np.eye(STATE_COUNT, dtype=np.float32)[previous_code[FREE_COLUMNS]].reshape(-1)
    baseline_mean = float(np.mean(baseline))

    # 相对功率消除了所有读数共同增加或减少所造成的影响。
    relative_probe_power = np.clip(
        probes - baseline_mean, -POWER_CLIP_DB, POWER_CLIP_DB
    ) / POWER_CLIP_DB
    movement_power_change = np.clip(
        baseline_mean - float(previous_best_power_dBm), -POWER_CLIP_DB, POWER_CLIP_DB
    ) / POWER_CLIP_DB
    repeat_difference = np.clip(
        abs(baseline[0] - baseline[1]), 0.0, REPEAT_CLIP_DB
    ) / REPEAT_CLIP_DB

    features = np.concatenate(
        (previous_one_hot, relative_probe_power, [movement_power_change, repeat_difference])
    ).astype(np.float32)
    if features.shape != (NETWORK_INPUT_DIM,):
        raise RuntimeError(f"internal feature shape must be ({NETWORK_INPUT_DIM},)")
    return features


# ============================== 4. 仅供训练和演示的仿真测量 ==============================

def make_simulated_measurement(
    angle_deg: float,
    transmit_power_dBm: float = 0.0,
    noise_power_dBm: float = -90.0,
    measurement_noise_std_dB: float = 0.15,
    separation_distance_M: float = 6.5,
    gain_drift_dB: float = 0.0,
    state_offsets: np.ndarray | None = None,
    phase_errors_rad: np.ndarray | None = None,
    amplitude_errors: np.ndarray | None = None,
    seed: int = 20260805,
):
    """创建仿真频谱仪函数；angle_deg不会进入神经网络。"""

    angle_deg = float(angle_deg)
    if not -60.0 <= angle_deg <= 60.0:
        raise ValueError("simulation angle must remain within [-60, +60] degrees")
    angle_rad = np.deg2rad(angle_deg)
    h12, _, _, _ = build_far_field_channel(angle_rad, separation_distance_M)
    transmit_power_W = 10.0 ** ((float(transmit_power_dBm) - 30.0) / 10.0)
    noise_power_W = 10.0 ** ((float(noise_power_dBm) - 30.0) / 10.0)
    drift_linear = 10.0 ** (float(gain_drift_dB) / 20.0)
    offsets = (
        np.zeros(COLUMN_COUNT, dtype=int)
        if state_offsets is None
        else validate_joint_code(state_offsets)
    )
    phases = (
        np.zeros(COLUMN_COUNT, dtype=float)
        if phase_errors_rad is None
        else np.asarray(phase_errors_rad, dtype=float).reshape(COLUMN_COUNT)
    )
    amplitudes = (
        np.ones(COLUMN_COUNT, dtype=float)
        if amplitude_errors is None
        else np.asarray(amplitude_errors, dtype=float).reshape(COLUMN_COUNT)
    )
    if not np.all(np.isfinite(phases)) or not np.all(np.isfinite(amplitudes)):
        raise ValueError("hardware errors must be finite")
    rng = np.random.default_rng(seed)

    def measure(code: np.ndarray) -> float:
        code = validate_joint_code(code)
        actual_states = (code + offsets) % STATE_COUNT
        hardware = amplitudes * np.exp(1j * phases)
        v1 = Compensation_Phasors[actual_states[:Columns]] * hardware[:Columns]
        v2 = Compensation_Phasors[actual_states[Columns:]] * hardware[Columns:]
        h_eff, _, _ = link_metrics(v1, v2, angle_rad, h12)
        signal_power_W = transmit_power_W * abs(drift_linear * h_eff) ** 2
        total_power_dBm = 10.0 * np.log10(max(signal_power_W + noise_power_W, 1e-30)) + 30.0
        return float(total_power_dBm + rng.normal(0.0, measurement_noise_std_dB))

    return measure


# ============================== 5. MLP网络 ==============================

class SimpleCodeNet(nn.Module):
    """根据探针响应预测30个参与优化列的四状态概率。"""

    def __init__(self) -> None:
        super().__init__()
        # 训练时保存出现过的完整编码变化模板。它不是网络输入，也不包含角度。
        self.transition_deltas = np.empty((0, FREE_COLUMN_COUNT), dtype=int)
        self.model = nn.Sequential(
            nn.Linear(NETWORK_INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, FREE_COLUMN_COUNT * STATE_COUNT),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(inputs)
        return logits.reshape(-1, FREE_COLUMN_COUNT, STATE_COUNT)


def predict_probabilities(
    model: SimpleCodeNet,
    features: np.ndarray,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """返回完整的32×4列状态概率；两块MS的参考列只允许状态0。"""

    device_object = torch.device(device or next(model.parameters()).device)
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device_object).reshape(1, -1)
    if feature_tensor.shape[1] != NETWORK_INPUT_DIM:
        raise ValueError(f"network input must contain {NETWORK_INPUT_DIM} values")
    model.eval()
    with torch.no_grad():
        free_column_probability = torch.softmax(model(feature_tensor)[0], dim=-1).cpu().numpy()
    probability = np.zeros((COLUMN_COUNT, STATE_COUNT), dtype=float)
    probability[FREE_COLUMNS] = free_column_probability
    probability[list(FIXED_COLUMNS), 0] = 1.0
    return probability


def save_model(model: SimpleCodeNet, path: str | Path) -> None:
    """只保存模型参数，避免把整个Python对象写入文件。"""

    torch.save(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "network_input_dim": NETWORK_INPUT_DIM,
            "model_state": model.state_dict(),
            "transition_deltas": torch.as_tensor(model.transition_deltas, dtype=torch.long),
        },
        Path(path),
    )


def load_model(path: str | Path, device: str | None = None) -> SimpleCodeNet:
    """创建相同网络并加载训练好的参数。"""

    device_object = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(Path(path), map_location=device_object, weights_only=True)
    if checkpoint.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("model format is outdated; run train.py to train the probe-input model")
    model = SimpleCodeNet().to(device_object)
    model.load_state_dict(checkpoint["model_state"])
    saved_deltas = checkpoint.get(
        "transition_deltas", torch.empty((0, FREE_COLUMN_COUNT), dtype=torch.long)
    )
    model.transition_deltas = saved_deltas.cpu().numpy().astype(int).reshape(-1, FREE_COLUMN_COUNT)
    model.eval()
    return model


# ============================== 6. 单独检查网络 ==============================

if __name__ == "__main__":
    example_input = torch.ones((3, NETWORK_INPUT_DIM), dtype=torch.float32)
    network = SimpleCodeNet()
    example_output = network(example_input)
    print("输入形状:", tuple(example_input.shape))
    print("输出形状:", tuple(example_output.shape))
    print("含义: 3个样本，每个样本预测30个参与优化列，每列有4种状态。")
