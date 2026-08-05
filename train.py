"""训练“频谱仪探针 -> 编码概率”MLP。

推荐直接在 VS Code 中打开本文件，先阅读顶部参数，再点击“运行Python文件”。
网络输入中没有角度；角度只在第2部分的仿真器内部用于产生功率数据和标签。
"""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import (
    FIXED_COLUMNS,
    FREE_COLUMNS,
    NETWORK_INPUT_DIM,
    PROBE_COUNT,
    SimpleCodeNet,
    build_probe_codes,
    build_probe_features,
    extract_free_column_code,
    make_simulated_measurement,
    reference_joint_code,
    save_model,
    validate_joint_code,
)


# ============================== 1. 可调整参数 ==============================

RANDOM_SEED = 20260805
ANGLE_STEPS_DEG = (0.5, 1.0, 2.0, 5.0, 10.0)
SIMULATION_REPEATS = 1

BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 2e-3
EARLY_STOPPING_PATIENCE = 40
CHANGE_LOSS_WEIGHT = 3.0

REAL_DATA_CSV: Path | None = None
FINE_TUNE_EPOCHS = 100
FINE_TUNE_LEARNING_RATE = 2e-4

MODEL_PATH = Path("probe_code_net.pth")
TRAINING_PLOT_PATH = Path("probe_training_history.png")
HEATMAP_PATH = Path("probe_code_heatmap.png")


# ============================== 2. 仿真数据集 ==============================

@dataclass
class ProbeDatasetArrays:
    """训练所需数组；angles只用于画图检查，绝不送入网络。"""

    features: np.ndarray
    target_codes: np.ndarray
    previous_codes: np.ndarray
    run_ids: np.ndarray
    angles_deg: np.ndarray

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def subset(self, indices: np.ndarray) -> "ProbeDatasetArrays":
        return ProbeDatasetArrays(
            self.features[indices],
            self.target_codes[indices],
            self.previous_codes[indices],
            self.run_ids[indices],
            self.angles_deg[indices],
        )


def _segment(start: float, stop: float, step_deg: float) -> np.ndarray:
    direction = 1.0 if stop >= start else -1.0
    values = np.arange(start, stop + direction * 1e-9, direction * step_deg)
    if not np.isclose(values[-1], stop):
        values = np.append(values, stop)
    return np.round(values, 10)


def _join_segments(*segments: np.ndarray) -> np.ndarray:
    result = [segments[0]]
    result.extend(segment[1:] for segment in segments[1:])
    return np.concatenate(result)


def build_trajectory_templates() -> list[tuple[str, np.ndarray]]:
    """构造正向、反向、返回0度和边界折返轨迹。"""

    templates = []
    for step in ANGLE_STEPS_DEG:
        templates.extend(
            [
                (
                    f"step{step:g}_positive",
                    _join_segments(_segment(0.0, 60.0, step), _segment(60.0, 0.0, step)),
                ),
                (
                    f"step{step:g}_negative",
                    _join_segments(_segment(0.0, -60.0, step), _segment(-60.0, 0.0, step)),
                ),
                (
                    f"step{step:g}_turn",
                    _join_segments(
                        _segment(-60.0, 60.0, step),
                        _segment(60.0, -60.0, step),
                    ),
                ),
            ]
        )
    return templates


def _sample_hardware(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为一整条轨迹固定一组单元误差，模拟不同实验批次。"""

    state_offsets = np.zeros(32, dtype=int)
    faulty_columns_mask = rng.random(FREE_COLUMNS.size) < 0.06
    state_offsets[FREE_COLUMNS[faulty_columns_mask]] = rng.choice(
        (-1, 1), size=int(np.sum(faulty_columns_mask))) % 4
    phase_errors = rng.normal(0.0, np.deg2rad(4.0), size=32)
    amplitude_errors = np.clip(rng.normal(1.0, 0.03, size=32), 0.85, 1.15)
    phase_errors[list(FIXED_COLUMNS)] = 0.0
    amplitude_errors[list(FIXED_COLUMNS)] = 1.0
    return state_offsets, phase_errors, amplitude_errors


def build_simulation_dataset(seed: int, split_name: str) -> ProbeDatasetArrays:
    """生成角度隐藏的轨迹样本；不同seed代表不同硬件和噪声批次。"""

    rng = np.random.default_rng(seed)
    feature_rows, targets, previous_targets, run_ids, current_angles = [], [], [], [], []
    noise_levels = np.asarray([-100.0, -90.0, -75.0])

    for repeat in range(SIMULATION_REPEATS):
        for template_name, path in build_trajectory_templates():
            run_id = f"{split_name}_{repeat}_{template_name}"
            state_offsets, phase_errors, amplitude_errors = _sample_hardware(rng)
            radius = float(6.5 + rng.normal(0.0, 0.08))
            transmit_power = float(rng.uniform(-3.0, 3.0))
            noise_power = float(rng.choice(noise_levels))
            measurement_std = {-100.0: 0.08, -90.0: 0.15, -75.0: 0.30}[noise_power]

            for step_index in range(1, path.size):
                previous_angle = float(path[step_index - 1])
                current_angle = float(path[step_index])
                previous_code = reference_joint_code(previous_angle, state_offsets)
                target_code = reference_joint_code(current_angle, state_offsets)

                previous_measure = make_simulated_measurement(
                    previous_angle,
                    transmit_power,
                    noise_power,
                    measurement_std,
                    radius,
                    gain_drift_dB=float(rng.normal(0.0, 0.35)),
                    state_offsets=state_offsets,
                    phase_errors_rad=phase_errors,
                    amplitude_errors=amplitude_errors,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                previous_best_power = previous_measure(previous_code)

                current_measure = make_simulated_measurement(
                    current_angle,
                    transmit_power,
                    noise_power,
                    measurement_std,
                    radius,
                    gain_drift_dB=float(rng.normal(0.0, 0.35)),
                    state_offsets=state_offsets,
                    phase_errors_rad=phase_errors,
                    amplitude_errors=amplitude_errors,
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                baseline = np.asarray([current_measure(previous_code) for _ in range(2)])
                probe_codes = build_probe_codes(previous_code)
                probe_powers = np.asarray([current_measure(code) for code in probe_codes])

                feature_rows.append(
                    build_probe_features(
                        previous_code,
                        previous_best_power,
                        baseline,
                        probe_powers,
                    )
                )
                targets.append(extract_free_column_code(target_code))
                previous_targets.append(extract_free_column_code(previous_code))
                run_ids.append(run_id)
                current_angles.append(current_angle)

    return ProbeDatasetArrays(
        features=np.asarray(feature_rows, dtype=np.float32),
        target_codes=np.asarray(targets, dtype=np.int64),
        previous_codes=np.asarray(previous_targets, dtype=np.int64),
        run_ids=np.asarray(run_ids, dtype=str),
        angles_deg=np.asarray(current_angles, dtype=float),
    )


# ============================== 3. 实测CSV ==============================

def measured_csv_header() -> list[str]:
    return (
        ["run_id", "previous_best_power_dBm", "baseline_1_dBm", "baseline_2_dBm"]
        + [f"probe_{index}_dBm" for index in range(PROBE_COUNT)]
        + [f"prev_c{index}" for index in range(32)]
        + [f"target_c{index}" for index in range(32)]
    )


def load_measured_csv(path: str | Path) -> ProbeDatasetArrays:
    """读取不含角度的实测校准数据，并自动构造128维输入。"""

    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != measured_csv_header():
            raise ValueError("CSV header does not match measured_csv_header()")
        for line_number, row in enumerate(reader, start=2):
            try:
                previous_code = validate_joint_code(
                    np.asarray([int(row[f"prev_c{i}"]) for i in range(32)])
                )
                target_code = validate_joint_code(
                    np.asarray([int(row[f"target_c{i}"]) for i in range(32)])
                )
                baseline = np.asarray(
                    [float(row["baseline_1_dBm"]), float(row["baseline_2_dBm"])]
                )
                probes = np.asarray([float(row[f"probe_{i}_dBm"]) for i in range(PROBE_COUNT)])
                features = build_probe_features(
                    previous_code,
                    float(row["previous_best_power_dBm"]),
                    baseline,
                    probes,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid value on CSV line {line_number}: {error}") from error
            if not row["run_id"].strip():
                raise ValueError(f"CSV line {line_number} has an empty run_id")
            rows.append((features, extract_free_column_code(target_code),
                         extract_free_column_code(previous_code), row["run_id"]))
    if not rows:
        raise ValueError("CSV file contains no measurement rows")
    return ProbeDatasetArrays(
        features=np.asarray([row[0] for row in rows], dtype=np.float32),
        target_codes=np.asarray([row[1] for row in rows], dtype=np.int64),
        previous_codes=np.asarray([row[2] for row in rows], dtype=np.int64),
        run_ids=np.asarray([row[3] for row in rows], dtype=str),
        angles_deg=np.full(len(rows), np.nan),
    )


def split_measured_by_run(data: ProbeDatasetArrays) -> tuple[ProbeDatasetArrays, ProbeDatasetArrays]:
    """整批划分实测数据，避免同一飞行批次泄漏到训练集和验证集。"""

    unique_runs = np.unique(data.run_ids)
    if unique_runs.size < 2:
        raise ValueError("real-data fine-tuning requires at least two different run_id values")
    validation_runs = set(unique_runs[::5] if unique_runs.size >= 5 else unique_runs[-1:])
    validation_mask = np.asarray([run_id in validation_runs for run_id in data.run_ids])
    return data.subset(~validation_mask), data.subset(validation_mask)


# ============================== 4. 损失函数和指标 ==============================

def weighted_code_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    previous_codes: torch.Tensor,
    change_weight: float = CHANGE_LOSS_WEIGHT,
) -> torch.Tensor:
    """状态发生变化的可控列获得更高权重，避免网络只复制上一编码。"""

    per_column_loss = nn.functional.cross_entropy(
        logits.reshape(-1, 4), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    weights = torch.where(
        targets != previous_codes,
        torch.as_tensor(change_weight, device=logits.device),
        torch.as_tensor(1.0, device=logits.device),
    )
    return torch.sum(per_column_loss * weights) / torch.sum(weights)


def evaluate_model(model: SimpleCodeNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    correct = changed_correct = changed_count = joint_correct = 0
    column_count = 0
    with torch.no_grad():
        for features, targets, previous_codes in loader:
            features = features.to(device)
            targets = targets.to(device)
            previous_codes = previous_codes.to(device)
            logits = model(features)
            batch_size = features.shape[0]
            loss_sum += float(weighted_code_loss(logits, targets, previous_codes).item()) * batch_size
            sample_count += batch_size
            predicted = torch.argmax(logits, dim=-1)
            matches = predicted == targets
            changes = targets != previous_codes
            correct += int(matches.sum().item())
            column_count += int(matches.numel())
            changed_correct += int((matches & changes).sum().item())
            changed_count += int(changes.sum().item())
            joint_correct += int(torch.all(matches, dim=1).sum().item())
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / column_count,
        "change_accuracy": changed_correct / max(changed_count, 1),
        "joint_accuracy": joint_correct / sample_count,
    }


# ============================== 5. 训练循环 ==============================

def train_model(
    model: SimpleCodeNet,
    train_data: ProbeDatasetArrays,
    validation_data: ProbeDatasetArrays,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    patience: int = EARLY_STOPPING_PATIENCE,
    seed: int = RANDOM_SEED,
    device: str | None = None,
) -> dict[str, object]:
    """完成训练、验证、早停并恢复验证损失最低的参数。"""

    torch.manual_seed(seed)
    device_object = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_object)
    # 保存训练轨迹中真实出现过的“整组状态变化”。在线阶段用这些模板
    # 生成相干候选，不需要知道产生模板时的角度。
    model.transition_deltas = np.unique(
        (train_data.target_codes - train_data.previous_codes) % 4,
        axis=0,
    )

    def make_loader(data: ProbeDatasetArrays, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(data.features),
            torch.from_numpy(data.target_codes),
            torch.from_numpy(data.previous_codes),
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )

    train_loader = make_loader(train_data, True)
    train_check_loader = make_loader(train_data, False)
    validation_loader = make_loader(validation_data, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    history: dict[str, object] = {
        "train_loss": [], "validation_loss": [],
        "train_accuracy": [], "validation_accuracy": [],
        "train_change_accuracy": [], "validation_change_accuracy": [],
        "train_joint_accuracy": [], "validation_joint_accuracy": [],
    }
    best_loss = np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for features, targets, previous_codes in train_loader:
            features = features.to(device_object)
            targets = targets.to(device_object)
            previous_codes = previous_codes.to(device_object)

            logits = model(features)
            loss = weighted_code_loss(logits, targets, previous_codes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_model(model, train_check_loader, device_object)
        validation_metrics = evaluate_model(model, validation_loader, device_object)
        for name in ("loss", "accuracy", "change_accuracy", "joint_accuracy"):
            history[f"train_{name}"].append(train_metrics[name])
            history[f"validation_{name}"].append(validation_metrics[name])

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | train loss {train_metrics['loss']:.4f} | "
                f"validation loss {validation_metrics['loss']:.4f} | "
                f"all accuracy {validation_metrics['accuracy']:.3f} | "
                f"change accuracy {validation_metrics['change_accuracy']:.3f}"
            )

        if validation_metrics["loss"] < best_loss - 1e-4:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_validation_loss"] = float(best_loss)
    return history


# ============================== 6. 绘图 ==============================

def plot_training_history(history: dict[str, object], save_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].plot(epochs, history["train_loss"], label="Training")
    axes[0].plot(epochs, history["validation_loss"], label="Validation")
    axes[0].set(title="Weighted cross-entropy", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["validation_accuracy"], label="All columns")
    axes[1].plot(epochs, history["validation_change_accuracy"], label="Changed columns")
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    axes[1].legend()
    axes[2].plot(epochs, history["train_joint_accuracy"], label="Training")
    axes[2].plot(epochs, history["validation_joint_accuracy"], label="Validation")
    axes[2].set(title="Exact joint-code accuracy", xlabel="Epoch", ylabel="Ratio", ylim=(0, 1.02))
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(Path(save_path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_code_heatmap(
    model: SimpleCodeNet,
    validation_data: ProbeDatasetArrays,
    save_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    first_run = validation_data.run_ids[0]
    indices = np.flatnonzero(validation_data.run_ids == first_run)
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(validation_data.features[indices]).to(device))
        predicted = torch.argmax(logits, dim=-1).cpu().numpy()
    true_codes = validation_data.target_codes[indices]
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 5.2), sharex=True)
    for axis, codes, title in (
        (axes[0], true_codes, "True free-column codes"),
        (axes[1], predicted, "Probe-MLP predicted codes"),
    ):
        image = axis.imshow(codes.T, aspect="auto", origin="lower", vmin=-0.5, vmax=3.5)
        axis.set(title=title, ylabel="Free column")
    axes[1].set_xlabel("Trajectory sample index (angle is not a network input)")
    figure.colorbar(image, ax=axes, ticks=(0, 1, 2, 3), fraction=0.025)
    figure.subplots_adjust(right=0.89, hspace=0.28)
    figure.savefig(Path(save_path), dpi=180, bbox_inches="tight")
    plt.close(figure)


# ============================== 7. 直接运行训练 ==============================

def main() -> None:
    print("PyTorch version:", torch.__version__)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("Network input dimension:", NETWORK_INPUT_DIM, "(no angle included)")

    print("\n[1/5] Generating simulation training data...")
    training_data = build_simulation_dataset(RANDOM_SEED, "train")
    validation_data = build_simulation_dataset(RANDOM_SEED + 1, "validation")
    print("Training samples:", len(training_data))
    print("Validation samples:", len(validation_data))

    print("\n[2/5] Creating the MLP...")
    model = SimpleCodeNet()
    print(model)

    print("\n[3/5] Training...")
    history = train_model(model, training_data, validation_data)
    print(
        f"Best epoch={history['best_epoch']}, "
        f"validation loss={history['best_validation_loss']:.4f}"
    )

    if REAL_DATA_CSV is not None:
        print("\n[4/5] Fine-tuning with measured data...")
        measured_data = load_measured_csv(REAL_DATA_CSV)
        measured_train, measured_validation = split_measured_by_run(measured_data)
        history = train_model(
            model,
            measured_train,
            measured_validation,
            epochs=FINE_TUNE_EPOCHS,
            learning_rate=FINE_TUNE_LEARNING_RATE,
            batch_size=min(BATCH_SIZE, len(measured_train)),
            seed=RANDOM_SEED + 2,
        )
    else:
        print("\n[4/5] REAL_DATA_CSV=None, skipping measured-data fine-tuning.")

    print("\n[5/5] Saving model and figures...")
    save_model(model, MODEL_PATH)
    plot_training_history(history, TRAINING_PLOT_PATH)
    plot_code_heatmap(model, validation_data, HEATMAP_PATH)
    print("Model:", MODEL_PATH)
    print("Training curves:", TRAINING_PLOT_PATH)
    print("Code heatmap:", HEATMAP_PATH)


if __name__ == "__main__":
    main()
