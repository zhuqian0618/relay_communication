"""训练“编码演化 -> 下一次CE初始概率”的初学者版MLP。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ce_demo import ANCHOR_REPEATS, CE_CONFIG, PowerDropMonitor, run_cold_ce_at_angle
from model import (
    FREE_COLUMNS,
    NETWORK_INPUT_DIM,
    SimpleCodeNet,
    build_code_history_features,
    extract_free_column_code,
    make_simulated_measurement,
    save_model,
)


# ============================== 1. 可调整参数 ==============================

RANDOM_SEED = 20260807
# 每个轨迹使用多个独立随机种子，减小“某一次纯CE随机结果”对标签的影响。
# 6次仍可在普通CPU上较快完成，同时比2次默认值提供更多编码演化事件。
SIMULATION_REPEATS = 6
SIMULATION_STEP_DEG = 0.5

BATCH_SIZE = 32
EPOCHS = 250
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 35
CHANGE_LOSS_WEIGHT = 3.0
CODE_LOSS_WEIGHT = 0.70
PROBABILITY_LOSS_WEIGHT = 0.30

REAL_DATA_NPZ: Path | None = None
MODEL_PATH = Path("code_evolution_net.pth")
TRAINING_PLOT_PATH = Path("code_evolution_training.png")
HEATMAP_PATH = Path("code_evolution_heatmap.png")


# ============================== 2. 数据结构 ==============================

@dataclass
class EvolutionDatasetArrays:
    features: np.ndarray
    target_codes: np.ndarray
    previous_codes: np.ndarray
    target_probabilities: np.ndarray
    run_ids: np.ndarray
    angles_deg: np.ndarray

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def subset(self, indices: np.ndarray) -> "EvolutionDatasetArrays":
        return EvolutionDatasetArrays(
            self.features[indices],
            self.target_codes[indices],
            self.previous_codes[indices],
            self.target_probabilities[indices],
            self.run_ids[indices],
            self.angles_deg[indices],
        )


def save_dataset_npz(data: EvolutionDatasetArrays, path: str | Path) -> None:
    np.savez_compressed(
        Path(path),
        features=data.features,
        target_codes=data.target_codes,
        previous_codes=data.previous_codes,
        target_probabilities=data.target_probabilities,
        run_ids=data.run_ids,
        angles_deg=data.angles_deg,
    )


def load_dataset_npz(path: str | Path) -> EvolutionDatasetArrays:
    with np.load(Path(path), allow_pickle=False) as data:
        result = EvolutionDatasetArrays(
            features=np.asarray(data["features"], dtype=np.float32),
            target_codes=np.asarray(data["target_codes"], dtype=np.int64),
            previous_codes=np.asarray(data["previous_codes"], dtype=np.int64),
            target_probabilities=np.asarray(data["target_probabilities"], dtype=np.float32),
            run_ids=np.asarray(data["run_ids"], dtype=str),
            angles_deg=np.asarray(data["angles_deg"], dtype=float),
        )
    if result.features.ndim != 2 or result.features.shape[1] != NETWORK_INPUT_DIM:
        raise ValueError(f"features must have shape [N, {NETWORK_INPUT_DIM}]")
    if result.target_codes.shape != (len(result), 30):
        raise ValueError("target_codes must have shape [N, 30]")
    if result.target_probabilities.shape != (len(result), 30, 4):
        raise ValueError("target_probabilities must have shape [N, 30, 4]")
    return result


def split_by_run(data: EvolutionDatasetArrays) -> tuple[EvolutionDatasetArrays, EvolutionDatasetArrays]:
    unique_runs = np.unique(data.run_ids)
    if unique_runs.size < 2:
        raise ValueError("at least two run_id values are required")
    validation_runs = set(unique_runs[::5] if unique_runs.size >= 5 else unique_runs[-1:])
    validation_mask = np.asarray([run_id in validation_runs for run_id in data.run_ids])
    return data.subset(~validation_mask), data.subset(validation_mask)


# ============================== 3. 2 dB触发轨迹数据 ==============================

def _segment(start: float, stop: float) -> np.ndarray:
    direction = 1.0 if stop >= start else -1.0
    values = np.arange(start, stop + direction * 1e-10, direction * SIMULATION_STEP_DEG)
    if not np.isclose(values[-1], stop):
        values = np.append(values, stop)
    return np.round(values, 10)


def _join(*segments: np.ndarray) -> np.ndarray:
    return np.concatenate((segments[0], *(segment[1:] for segment in segments[1:])))


def trajectory_templates() -> list[tuple[str, np.ndarray]]:
    return [
        ("positive", _join(_segment(0.0, 60.0), _segment(60.0, 0.0))),
        ("negative", _join(_segment(0.0, -60.0), _segment(-60.0, 0.0))),
        ("full_turn", _join(_segment(-60.0, 60.0), _segment(60.0, -60.0))),
    ]


def _anchor_readings(angle_deg: float, code: np.ndarray, seed: int) -> np.ndarray:
    measure = make_simulated_measurement(angle_deg, seed=seed)
    return np.asarray([measure(code) for _ in range(ANCHOR_REPEATS)])


def build_simulation_dataset(seed: int, split_name: str) -> EvolutionDatasetArrays:
    """初始纯CE，移动到下降2 dB时再用纯CE生成监督标签。"""

    features, targets, previous_targets, target_probabilities = [], [], [], []
    run_ids, trigger_angles = [], []

    for repeat in range(SIMULATION_REPEATS):
        for template_index, (template_name, path) in enumerate(trajectory_templates()):
            run_seed = seed + repeat * 100000 + template_index * 10000
            run_id = f"{split_name}_{repeat}_{template_name}"

            initial = run_cold_ce_at_angle(float(path[0]), run_seed, CE_CONFIG)
            latest_best = initial.best_code.copy()
            previous_best = None
            monitor = PowerDropMonitor()
            monitor.calibrate(_anchor_readings(float(path[0]), latest_best, run_seed + 1))

            for position_index, angle_deg in enumerate(path[1:], start=1):
                measure = make_simulated_measurement(
                    float(angle_deg), seed=run_seed + 1000 + position_index
                )
                if not monitor.update(measure(latest_best)):
                    continue

                label = run_cold_ce_at_angle(
                    float(angle_deg), run_seed + 500000 + len(features), CE_CONFIG
                )
                features.append(build_code_history_features(latest_best, previous_best))
                targets.append(extract_free_column_code(label.best_code))
                previous_targets.append(extract_free_column_code(latest_best))
                target_probabilities.append(label.final_probability[FREE_COLUMNS])
                run_ids.append(run_id)
                trigger_angles.append(float(angle_deg))

                previous_best = latest_best.copy()
                latest_best = label.best_code.copy()
                monitor.calibrate(
                    _anchor_readings(
                        float(angle_deg), latest_best, run_seed + 800000 + len(features)
                    )
                )

    if not features:
        raise RuntimeError("simulation generated no 2 dB trigger samples")
    return EvolutionDatasetArrays(
        features=np.asarray(features, dtype=np.float32),
        target_codes=np.asarray(targets, dtype=np.int64),
        previous_codes=np.asarray(previous_targets, dtype=np.int64),
        target_probabilities=np.asarray(target_probabilities, dtype=np.float32),
        run_ids=np.asarray(run_ids, dtype=str),
        angles_deg=np.asarray(trigger_angles, dtype=float),
    )


# ============================== 4. 损失函数 ==============================

def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    previous_codes: torch.Tensor,
    target_probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_column_ce = nn.functional.cross_entropy(
        logits.reshape(-1, 4), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    weights = torch.where(
        targets != previous_codes,
        torch.as_tensor(CHANGE_LOSS_WEIGHT, device=logits.device),
        torch.as_tensor(1.0, device=logits.device),
    )
    code_loss = torch.sum(per_column_ce * weights) / torch.sum(weights)
    per_column_kl = nn.functional.kl_div(
        nn.functional.log_softmax(logits, dim=-1),
        target_probabilities,
        reduction="none",
    ).sum(dim=-1)
    probability_loss = per_column_kl.mean()
    total = CODE_LOSS_WEIGHT * code_loss + PROBABILITY_LOSS_WEIGHT * probability_loss
    return total, code_loss, probability_loss


def evaluate_model(model: SimpleCodeNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "code_loss": 0.0, "probability_loss": 0.0}
    sample_count = correct = changed_correct = changed_count = joint_correct = 0
    variable_count = 0
    with torch.no_grad():
        for batch in loader:
            features, targets, previous_codes, target_probabilities = [item.to(device) for item in batch]
            logits = model(features)
            loss, code_loss, probability_loss = combined_loss(
                logits, targets, previous_codes, target_probabilities
            )
            batch_size = features.shape[0]
            sample_count += batch_size
            for name, value in (
                ("loss", loss), ("code_loss", code_loss),
                ("probability_loss", probability_loss),
            ):
                totals[name] += float(value.item()) * batch_size
            predicted = torch.argmax(logits, dim=-1)
            matches = predicted == targets
            changes = targets != previous_codes
            correct += int(matches.sum().item())
            variable_count += int(matches.numel())
            changed_correct += int((matches & changes).sum().item())
            changed_count += int(changes.sum().item())
            joint_correct += int(torch.all(matches, dim=1).sum().item())
    return {
        **{name: value / sample_count for name, value in totals.items()},
        "accuracy": correct / variable_count,
        "change_accuracy": changed_correct / max(changed_count, 1),
        "joint_accuracy": joint_correct / sample_count,
    }


# ============================== 5. 训练循环 ==============================

def train_model(
    model: SimpleCodeNet,
    train_data: EvolutionDatasetArrays,
    validation_data: EvolutionDatasetArrays,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    patience: int = EARLY_STOPPING_PATIENCE,
    seed: int = RANDOM_SEED,
    device: str | None = None,
) -> dict[str, object]:
    torch.manual_seed(seed)
    device_object = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_object)

    def loader(data: EvolutionDatasetArrays, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(data.features),
            torch.from_numpy(data.target_codes),
            torch.from_numpy(data.previous_codes),
            torch.from_numpy(data.target_probabilities),
        )
        return DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed) if shuffle else None,
        )

    training_loader = loader(train_data, True)
    training_check_loader = loader(train_data, False)
    validation_loader = loader(validation_data, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    metric_names = (
        "loss", "code_loss", "probability_loss", "accuracy",
        "change_accuracy", "joint_accuracy",
    )
    history: dict[str, object] = {
        f"{split}_{name}": [] for split in ("train", "validation") for name in metric_names
    }
    best_loss, best_epoch, best_state, stale_epochs = np.inf, 0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in training_loader:
            features, targets, previous_codes, target_probabilities = [
                item.to(device_object) for item in batch
            ]
            loss, _, _ = combined_loss(
                model(features), targets, previous_codes, target_probabilities
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_model(model, training_check_loader, device_object)
        validation_metrics = evaluate_model(model, validation_loader, device_object)
        for name in metric_names:
            history[f"train_{name}"].append(train_metrics[name])
            history[f"validation_{name}"].append(validation_metrics[name])
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | val total {validation_metrics['loss']:.4f} | "
                f"CE {validation_metrics['code_loss']:.4f} | "
                f"KL {validation_metrics['probability_loss']:.4f} | "
                f"accuracy {validation_metrics['accuracy']:.3f}"
            )
        if validation_metrics["loss"] < best_loss - 1e-4:
            best_loss, best_epoch = validation_metrics["loss"], epoch
            best_state, stale_epochs = deepcopy(model.state_dict()), 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_validation_loss"] = float(best_loss)
    return history


# ============================== 6. 绘图和入口 ==============================

def plot_training_history(history: dict[str, object], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].plot(epochs, history["train_loss"], label="Training")
    axes[0].plot(epochs, history["validation_loss"], label="Validation")
    axes[0].set(title="Combined loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["validation_code_loss"], label="Cross entropy")
    axes[1].plot(epochs, history["validation_probability_loss"], label="KL divergence")
    axes[1].set(title="Validation loss parts", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    axes[2].plot(epochs, history["validation_accuracy"], label="All columns")
    axes[2].plot(epochs, history["validation_change_accuracy"], label="Changed columns")
    axes[2].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(Path(path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_heatmap(model: SimpleCodeNet, data: EvolutionDatasetArrays, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    with torch.no_grad():
        predicted = torch.argmax(model(torch.from_numpy(data.features).to(device)), dim=-1).cpu().numpy()
    figure, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for axis, codes, title in (
        (axes[0], data.target_codes, "Cold-CE target codes"),
        (axes[1], predicted, "Neural predicted codes"),
    ):
        image = axis.imshow(codes.T, aspect="auto", origin="lower", vmin=-0.5, vmax=3.5)
        axis.set(title=title, ylabel="Free column")
    axes[1].set_xlabel("2 dB trigger sample (angle is not an input)")
    figure.colorbar(image, ax=axes, ticks=(0, 1, 2, 3), fraction=0.025)
    figure.subplots_adjust(right=0.89, hspace=0.28)
    figure.savefig(Path(path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    print("PyTorch:", torch.__version__)
    print("Network input:", NETWORK_INPUT_DIM, "(code history only)")
    print("Generating 2 dB trigger datasets with the shared cold CE...")
    training_data = build_simulation_dataset(RANDOM_SEED, "train")
    validation_data = build_simulation_dataset(RANDOM_SEED + 1, "validation")
    print("Training/validation samples:", len(training_data), len(validation_data))

    if REAL_DATA_NPZ is not None:
        real_data = load_dataset_npz(REAL_DATA_NPZ)
        training_data, validation_data = split_by_run(real_data)
        print("Using measured NPZ samples:", len(real_data))

    model = SimpleCodeNet()
    print(model)
    history = train_model(model, training_data, validation_data)
    save_model(model, MODEL_PATH)
    plot_training_history(history, TRAINING_PLOT_PATH)
    plot_heatmap(model, validation_data, HEATMAP_PATH)
    print(
        f"Best epoch={history['best_epoch']}, "
        f"validation loss={history['best_validation_loss']:.4f}"
    )
    print("Saved:", MODEL_PATH, TRAINING_PLOT_PATH, HEATMAP_PATH)


if __name__ == "__main__":
    main()
