"""Beginner-friendly MLP + cross-entropy (CE) code optimization.

The neural network learns only the mapping

    commanded angle -> probability of each 2-bit metasurface state.

It then warm-starts a small, fixed-budget CE search.  The final code is always
chosen from candidates measured by ``measure(code)``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .Channel_Modeling import build_far_field_channel, link_metrics
from .MS_Configuration import Columns, Compensation_Phasors, calculate_2bit_compensation_code

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # CE and dataset helpers remain usable before PyTorch is installed.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


STATE_COUNT = 4
VARIABLE_COUNT = 2 * Columns
FOURIER_HARMONICS = 24
NETWORK_INPUT_DIM = 2 * FOURIER_HARMONICS
MODEL_FORMAT_VERSION = 2
FIXED_VARIABLES = (0, Columns)
FREE_VARIABLES = np.asarray(
    [index for index in range(VARIABLE_COUNT) if index not in FIXED_VARIABLES], dtype=int
)
MeasureFunction = Callable[[np.ndarray], float]


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for network training and prediction. "
            "Install requirements-pytorch.txt first."
        )


def validate_angle(angle_deg: float) -> float:
    angle_deg = float(angle_deg)
    if not -60.0 <= angle_deg <= 60.0:
        raise ValueError("angle_deg must remain within [-60, +60] degrees")
    return angle_deg


def validate_joint_code(code: np.ndarray) -> np.ndarray:
    result = np.asarray(code, dtype=int).reshape(-1).copy()
    if result.size != VARIABLE_COUNT:
        raise ValueError(f"joint code must contain {VARIABLE_COUNT} states")
    if np.any((result < 0) | (result >= STATE_COUNT)):
        raise ValueError("all code states must be integers from 0 to 3")
    result[list(FIXED_VARIABLES)] = 0
    return result


def angle_features(angles_deg: np.ndarray | float) -> np.ndarray:
    """Encode each angle with 24 sine/cosine pairs for quantization boundaries.

    The 2-bit code changes many times over the 120-degree arc.  First-order
    ``sin(theta), cos(theta)`` features are too smooth to represent all those
    transitions efficiently.  Fourier features remain deterministic inputs;
    they do not add another neural-network type.
    """

    angles = np.asarray(angles_deg, dtype=float).reshape(-1)
    if np.any((angles < -60.0) | (angles > 60.0)):
        raise ValueError("all angles must remain within [-60, +60] degrees")
    radians = np.deg2rad(angles)
    features = []
    for harmonic in range(1, FOURIER_HARMONICS + 1):
        features.extend((np.sin(harmonic * radians), np.cos(harmonic * radians)))
    return np.column_stack(features).astype(np.float32)


def reference_joint_code(angle_deg: float) -> np.ndarray:
    """Build an ideal-model joint code and normalize each MS reference column."""

    angle_deg = validate_angle(angle_deg)
    single_ms = np.asarray(calculate_2bit_compensation_code(angle_deg)[2], dtype=int)
    # A common phase does not change beam direction.  Subtracting the first
    # state preserves all relative phases while fixing column 1 to state 0.
    single_ms = (single_ms - single_ms[0]) % STATE_COUNT
    return validate_joint_code(np.concatenate((single_ms, single_ms)))


def _angle_grid(start: float, stop: float, step_deg: float) -> np.ndarray:
    values = np.arange(start, stop + 1e-10, step_deg, dtype=float)
    if values.size == 0 or values[-1] > stop + 1e-8:
        values = values[values <= stop + 1e-8]
    return np.round(values, 10)


def build_simulation_datasets(
    angle_step_deg: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train angles/codes and half-step-offset validation angles/codes."""

    if angle_step_deg <= 0.0 or angle_step_deg > 20.0:
        raise ValueError("angle_step_deg must be in (0, 20]")
    train_angles = _angle_grid(-60.0, 60.0, angle_step_deg)
    half_step = angle_step_deg / 2.0
    validation_angles = _angle_grid(-60.0 + half_step, 60.0 - half_step, angle_step_deg)
    train_codes = np.asarray([reference_joint_code(angle) for angle in train_angles], dtype=np.int64)
    validation_codes = np.asarray(
        [reference_joint_code(angle) for angle in validation_angles], dtype=np.int64
    )
    return train_angles, train_codes, validation_angles, validation_codes


def load_measured_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``angle_deg,c0,...,c31`` labels produced by full experimental CE."""

    path = Path(path)
    expected_header = ["angle_deg"] + [f"c{index}" for index in range(VARIABLE_COUNT)]
    angles, codes = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected_header:
            raise ValueError(
                "CSV header must be exactly: " + ",".join(expected_header)
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                angle = validate_angle(float(row["angle_deg"]))
                raw_code = np.asarray(
                    [int(row[f"c{index}"]) for index in range(VARIABLE_COUNT)], dtype=int
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid value on CSV line {line_number}: {error}") from error
            if np.any((raw_code < 0) | (raw_code > 3)):
                raise ValueError(f"CSV line {line_number} contains a code outside 0..3")
            if raw_code[0] != 0 or raw_code[Columns] != 0:
                raise ValueError(
                    f"CSV line {line_number} must use state 0 for c0 and c{Columns}"
                )
            angles.append(angle)
            codes.append(raw_code)
    if not angles:
        raise ValueError("CSV file contains no measurement rows")
    order = np.argsort(angles)
    return np.asarray(angles)[order], np.asarray(codes, dtype=np.int64)[order]


def split_measured_dataset(
    angles: np.ndarray,
    codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use every fifth sorted real measurement as a small validation set."""

    angles = np.asarray(angles, dtype=float)
    codes = np.asarray(codes, dtype=np.int64)
    if angles.size != codes.shape[0]:
        raise ValueError("measured angles and codes must have equal sample counts")
    if angles.size < 5:
        return angles, codes, angles.copy(), codes.copy()
    validation_mask = np.zeros(angles.size, dtype=bool)
    validation_mask[::5] = True
    return angles[~validation_mask], codes[~validation_mask], angles[validation_mask], codes[validation_mask]


if nn is not None:

    class SimpleCodeNet(nn.Module):
        """Two-hidden-layer MLP that produces 32 independent 4-state logits."""

        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(NETWORK_INPUT_DIM, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, VARIABLE_COUNT * STATE_COUNT),
            )

        def forward(self, inputs):
            return self.layers(inputs).reshape(-1, VARIABLE_COUNT, STATE_COUNT)

else:

    class SimpleCodeNet:  # pragma: no cover - exercised only before torch installation.
        def __init__(self) -> None:
            require_torch()


def _free_variable_loss(logits, labels):
    return nn.functional.cross_entropy(
        logits[:, FREE_VARIABLES].reshape(-1, STATE_COUNT),
        labels[:, FREE_VARIABLES].reshape(-1),
    )


def _free_variable_accuracy(logits, labels) -> float:
    predicted = torch.argmax(logits[:, FREE_VARIABLES], dim=-1)
    return float((predicted == labels[:, FREE_VARIABLES]).float().mean().item())


def _evaluate_model(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            batch_size = features.shape[0]
            loss_sum += float(_free_variable_loss(logits, labels).item()) * batch_size
            correct_sum += _free_variable_accuracy(logits, labels) * batch_size
            sample_count += batch_size
    return loss_sum / sample_count, correct_sum / sample_count


def train_model(
    model,
    train_angles_deg: np.ndarray,
    train_codes: np.ndarray,
    validation_angles_deg: np.ndarray,
    validation_codes: np.ndarray,
    epochs: int = 300,
    learning_rate: float = 2e-3,
    batch_size: int = 16,
    seed: int = 20260805,
    device: str | None = None,
    early_stopping_patience: int = 40,
) -> dict[str, object]:
    """Train the MLP and return loss/accuracy values for every epoch."""

    require_torch()
    if epochs <= 0 or learning_rate <= 0.0 or batch_size <= 0 or early_stopping_patience <= 0:
        raise ValueError("epochs, learning_rate, batch_size, and patience must be positive")
    torch.manual_seed(seed)
    device_object = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device_object)

    train_dataset = TensorDataset(
        torch.from_numpy(angle_features(train_angles_deg)),
        torch.as_tensor(train_codes, dtype=torch.long),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(angle_features(validation_angles_deg)),
        torch.as_tensor(validation_codes, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True, generator=generator
    )
    train_evaluation_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }
    best_validation_loss = np.inf
    best_validation_accuracy = 0.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for _ in range(int(epochs)):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device_object), labels.to(device_object)
            optimizer.zero_grad()
            loss = _free_variable_loss(model(features), labels)
            loss.backward()
            optimizer.step()
        train_loss, train_accuracy = _evaluate_model(model, train_evaluation_loader, device_object)
        validation_loss, validation_accuracy = _evaluate_model(model, validation_loader, device_object)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_epoch = len(history["train_loss"])
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_validation_loss"] = float(best_validation_loss)
    history["best_validation_accuracy"] = float(best_validation_accuracy)
    return history


def predict_probabilities(model, angle_deg: float, device: str | None = None) -> np.ndarray:
    """Return one 32x4 state-probability matrix for a commanded angle."""

    require_torch()
    angle_deg = validate_angle(angle_deg)
    device_object = torch.device(device or next(model.parameters()).device)
    features = torch.from_numpy(angle_features(angle_deg)).to(device_object)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(features)[0], dim=-1).cpu().numpy()
    probabilities[list(FIXED_VARIABLES)] = [1.0, 0.0, 0.0, 0.0]
    return probabilities


def save_model(model, path: str | Path) -> None:
    require_torch()
    torch.save(
        {
            "format_version": MODEL_FORMAT_VERSION,
            "fourier_harmonics": FOURIER_HARMONICS,
            "model_state": model.state_dict(),
        },
        Path(path),
    )


def load_model(path: str | Path, device: str | None = None):
    require_torch()
    device_object = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(Path(path), map_location=device_object, weights_only=True)
    if checkpoint.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError(
            "This model uses the old 2-input architecture. Retrain it with "
            "Main_simple_neural_ce.py --mode train before running the demo."
        )
    model = SimpleCodeNet().to(device_object)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def plot_training_history(
    history: dict[str, object],
    save_path: str | Path = "simple_training_history.png",
):
    """Plot training/validation loss and free-variable code accuracy."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("plot_training_history requires matplotlib") from error
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].plot(epochs, history["train_loss"], label="Training loss")
    axes[0].plot(epochs, history["validation_loss"], label="Validation loss")
    axes[0].axvline(history["best_epoch"], color="gray", linestyle="--", label="Best epoch")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss during training")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="Training accuracy")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation accuracy")
    axes[1].axvline(history["best_epoch"], color="gray", linestyle="--", label="Best epoch")
    axes[1].set(xlabel="Epoch", ylabel="Free-variable accuracy", ylim=(0.0, 1.02), title="Code accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(Path(save_path), dpi=180, bbox_inches="tight")
    return figure


def plot_code_heatmap(
    model,
    angles_deg: np.ndarray,
    true_codes: np.ndarray,
    save_path: str | Path = "simple_code_heatmap.png",
):
    """Plot true and predicted joint codes across angle."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("plot_code_heatmap requires matplotlib") from error
    predicted = np.asarray([
        np.argmax(predict_probabilities(model, angle), axis=1) for angle in angles_deg
    ])
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 5.6), sharex=True)
    images = []
    for axis, codes, title in (
        (axes[0], true_codes, "True 2-bit codes"),
        (axes[1], predicted, "MLP-predicted 2-bit codes"),
    ):
        image = axis.imshow(
            codes.T,
            aspect="auto",
            origin="lower",
            extent=(float(angles_deg[0]), float(angles_deg[-1]), 1, VARIABLE_COUNT),
            vmin=-0.5,
            vmax=3.5,
            interpolation="nearest",
        )
        images.append(image)
        axis.set(ylabel="Joint variable", title=title)
    axes[1].set_xlabel("Commanded angle (deg)")
    colorbar = figure.colorbar(images[-1], ax=axes, ticks=(0, 1, 2, 3), fraction=0.025)
    colorbar.set_label("2-bit state")
    figure.subplots_adjust(right=0.88, hspace=0.28)
    figure.savefig(Path(save_path), dpi=180, bbox_inches="tight")
    return figure


@dataclass
class CEConfig:
    generations: int = 3
    candidates_per_generation: int = 12
    elite_count: int = 3
    smoothing: float = 0.60
    minimum_probability: float = 0.01

    @property
    def measurement_count(self) -> int:
        return self.generations * self.candidates_per_generation


@dataclass
class CEResult:
    best_code: np.ndarray
    best_power_dBm: float
    measurement_count: int
    measured_codes: np.ndarray
    measured_powers_dBm: np.ndarray
    final_probability: np.ndarray


def _normalize_probability(probability: np.ndarray, minimum: float) -> np.ndarray:
    probability = np.maximum(np.asarray(probability, dtype=float), minimum)
    probability /= probability.sum(axis=1, keepdims=True)
    probability[list(FIXED_VARIABLES)] = [1.0, 0.0, 0.0, 0.0]
    return probability


def run_fixed_ce(
    measure: MeasureFunction,
    initial_probability: np.ndarray,
    seed: int = 20260805,
    config: CEConfig | None = None,
    forced_codes: tuple[np.ndarray, ...] = (),
) -> CEResult:
    """Run exactly three generations x twelve measured candidates by default."""

    config = config or CEConfig()
    probability = np.asarray(initial_probability, dtype=float).copy()
    if probability.shape != (VARIABLE_COUNT, STATE_COUNT):
        raise ValueError("initial_probability must have shape (32, 4)")
    probability = _normalize_probability(probability, config.minimum_probability)
    forced_codes = tuple(validate_joint_code(code) for code in forced_codes)
    rng = np.random.default_rng(seed)
    all_codes, all_powers = [], []

    for _ in range(config.generations):
        samples = np.empty((config.candidates_per_generation, VARIABLE_COUNT), dtype=int)
        for variable in range(VARIABLE_COUNT):
            samples[:, variable] = rng.choice(
                STATE_COUNT, config.candidates_per_generation, p=probability[variable]
            )
        samples[:, list(FIXED_VARIABLES)] = 0
        for index, code in enumerate(forced_codes[: config.candidates_per_generation]):
            samples[index] = code
        powers = np.asarray([float(measure(code.copy())) for code in samples])
        all_codes.append(samples.copy())
        all_powers.append(powers.copy())
        elite_count = min(config.elite_count, config.candidates_per_generation)
        elite = samples[np.argsort(powers)[-elite_count:]]
        elite_probability = np.column_stack([
            np.mean(elite == state, axis=0) for state in range(STATE_COUNT)
        ])
        probability = (
            (1.0 - config.smoothing) * probability
            + config.smoothing * elite_probability
        )
        probability = _normalize_probability(probability, config.minimum_probability)

    measured_codes = np.vstack(all_codes)
    measured_powers = np.concatenate(all_powers)
    best_index = int(np.argmax(measured_powers))
    return CEResult(
        best_code=measured_codes[best_index].copy(),
        best_power_dBm=float(measured_powers[best_index]),
        measurement_count=int(measured_powers.size),
        measured_codes=measured_codes,
        measured_powers_dBm=measured_powers,
        final_probability=probability,
    )


def cold_start_ce(
    measure: MeasureFunction,
    seed: int = 20260805,
    config: CEConfig | None = None,
) -> CEResult:
    probability = np.full((VARIABLE_COUNT, STATE_COUNT), 1.0 / STATE_COUNT)
    return run_fixed_ce(measure, probability, seed=seed, config=config)


def warm_start_ce(
    model,
    angle_deg: float,
    previous_code: np.ndarray,
    measure: MeasureFunction,
    seed: int = 20260805,
    config: CEConfig | None = None,
) -> CEResult:
    previous_code = validate_joint_code(previous_code)
    network_probability = predict_probabilities(model, angle_deg)
    previous_probability = np.eye(STATE_COUNT)[previous_code]
    initial_probability = (
        0.70 * network_probability
        + 0.20 * previous_probability
        + 0.10 / STATE_COUNT
    )
    network_code = validate_joint_code(np.argmax(network_probability, axis=1))
    return run_fixed_ce(
        measure,
        initial_probability,
        seed=seed,
        config=config,
        forced_codes=(network_code, previous_code),
    )


def make_simulated_measurement(
    angle_deg: float,
    transmit_power_dBm: float = 0.0,
    noise_power_dBm: float = -90.0,
    measurement_noise_std_dB: float = 0.15,
    seed: int = 20260805,
) -> MeasureFunction:
    """Create a spectrum-analyzer-like power function for CE comparisons."""

    angle_deg = validate_angle(angle_deg)
    angle_rad = np.deg2rad(angle_deg)
    h12, _, _, _ = build_far_field_channel(angle_rad)
    transmit_power_W = 10.0 ** ((float(transmit_power_dBm) - 30.0) / 10.0)
    noise_power_W = 10.0 ** ((float(noise_power_dBm) - 30.0) / 10.0)
    rng = np.random.default_rng(seed)

    def measure(code: np.ndarray) -> float:
        code = validate_joint_code(code)
        v1 = Compensation_Phasors[code[:Columns]]
        v2 = Compensation_Phasors[code[Columns:]]
        h_eff, _, _ = link_metrics(v1, v2, angle_rad, h12)
        signal_power_W = transmit_power_W * abs(h_eff) ** 2
        total_power_dBm = 10.0 * np.log10(max(signal_power_W + noise_power_W, 1e-30)) + 30.0
        return float(total_power_dBm + rng.normal(0.0, measurement_noise_std_dB))

    return measure
