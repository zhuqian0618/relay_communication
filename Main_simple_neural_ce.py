"""Beginner entry point for the angle -> MLP -> 36-measurement CE workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from relay_sim.Simple_Neural_CE import (
    SimpleCodeNet,
    build_simulation_datasets,
    cold_start_ce,
    load_measured_csv,
    load_model,
    make_simulated_measurement,
    plot_code_heatmap,
    plot_training_history,
    predict_probabilities,
    reference_joint_code,
    save_model,
    split_measured_dataset,
    train_model,
    warm_start_ce,
)


def build_test_path(step_deg: float) -> np.ndarray:
    """Build 0 -> +60 -> 0 -> -60 -> 0 without duplicate turning points."""

    if step_deg <= 0.0:
        raise ValueError("test step must be positive")

    def segment(start: float, stop: float) -> np.ndarray:
        direction = 1.0 if stop >= start else -1.0
        values = np.arange(start, stop + direction * 1e-9, direction * step_deg)
        if not np.isclose(values[-1], stop):
            values = np.append(values, stop)
        return values

    return np.concatenate((
        segment(0.0, 60.0),
        segment(60.0, 0.0)[1:],
        segment(0.0, -60.0)[1:],
        segment(-60.0, 0.0)[1:],
    ))


def train_command(args: argparse.Namespace) -> None:
    # Step 1: create dense simulation labels.  Power and noise are deliberately
    # absent because they do not change the ideal optimal phase code.
    train_angles, train_codes, validation_angles, validation_codes = build_simulation_datasets(
        args.angle_step
    )
    print(f"Simulation training samples: {train_angles.size}")
    print(f"Offset validation samples: {validation_angles.size}")

    # Step 2: train the simple two-hidden-layer MLP.
    model = SimpleCodeNet()
    history = train_model(
        model,
        train_angles,
        train_codes,
        validation_angles,
        validation_codes,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(
        f"Simulation training complete: best epoch={history['best_epoch']}, "
        f"validation loss={history['best_validation_loss']:.4f}, "
        f"validation accuracy={history['best_validation_accuracy']:.3f}"
    )
    if not args.no_plot:
        plot_training_history(history, args.history_plot)
        plot_code_heatmap(model, validation_angles, validation_codes, args.heatmap_plot)
        print(f"Saved training curves: {args.history_plot}")
        print(f"Saved code heatmap: {args.heatmap_plot}")

    # Step 3: optionally fine-tune on sparse full-CE experimental labels.
    if args.real_data is not None:
        real_angles, real_codes = load_measured_csv(args.real_data)
        real_train_angles, real_train_codes, real_validation_angles, real_validation_codes = (
            split_measured_dataset(real_angles, real_codes)
        )
        fine_tune_history = train_model(
            model,
            real_train_angles,
            real_train_codes,
            real_validation_angles,
            real_validation_codes,
            epochs=args.fine_tune_epochs,
            learning_rate=args.fine_tune_learning_rate,
            batch_size=min(args.batch_size, real_train_angles.size),
            seed=args.seed + 1,
        )
        print(
            f"Real-data fine-tuning complete: samples={real_angles.size}, "
            f"best validation accuracy={fine_tune_history['best_validation_accuracy']:.3f}"
        )
        if not args.no_plot:
            plot_training_history(fine_tune_history, args.fine_tune_plot)
            print(f"Saved fine-tuning curves: {args.fine_tune_plot}")

    save_model(model, args.model)
    print(f"Saved model: {args.model}")


def demo_command(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    path = build_test_path(args.test_step)
    scenarios = (
        ("High SNR", -100.0),
        ("Default SNR", -90.0),
        ("Low SNR", -70.0),
    )
    scenario_results = []

    for scenario_index, (name, noise_power_dBm) in enumerate(scenarios):
        previous_code = reference_joint_code(0.0)
        warm_powers, cold_powers = [], []
        for angle_index, angle in enumerate(path):
            common_seed = args.seed + scenario_index * 100000 + angle_index
            warm_measure = make_simulated_measurement(
                float(angle),
                transmit_power_dBm=args.transmit_power,
                noise_power_dBm=noise_power_dBm,
                seed=common_seed,
            )
            cold_measure = make_simulated_measurement(
                float(angle),
                transmit_power_dBm=args.transmit_power,
                noise_power_dBm=noise_power_dBm,
                seed=common_seed,
            )
            warm = warm_start_ce(
                model, float(angle), previous_code, warm_measure, seed=common_seed
            )
            cold = cold_start_ce(cold_measure, seed=common_seed)
            if warm.measurement_count != 36 or cold.measurement_count != 36:
                raise RuntimeError("both CE methods must use exactly 36 measurements")
            previous_code = warm.best_code
            warm_powers.append(warm.best_power_dBm)
            cold_powers.append(cold.best_power_dBm)
        warm_powers = np.asarray(warm_powers)
        cold_powers = np.asarray(cold_powers)
        improvement = warm_powers - cold_powers
        scenario_results.append((name, noise_power_dBm, warm_powers, cold_powers))
        print(
            f"{name:11s}: noise={noise_power_dBm:6.1f} dBm, "
            f"mean warm-cold gain={improvement.mean():+.3f} dB, "
            f"warm-win ratio={np.mean(improvement > 0):.3f}, reads/location=36"
        )

    if args.no_plot:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("demo plotting requires matplotlib") from error
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True)
    for axis, (name, noise_power_dBm, warm_powers, cold_powers) in zip(axes, scenario_results):
        axis.plot(path, warm_powers, "-o", ms=3, label="MLP warm-start CE")
        axis.plot(path, cold_powers, "--s", ms=3, label="Cold-start CE")
        axis.set(ylabel="Power (dBm)", title=f"{name}: noise={noise_power_dBm:.0f} dBm")
        axis.legend()
    axes[-1].set_xlabel("Commanded UAV2 angle (deg)")
    figure.tight_layout()
    figure.savefig(args.demo_plot, dpi=180, bbox_inches="tight")
    print(f"Saved CE comparison: {args.demo_plot}")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "demo"), default="train")
    parser.add_argument("--model", type=Path, default=Path("simple_code_net.pt"))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--no-plot", action="store_true")

    # Training options.
    parser.add_argument("--angle-step", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--real-data", type=Path)
    parser.add_argument("--fine-tune-epochs", type=int, default=100)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=2e-4)
    parser.add_argument("--history-plot", type=Path, default=Path("simple_training_history.png"))
    parser.add_argument("--fine-tune-plot", type=Path, default=Path("simple_finetune_history.png"))
    parser.add_argument("--heatmap-plot", type=Path, default=Path("simple_code_heatmap.png"))

    # Demo options.  Transmit and noise powers affect measured performance, not
    # the two neural-network angle inputs.
    parser.add_argument("--test-step", type=float, default=10.0)
    parser.add_argument("--transmit-power", type=float, default=0.0)
    parser.add_argument("--demo-plot", type=Path, default=Path("simple_ce_comparison.png"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        train_command(args)
    else:
        demo_command(args)


if __name__ == "__main__":
    main()
