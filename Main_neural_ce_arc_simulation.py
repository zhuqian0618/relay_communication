"""Train and evaluate neural-assisted CE on the horizontal ±60° UAV arc."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from relay_sim.Arc_Experiment import (
    ArcExperimentSimulator,
    evaluate_prior,
    experiment_path,
    generate_training_examples,
    split_examples_by_trajectory,
)
from relay_sim.Neural_CE import (
    BudgetedNeuralCEOptimizer,
    CalibrationRecord,
    NeuralCodePrior,
    TrajectoryState,
    full_ce_optimize,
    initialize_zero_anchor,
    physics_reference_code,
)


def create_prior(backend: str, seed: int):
    if backend in {"auto", "torch"}:
        try:
            from relay_sim.Torch_Neural_Prior import TorchNeuralCodePrior
        except ImportError:
            if backend == "torch":
                raise RuntimeError(
                    "PyTorch backend requested. Install requirements-pytorch.txt first."
                ) from None
        else:
            return TorchNeuralCodePrior(seed=seed), "torch"
    return NeuralCodePrior(seed=seed), "numpy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "torch", "numpy"), default="auto")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--step-deg", type=float, default=10.0)
    parser.add_argument("--training-trajectories", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--quick-labels", action="store_true", help="Use analytical labels only for a smoke test")
    parser.add_argument("--full-baseline", action="store_true", help="Run 1800-measurement CE at every test angle")
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    prior, backend = create_prior(args.backend, args.seed)
    print(f"Neural backend: {backend}")
    print("Generating trajectory training data...")
    training_examples = generate_training_examples(
        trajectory_count=args.training_trajectories,
        step_deg=args.step_deg,
        seed=args.seed,
        full_ce_labels=not args.quick_labels,
    )
    training_split, validation_split = split_examples_by_trajectory(
        training_examples, validation_fraction=0.2, seed=args.seed
    )
    print(f"Trajectory-safe split: train={len(training_split)}, validation={len(validation_split)}")
    metrics = prior.fit(training_split, epochs=args.epochs)
    print("Training metrics:", metrics)
    if validation_split:
        validation_metrics = evaluate_prior(prior, validation_split)
        print("Held-out trajectory metrics:", validation_metrics)
    if args.model_out:
        prior.save(args.model_out)
        print(f"Saved model: {args.model_out}")

    simulator = ArcExperimentSimulator(args.seed + 99991)
    state = TrajectoryState()
    simulator.move_to(0.0, true_angle_deg=0.0)
    zero_result = initialize_zero_anchor(state, simulator.measure, seed=args.seed + 1)
    zero_anchor = state.calibrations[0.0]
    print(
        f"0-degree calibration: {zero_anchor.measurement_count} reads, "
        f"{zero_anchor.power_dBm:.2f} dBm, repeat span={zero_anchor.noise_span_dB:.2f} dB"
    )

    optimizer = BudgetedNeuralCEOptimizer(prior, seed=args.seed + 2)
    path = experiment_path(args.step_deg)
    angles, powers, reference_powers, measurement_counts, gaps = [0.0], [zero_anchor.power_dBm], [], [zero_anchor.measurement_count], []
    reference_at_zero = simulator.clean_power_dBm(physics_reference_code(0.0))
    reference_powers.append(reference_at_zero)
    gaps.append(reference_at_zero - zero_anchor.power_dBm)

    for angle in path[1:]:
        delta = float(angle - state.angle_cmd_deg)
        simulator.move_to(float(angle))
        result = optimizer.optimize(state, float(angle), delta, simulator.measure)
        if args.full_baseline:
            baseline = full_ce_optimize(
                lambda code: simulator.clean_power_dBm(code),
                seed=args.seed + int(round((angle + 60.0) * 10)),
            ).best_power_dBm
        else:
            baseline = simulator.clean_power_dBm(physics_reference_code(float(angle)))
        angles.append(float(angle))
        powers.append(result.best_power_dBm)
        reference_powers.append(baseline)
        measurement_counts.append(result.measurement_count)
        gaps.append(baseline - result.best_power_dBm)
        print(
            f"angle={angle:+6.1f} deg, reads={result.measurement_count:2d}, "
            f"power={result.best_power_dBm:7.2f} dBm, gap={baseline-result.best_power_dBm:+5.2f} dB, "
            f"confidence={result.confidence:.3f}"
        )

    angles_array = np.asarray(angles)
    gaps_array = np.asarray(gaps[1:])
    counts_array = np.asarray(measurement_counts[1:])
    print(f"Normal-update read range: {counts_array.min()}-{counts_array.max()}")
    print(f"Median reference gap: {np.median(gaps_array):.3f} dB")
    print(f"95th-percentile reference gap: {np.percentile(gaps_array, 95):.3f} dB")

    simulator.move_to(0.0, true_angle_deg=0.0)
    zero_check = optimizer.check_zero_anchor(state, simulator.measure)
    print(
        f"Return-to-zero drop={zero_check.power_drop_dB:.2f} dB, "
        f"recalibration_required={zero_check.calibration_required}"
    )

    if args.no_plot:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib; use --no-plot on a headless computer") from error
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 8.2), sharex=True)
    axes[0].plot(angles, powers, "-o", label="Neural-assisted CE")
    axes[0].plot(angles, reference_powers, "--", label="Reference")
    axes[0].set_ylabel("Received power (dBm)")
    axes[0].legend()
    axes[1].step(angles, measurement_counts, where="mid", color="#E76F51")
    axes[1].axhline(44, color="#6B7280", linestyle="--", label="Normal budget")
    axes[1].set_ylabel("Spectrum reads")
    axes[1].legend()
    axes[2].plot(angles, gaps, "-s", color="#2A9D8F")
    axes[2].axhline(1.0, color="#6B7280", linestyle="--", label="1 dB target")
    axes[2].set(xlabel="Commanded UAV2 angle (deg)", ylabel="Reference gap (dB)")
    axes[2].legend()
    figure.suptitle("Neural-assisted CE on 0°→+60°→0°→-60°→0°")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
