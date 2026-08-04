import unittest

import numpy as np

from relay_sim.Arc_Experiment import (
    ArcExperimentSimulator,
    DomainRandomization,
    generate_training_examples,
    split_examples_by_trajectory,
)
from relay_sim.Neural_CE import (
    BudgetedNeuralCEOptimizer,
    CalibrationRecord,
    NeuralCodePrior,
    TrajectoryState,
    full_ce_optimize,
    physics_reference_code,
)


class NeuralCETest(unittest.TestCase):
    def setUp(self) -> None:
        self.noiseless = DomainRandomization(
            command_angle_error_std_deg=0.0,
            radius_error_std_fraction=0.0,
            phase_error_std_deg=0.0,
            gain_error_std_dB=0.0,
            receiver_drift_std_dB=0.0,
            measurement_noise_std_dB=0.0,
            failed_element_probability=0.0,
        )

    def test_physics_code_respects_fixed_references_and_bounds(self) -> None:
        for angle in (-60.0, 0.0, 60.0):
            code = physics_reference_code(angle)
            self.assertEqual(code.shape, (32,))
            self.assertEqual(int(code[0]), 0)
            self.assertEqual(int(code[16]), 0)
            self.assertTrue(np.all((0 <= code) & (code <= 3)))

    def test_simulator_rejects_out_of_arc_commands(self) -> None:
        simulator = ArcExperimentSimulator(1, self.noiseless)
        with self.assertRaises(ValueError):
            simulator.move_to(60.01)
        with self.assertRaises(ValueError):
            simulator.move_to(-60.01)

    def test_full_ce_reports_exact_measurement_count(self) -> None:
        target = physics_reference_code(30.0)

        def measure(code: np.ndarray) -> float:
            return -float(np.count_nonzero(code != target))

        result = full_ce_optimize(measure, seed=3, population_size=12, max_iterations=4)
        self.assertEqual(result.measurement_count, 48)
        self.assertEqual(result.measured_codes.shape, (48, 32))
        self.assertEqual(result.measured_powers_dBm.shape, (48,))

    def test_budgeted_optimizer_never_exceeds_44_reads(self) -> None:
        simulator = ArcExperimentSimulator(4, self.noiseless)
        state = TrajectoryState()
        simulator.move_to(0.0, true_angle_deg=0.0)
        zero_code = physics_reference_code(0.0)
        state.add_calibration(CalibrationRecord(0.0, zero_code, simulator.measure(zero_code)))
        simulator.move_to(10.0, true_angle_deg=10.0)
        optimizer = BudgetedNeuralCEOptimizer(NeuralCodePrior(seed=5), seed=6)
        result = optimizer.optimize(state, 10.0, 10.0, simulator.measure)
        self.assertIn(result.measurement_count, (20, 32, 44))
        self.assertLessEqual(result.measurement_count, 44)
        self.assertTrue(any(np.array_equal(result.best_code, item.code) for item in result.observations))
        self.assertEqual(int(result.best_code[0]), 0)
        self.assertEqual(int(result.best_code[16]), 0)

    def test_optimizer_enforces_command_continuity(self) -> None:
        simulator = ArcExperimentSimulator(7, self.noiseless)
        state = TrajectoryState()
        zero_code = physics_reference_code(0.0)
        state.add_calibration(CalibrationRecord(0.0, zero_code, simulator.clean_power_dBm(zero_code)))
        simulator.move_to(10.0, true_angle_deg=10.0)
        optimizer = BudgetedNeuralCEOptimizer(NeuralCodePrior(seed=8), seed=9)
        optimizer.optimize(state, 10.0, 10.0, simulator.measure)
        simulator.move_to(25.0, true_angle_deg=25.0)
        with self.assertRaises(ValueError):
            optimizer.optimize(state, 25.0, 5.0, simulator.measure)

    def test_quick_training_data_stays_inside_arc(self) -> None:
        examples = generate_training_examples(
            trajectory_count=1,
            step_deg=30.0,
            seed=10,
            full_ce_labels=False,
            candidates_per_position=6,
        )
        self.assertGreater(len(examples), 0)
        self.assertTrue(all(-60.0 <= example.angle_cmd_deg <= 60.0 for example in examples))
        self.assertTrue(all(example.candidate_codes.shape == (6, 32) for example in examples))

    def test_trajectory_split_keeps_each_path_in_one_partition(self) -> None:
        examples = generate_training_examples(
            trajectory_count=3,
            step_deg=60.0,
            seed=11,
            full_ce_labels=False,
            candidates_per_position=4,
        )
        training, validation = split_examples_by_trajectory(examples, validation_fraction=0.34, seed=12)
        training_ids = {example.trajectory_id for example in training}
        validation_ids = {example.trajectory_id for example in validation}
        self.assertTrue(training_ids)
        self.assertTrue(validation_ids)
        self.assertTrue(training_ids.isdisjoint(validation_ids))


if __name__ == "__main__":
    unittest.main()
