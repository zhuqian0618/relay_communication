import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ce_demo import CE_CONFIG, PowerDropMonitor, neural_initial_probability
from model import (
    FIXED_COLUMNS,
    FREE_COLUMNS,
    NETWORK_INPUT_DIM,
    SimpleCodeNet,
    build_code_history_features,
    load_model,
    reference_joint_code,
    save_model,
)
from relay_sim.CE_Optimizer import (
    CEConfig,
    make_scalar_measurement_evaluator,
    run_ce,
    uniform_probability,
)
from train import (
    EvolutionDatasetArrays,
    load_dataset_npz,
    save_dataset_npz,
    train_model,
)


class CodeHistoryFeatureTest(unittest.TestCase):
    def test_first_trigger_has_241_values_and_invalid_history(self) -> None:
        latest = reference_joint_code(0.0)
        features = build_code_history_features(latest, None)
        self.assertEqual(features.shape, (241,))
        self.assertEqual(NETWORK_INPUT_DIM, 241)
        self.assertEqual(features[-1], 0.0)
        expected_keep = np.tile([1.0, 0.0, 0.0, 0.0], 30)
        np.testing.assert_array_equal(features[120:240], expected_keep)

    def test_valid_history_encodes_modulo_four_change(self) -> None:
        previous = reference_joint_code(0.0)
        latest = previous.copy()
        latest[FREE_COLUMNS[0]] = 3
        latest[FREE_COLUMNS[1]] = 1
        features = build_code_history_features(latest, previous)
        changes = features[120:240].reshape(30, 4)
        self.assertEqual(features[-1], 1.0)
        self.assertEqual(int(np.argmax(changes[0])), 3)
        self.assertEqual(int(np.argmax(changes[1])), 1)

    def test_network_output_and_model_round_trip(self) -> None:
        torch.manual_seed(3)
        model = SimpleCodeNet()
        sample = torch.randn((2, NETWORK_INPUT_DIM))
        before = model(sample).detach().numpy()
        self.assertEqual(before.shape, (2, 30, 4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            save_model(model, path)
            restored = load_model(path, device="cpu")
            after = restored(sample).detach().numpy()
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)

    def test_neural_probability_keeps_both_reference_columns_fixed(self) -> None:
        model = SimpleCodeNet()
        latest = reference_joint_code(0.0)
        probability, features = neural_initial_probability(model, latest, None)
        self.assertEqual(features[-1], 0.0)
        np.testing.assert_array_equal(probability[list(FIXED_COLUMNS)], [[1, 0, 0, 0]] * 2)


class PowerTriggerTest(unittest.TestCase):
    def test_only_two_dB_drop_triggers(self) -> None:
        monitor = PowerDropMonitor(threshold_dB=2.0, window_size=5)
        monitor.calibrate(np.full(5, -50.0))
        for value in [-51.8] * 5:
            triggered = monitor.update(value)
        self.assertFalse(triggered)
        monitor.calibrate(np.full(5, -50.0))
        for value in [-52.0] * 5:
            triggered = monitor.update(value)
        self.assertTrue(triggered)
        self.assertAlmostEqual(monitor.last_drop_dB, 2.0)

    def test_trigger_is_invariant_to_common_power_offset(self) -> None:
        first = PowerDropMonitor()
        second = PowerDropMonitor()
        first.calibrate(np.full(5, -50.0))
        second.calibrate(np.full(5, -33.0))
        first_result = second_result = False
        for _ in range(5):
            first_result = first.update(-52.1)
            second_result = second.update(-35.1)
        self.assertEqual(first_result, second_result)
        self.assertTrue(first_result)


class SharedCETest(unittest.TestCase):
    @staticmethod
    def target_evaluator(target: np.ndarray):
        def evaluate(candidates: np.ndarray, _rng: np.random.Generator):
            scores = -np.count_nonzero(candidates != target, axis=1).astype(float)
            return scores, scores - 50.0

        return evaluate

    def test_default_config_matches_original_main_program(self) -> None:
        self.assertEqual(CE_CONFIG.population_size, 50)
        self.assertEqual(CE_CONFIG.max_iterations, 25)
        self.assertEqual(CE_CONFIG.elite_fraction, 0.20)
        self.assertEqual(CE_CONFIG.smoothing, 0.60)
        self.assertEqual(CE_CONFIG.minimum_probability, 0.02)
        self.assertEqual(CE_CONFIG.pilot_symbols_L, 4)
        self.assertEqual(CE_CONFIG.convergence_probability, 0.90)

    def test_reference_columns_are_fixed_and_best_code_was_measured(self) -> None:
        target = reference_joint_code(10.0)
        config = CEConfig(population_size=20, max_iterations=5)
        result = run_ce(
            self.target_evaluator(target), uniform_probability(), config=config, seed=2
        )
        self.assertTrue(np.all(result.measured_codes[:, list(FIXED_COLUMNS)] == 0))
        np.testing.assert_array_equal(
            result.final_probability[list(FIXED_COLUMNS)], [[1, 0, 0, 0]] * 2
        )
        self.assertTrue(any(np.array_equal(result.best_code, code) for code in result.measured_codes))
        self.assertEqual(result.candidate_measurement_count, 20 * result.iteration_count)
        self.assertEqual(result.raw_read_count, 4 * result.candidate_measurement_count)

    def test_stop_condition_is_probability_or_maximum_iterations(self) -> None:
        target = reference_joint_code(5.0)
        fast = CEConfig(
            population_size=20, max_iterations=8, convergence_probability=0.26
        )
        fast_result = run_ce(
            self.target_evaluator(target), uniform_probability(), config=fast, seed=1
        )
        self.assertTrue(fast_result.reached_probability_threshold)
        self.assertLess(fast_result.iteration_count, fast.max_iterations)

        capped = CEConfig(
            population_size=20, max_iterations=3, convergence_probability=1.0
        )
        capped_result = run_ce(
            self.target_evaluator(target), uniform_probability(), config=capped, seed=1
        )
        self.assertEqual(capped_result.iteration_count, 3)
        self.assertTrue(capped_result.reached_max_iterations)

    def test_spectrum_analyzer_adapter_reads_each_candidate_L_times(self) -> None:
        calls = []

        def measure(code):
            calls.append(np.asarray(code).copy())
            return -60.0

        config = CEConfig(population_size=10, max_iterations=1, pilot_symbols_L=4)
        evaluator = make_scalar_measurement_evaluator(measure, config=config)
        result = run_ce(evaluator, uniform_probability(), config=config, seed=12)
        self.assertEqual(len(calls), result.raw_read_count)
        self.assertEqual(result.raw_read_count, 40)


class TrainingTest(unittest.TestCase):
    @staticmethod
    def synthetic_dataset() -> EvolutionDatasetArrays:
        rng = np.random.default_rng(4)
        previous = rng.integers(0, 4, size=(96, 30), dtype=np.int64)
        targets = previous.copy()
        targets[:, 0] = (targets[:, 0] + (np.arange(96) % 2)) % 4
        features = np.zeros((96, NETWORK_INPUT_DIM), dtype=np.float32)
        features[:, :120] = np.eye(4)[previous].reshape(96, -1)
        changes = (previous - np.roll(previous, 1, axis=0)) % 4
        features[:, 120:240] = np.eye(4)[changes].reshape(96, -1)
        features[:, -1] = 1.0
        probabilities = np.eye(4, dtype=np.float32)[targets]
        return EvolutionDatasetArrays(
            features, targets, previous, probabilities,
            np.asarray(["run"] * 96), np.zeros(96),
        )

    def test_short_training_reduces_combined_loss(self) -> None:
        data = self.synthetic_dataset()
        history = train_model(
            SimpleCodeNet(), data, data, epochs=15, learning_rate=3e-3,
            batch_size=24, patience=15, seed=4, device="cpu",
        )
        self.assertLess(history["train_loss"][-1], history["train_loss"][0])

    def test_npz_round_trip(self) -> None:
        data = self.synthetic_dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            save_dataset_npz(data, path)
            restored = load_dataset_npz(path)
        np.testing.assert_array_equal(data.features, restored.features)
        np.testing.assert_array_equal(data.target_probabilities, restored.target_probabilities)


if __name__ == "__main__":
    unittest.main()
