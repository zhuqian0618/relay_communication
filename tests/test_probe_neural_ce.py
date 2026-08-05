import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ce_demo import cold_start_ce, probe_assisted_ce
from model import (
    FIXED_COLUMNS,
    FREE_COLUMNS,
    NETWORK_INPUT_DIM,
    SimpleCodeNet,
    build_probe_codes,
    build_probe_features,
    load_model,
    reference_joint_code,
    restore_joint_column_code,
    save_model,
)
from train import (
    ProbeDatasetArrays,
    build_simulation_dataset,
    load_measured_csv,
    measured_csv_header,
    train_model,
)


class ProbeFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_code = reference_joint_code(0.0)
        self.previous_power = -50.0
        self.baseline = np.asarray([-52.0, -51.8])
        self.probes = np.asarray([-51.0, -53.0, -50.5, -54.0, -51.5, -52.5])

    def test_probe_codes_and_fixed_columns(self) -> None:
        probes = build_probe_codes(self.previous_code)
        self.assertEqual(probes.shape, (6, 32))
        self.assertTrue(np.all(probes[:, list(FIXED_COLUMNS)] == 0))
        self.assertTrue(np.all(np.count_nonzero(probes != self.previous_code, axis=1) == 10))

    def test_input_has_128_values_and_no_angle(self) -> None:
        features = build_probe_features(
            self.previous_code, self.previous_power, self.baseline, self.probes
        )
        self.assertEqual(features.shape, (NETWORK_INPUT_DIM,))
        self.assertEqual(NETWORK_INPUT_DIM, 128)

    def test_common_power_offset_does_not_change_features(self) -> None:
        original = build_probe_features(
            self.previous_code, self.previous_power, self.baseline, self.probes
        )
        shifted = build_probe_features(
            self.previous_code,
            self.previous_power + 17.0,
            self.baseline + 17.0,
            self.probes + 17.0,
        )
        np.testing.assert_allclose(original, shifted, atol=1e-7)

    def test_restore_joint_column_code_keeps_reference_columns_zero(self) -> None:
        joint = restore_joint_column_code(np.full(30, 3))
        self.assertEqual(joint.shape, (32,))
        self.assertTrue(np.all(joint[list(FIXED_COLUMNS)] == 0))


class DatasetAndCSVTest(unittest.TestCase):
    def test_simulation_contains_directions_steps_and_boundaries(self) -> None:
        data = build_simulation_dataset(9, "test")
        self.assertEqual(data.features.shape[1], 128)
        self.assertEqual(data.target_codes.shape[1], 30)
        self.assertGreater(len(data), 1000)
        self.assertTrue(np.all((data.target_codes >= 0) & (data.target_codes <= 3)))
        self.assertTrue(np.any(data.angles_deg == -60.0))
        self.assertTrue(np.any(data.angles_deg == 60.0))
        self.assertTrue(any("positive" in run_id for run_id in data.run_ids))
        self.assertTrue(any("negative" in run_id for run_id in data.run_ids))
        self.assertTrue(any("turn" in run_id for run_id in data.run_ids))

    def test_measured_csv_round_trip(self) -> None:
        previous = reference_joint_code(0.0)
        target = reference_joint_code(5.0)
        row = (
            ["flight_1", -50.0, -51.0, -50.8]
            + [-50.0, -52.0, -49.5, -53.0, -50.4, -51.5]
            + previous.tolist()
            + target.tolist()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measured.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(measured_csv_header())
                writer.writerow(row)
            data = load_measured_csv(path)
        self.assertEqual(len(data), 1)
        self.assertEqual(data.features.shape, (1, 128))
        np.testing.assert_array_equal(data.target_codes[0], target[FREE_COLUMNS])


class ModelAndCETest(unittest.TestCase):
    def test_output_shape(self) -> None:
        output = SimpleCodeNet()(torch.ones((3, 128)))
        self.assertEqual(tuple(output.shape), (3, 30, 4))

    def test_save_load_prediction_is_identical(self) -> None:
        torch.manual_seed(3)
        model = SimpleCodeNet()
        sample = torch.randn((2, 128))
        before = model(sample).detach().numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            save_model(model, path)
            restored = load_model(path, device="cpu")
            after = restored(sample).detach().numpy()
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)

    def test_short_training_reduces_loss(self) -> None:
        rng = np.random.default_rng(4)
        previous = rng.integers(0, 4, size=(96, 30), dtype=np.int64)
        targets = previous.copy()
        targets[:, 0] = (targets[:, 0] + (np.arange(96) % 2)) % 4
        features = np.zeros((96, 128), dtype=np.float32)
        features[:, :120] = np.eye(4)[previous].reshape(96, -1)
        features[:, 120] = np.arange(96) % 2
        data = ProbeDatasetArrays(features, targets, previous, np.asarray(["r"] * 96), np.zeros(96))
        model = SimpleCodeNet()
        history = train_model(
            model, data, data, epochs=12, learning_rate=3e-3,
            batch_size=24, patience=12, seed=4, device="cpu"
        )
        self.assertLess(history["train_loss"][-1], history["train_loss"][0])

    def test_both_ce_methods_use_exactly_36_real_measure_calls(self) -> None:
        target = reference_joint_code(10.0)

        def make_counter():
            seen = []

            def measure(code: np.ndarray) -> float:
                seen.append(code.copy())
                return -float(np.count_nonzero(code != target))

            return measure, seen

        cold_measure, cold_seen = make_counter()
        cold = cold_start_ce(cold_measure, seed=2)
        warm_measure, warm_seen = make_counter()
        warm = probe_assisted_ce(
            SimpleCodeNet(), reference_joint_code(0.0), -20.0, warm_measure, seed=2
        )
        self.assertEqual(len(cold_seen), 36)
        self.assertEqual(len(warm_seen), 36)
        self.assertEqual(cold.measurement_count, 36)
        self.assertEqual(warm.measurement_count, 36)
        self.assertTrue(any(np.array_equal(warm.best_code, code) for code in warm_seen))

    def test_invalid_probe_reading_disables_network_without_exceeding_budget(self) -> None:
        calls = 0

        def measure(code: np.ndarray) -> float:
            nonlocal calls
            calls += 1
            return np.nan if calls == 3 else -float(np.sum(code))

        result = probe_assisted_ce(
            SimpleCodeNet(), reference_joint_code(0.0), -10.0, measure, seed=3
        )
        self.assertEqual(calls, 36)
        self.assertEqual(result.used_network_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
