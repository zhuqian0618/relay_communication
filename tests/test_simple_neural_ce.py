import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from relay_sim.Simple_Neural_CE import (
    CEConfig,
    FIXED_VARIABLES,
    SimpleCodeNet,
    angle_features,
    build_simulation_datasets,
    cold_start_ce,
    load_measured_csv,
    load_model,
    reference_joint_code,
    save_model,
    train_model,
)

try:
    import torch
except ImportError:
    torch = None


class DatasetAndCETest(unittest.TestCase):
    def test_default_simulation_dataset(self) -> None:
        train_angles, train_codes, validation_angles, validation_codes = build_simulation_datasets()
        self.assertEqual(train_angles.size, 241)
        self.assertEqual(validation_angles.size, 240)
        self.assertAlmostEqual(train_angles[0], -60.0)
        self.assertAlmostEqual(train_angles[-1], 60.0)
        self.assertEqual(train_codes.shape, (241, 32))
        self.assertEqual(validation_codes.shape, (240, 32))
        self.assertTrue(np.all((train_codes >= 0) & (train_codes <= 3)))
        self.assertTrue(np.all(train_codes[:, list(FIXED_VARIABLES)] == 0))
        self.assertEqual(angle_features(train_angles).shape, (241, 48))

    def test_reference_code_validates_angle(self) -> None:
        with self.assertRaises(ValueError):
            reference_joint_code(61.0)

    def test_cold_ce_uses_exactly_36_measurements(self) -> None:
        target = reference_joint_code(20.0)
        seen = []

        def measure(code: np.ndarray) -> float:
            seen.append(code.copy())
            return -float(np.count_nonzero(code != target))

        result = cold_start_ce(measure, seed=4)
        self.assertEqual(result.measurement_count, 36)
        self.assertEqual(len(seen), 36)
        self.assertTrue(any(np.array_equal(result.best_code, code) for code in seen))
        self.assertEqual(result.measured_codes.shape, (36, 32))

    def test_csv_loader(self) -> None:
        header = ["angle_deg"] + [f"c{index}" for index in range(32)]
        code = reference_joint_code(0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measured.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerow([0.0, *code.tolist()])
            angles, codes = load_measured_csv(path)
        self.assertEqual(angles.tolist(), [0.0])
        self.assertTrue(np.array_equal(codes[0], code))

    def test_csv_rejects_invalid_state(self) -> None:
        header = ["angle_deg"] + [f"c{index}" for index in range(32)]
        code = reference_joint_code(0.0)
        code[3] = 8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerow([0.0, *code.tolist()])
            with self.assertRaisesRegex(ValueError, "outside 0..3"):
                load_measured_csv(path)


@unittest.skipIf(torch is None, "PyTorch is not installed in the verification runtime")
class TorchModelTest(unittest.TestCase):
    def test_output_shape(self) -> None:
        model = SimpleCodeNet()
        output = model(torch.from_numpy(angle_features([-10.0, 0.0, 10.0])))
        self.assertEqual(tuple(output.shape), (3, 32, 4))

    def test_training_loss_decreases_and_model_round_trip(self) -> None:
        train_angles, train_codes, validation_angles, validation_codes = build_simulation_datasets(10.0)
        model = SimpleCodeNet()
        history = train_model(
            model,
            train_angles,
            train_codes,
            validation_angles,
            validation_codes,
            epochs=30,
            learning_rate=3e-3,
            batch_size=8,
            seed=8,
        )
        self.assertLess(history["train_loss"][-1], history["train_loss"][0])
        self.assertGreater(history["best_epoch"], 0)
        sample = torch.from_numpy(angle_features([0.0]))
        before = model(sample).detach().numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_model(model, path)
            restored = load_model(path, device="cpu")
            after = restored(sample).detach().numpy()
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
