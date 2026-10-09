"""Checks use synthetic input only; no training or TEST scoring."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eventedgetcn import decode_event_state
from run_example import (
    load_model, predict, make_synthetic_aggregate,
    NORMALIZATION_MEAN_W, NORMALIZATION_STD_W,
)


class ExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model, cls.decoder = load_model()
        cls.aggregate = make_synthetic_aggregate(cls.model.cfg.ws, 40)

    def infer(self, aggregate):
        return predict(aggregate, self.model, self.decoder,
                       NORMALIZATION_MEAN_W, NORMALIZATION_STD_W)

    def test_exclusive_past(self):
        endpoint = self.model.cfg.ws + 10
        original = self.infer(self.aggregate)
        changed = self.aggregate.copy()
        changed[endpoint:] += 5000
        updated = self.infer(changed)
        index = np.flatnonzero(original["endpoint"] == endpoint)[0]
        for key in ("probability", "raw_power_w"):
            np.testing.assert_array_equal(
                original[key][:index + 1], updated[key][:index + 1]
            )

    def test_inference_and_power_gating(self):
        result = self.infer(self.aggregate)
        self.assertEqual(len(result["endpoint"]), 40)
        self.assertTrue(np.isfinite(result["probability"]).all())
        self.assertTrue(((result["probability"] >= 0)
                         & (result["probability"] <= 1)).all())
        self.assertTrue((result["raw_power_w"] >= 0).all())
        np.testing.assert_array_equal(
            result["gated_power_w"],
            result["raw_power_w"] * result["decoded_state"],
        )

    def test_missing_history_is_skipped(self):
        aggregate = self.aggregate.copy()
        aggregate[self.model.cfg.ws] = np.nan
        result = self.infer(aggregate)
        np.testing.assert_array_equal(result["endpoint"], [self.model.cfg.ws])

    def test_decoder_does_not_merge_across_missing_time(self):
        state = decode_event_state(
            np.array([1., 0., 1.]), threshold=.5, min_on=1, min_off=5,
            merge_gap=5, min_cycle_len=1, max_cycle_len=10,
            time_index=np.array([240, 241, 250]),
        )
        np.testing.assert_array_equal(state, [1, 0, 1])


if __name__ == "__main__":
    unittest.main()

