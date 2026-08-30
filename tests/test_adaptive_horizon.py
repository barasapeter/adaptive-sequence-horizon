import unittest

from adaptive_horizon import (
    AdaptiveHorizon,
    ending_streak,
    parse_binary_sequence,
    state_features,
    walk_forward_backtest,
)


class AdaptiveHorizonTests(unittest.TestCase):
    def test_parse_native_and_legacy_sequences(self):
        self.assertEqual(parse_binary_sequence("0, 1, 0"), [0, 1, 0])
        self.assertEqual(parse_binary_sequence("L P L"), [0, 1, 0])

    def test_full_terminal_streak_is_not_clipped_to_window(self):
        history = [0] * 10_000
        features = state_features(history, 5)
        self.assertEqual(features.ending_bit, 0)
        self.assertEqual(features.ending_streak_length, 10_000)
        self.assertEqual(features.longest_streak, 5)

    def test_streak_changes_cleanly_after_reversal(self):
        self.assertEqual(ending_streak([0] * 1_000 + [1, 1]), (1, 2))

    def test_constant_zero_sequence_is_safe_and_learns_zero(self):
        rows = walk_forward_backtest(
            [0] * 500,
            n_min=2,
            n_max=8,
            horizon=4,
            warmup=40,
            performance_memory=50,
            min_resolved_per_n=2,
            prior_strength=2.0,
            min_lift=-1.0,
            ensemble_size=3,
        )
        forecasts = [row for row in rows if not row.abstained]
        self.assertTrue(forecasts)
        self.assertTrue(all(row.target == 0 for row in forecasts[-25:]))
        self.assertTrue(all(row.hit == 1 for row in forecasts[-25:]))

    def test_constant_one_sequence_is_safe_and_learns_one(self):
        rows = walk_forward_backtest(
            [1] * 500,
            n_min=2,
            n_max=8,
            horizon=4,
            warmup=40,
            performance_memory=50,
            min_resolved_per_n=2,
            prior_strength=2.0,
            min_lift=-1.0,
            ensemble_size=3,
        )
        forecasts = [row for row in rows if not row.abstained]
        self.assertTrue(forecasts)
        self.assertTrue(all(row.target == 1 for row in forecasts[-25:]))
        self.assertTrue(all(row.hit == 1 for row in forecasts[-25:]))

    def test_default_model_can_abstain_without_evidence(self):
        model = AdaptiveHorizon(n_min=2, n_max=5)
        forecast = model.forecast([0, 1, 0, 1, 0, 1])
        self.assertTrue(forecast.abstained)
        self.assertIsNone(forecast.target)

    def test_invalid_non_binary_values_are_rejected(self):
        with self.assertRaises(ValueError):
            walk_forward_backtest([0, 1, 2, 0], n_min=1, n_max=1, horizon=1)


if __name__ == "__main__":
    unittest.main()
