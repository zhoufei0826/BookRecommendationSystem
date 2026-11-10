import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bpr_model import BPRMF
from evaluate import evaluate, validate_recommendations


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.targets = {1: {10, 11}, 2: {20}}

    def test_metrics_for_partial_hits(self):
        result = evaluate(
            {1: [10, 30], 2: [32, 20]},
            self.targets,
            top_k=2,
        )
        self.assertAlmostEqual(result["micro_precision"], 0.5)
        self.assertAlmostEqual(result["micro_recall"], 2 / 3)
        self.assertAlmostEqual(result["micro_f1"], 4 / 7)
        self.assertAlmostEqual(result["macro_f1"], (0.5 + 2 / 3) / 2)

    def test_rejects_missing_duplicate_and_seen_recommendations(self):
        with self.assertRaisesRegex(ValueError, "缺少用户"):
            evaluate({1: [10, 11]}, self.targets, top_k=2)
        with self.assertRaisesRegex(ValueError, "重复推荐"):
            evaluate({1: [10, 10], 2: [20, 21]}, self.targets, top_k=2)
        with self.assertRaisesRegex(ValueError, "已交互"):
            validate_recommendations(
                {1: [10, 30], 2: [20, 21]},
                self.targets,
                top_k=2,
                user_train_items={1: {10}},
            )

    def test_rejects_wrong_recommendation_count(self):
        with self.assertRaisesRegex(ValueError, "推荐数"):
            evaluate({1: [10], 2: [20, 21]}, self.targets, top_k=2)


class BPRTests(unittest.TestCase):
    def test_training_improves_positive_pair_and_filters_seen_items(self):
        interactions = pd.DataFrame(
            {"user_id": [1, 2], "item_id": [10, 11]}
        )
        model = BPRMF(
            n_factors=8,
            n_epochs=80,
            batch_size=2,
            learning_rate=0.05,
            reg_lambda=0.001,
            random_seed=42,
        )
        model.fit(interactions)

        user_idx = model.user_to_idx[1]
        self.assertGreater(
            model.P[user_idx] @ model.Q[model.item_to_idx[10]],
            model.P[user_idx] @ model.Q[model.item_to_idx[11]],
        )
        recommendations = model.recommend_all(
            [1, 2],
            {1: {10}, 2: {11}},
            top_k=1,
        )
        self.assertEqual(recommendations, {1: [11], 2: [10]})


if __name__ == "__main__":
    unittest.main()
