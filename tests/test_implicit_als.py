"""Small numerical and recommendation-contract checks for member C's ALS."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from implicit_als import ImplicitALS, _cg_update


class ImplicitALSTests(unittest.TestCase):
    def test_cg_update_matches_small_exact_als_solution(self):
        fixed = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        moving = np.zeros((1, 2), dtype=np.float32)
        gram = fixed.T @ fixed
        _cg_update(np.array([0, 2]), np.array([0, 1]), fixed,
                   moving, gram, 2.0, 0.1, 2)
        expected = np.linalg.solve(
            gram + 2.0 * (fixed[:2].T @ fixed[:2]) + 0.1 * np.eye(2),
            3.0 * fixed[:2].sum(axis=0),
        )
        np.testing.assert_allclose(moving[0], expected, rtol=1e-5, atol=1e-5)

    def test_recommendations_have_top_k_unseen_unique_items(self):
        pairs = [(u, (u + offset) % 12) for u in range(4) for offset in range(3)]
        train = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        model = ImplicitALS(factors=4, iterations=2, alpha=10, seed=42).fit(train)
        results = model.recommend(range(4), weights=(0.0, 0.25), top_k=3)
        for recs in results.values():
            for uid, items in recs.items():
                self.assertEqual(len(items), 3)
                self.assertEqual(len(set(items)), 3)
                self.assertFalse(set(items) & set(train.loc[train.user_id == uid, 'item_id']))


if __name__ == '__main__':
    unittest.main()
