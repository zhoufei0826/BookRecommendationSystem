import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ensemble import make_predictions, top_items
from item_knn import compute_similarity, recommend_all, truncate_neighbors


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        matrix = csr_matrix(np.array([[1, 1, 0, 0, 0], [1, 0, 1, 0, 1],
                                      [0, 1, 1, 1, 0]], dtype=np.float32))
        self.model = SimpleNamespace(
            matrix=matrix, user_ids=np.array([10, 20, 40]),
            item_ids=np.array([101, 103, 211, 320, 890]),
            popularity=np.asarray(matrix.sum(axis=0)).ravel(),
            user_factors=np.array([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32),
            item_factors=np.array([[1, 0], [0, 1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6]], dtype=np.float32))
        self.neighbors = truncate_neighbors(compute_similarity(matrix), 4)

    def test_zero_als_weight_matches_b_with_nonconsecutive_ids(self):
        m = self.model
        history = {int(uid): set(m.item_ids[m.matrix[row].indices].tolist())
                   for row, uid in enumerate(m.user_ids)}
        expected, _ = recommend_all(self.neighbors, history, dict(zip(m.item_ids, m.popularity)),
                                     dict(zip(m.user_ids, range(3))), dict(zip(m.item_ids, range(5))), top_k=2)
        got = make_predictions(m, self.neighbors, m.user_ids, {'knn': (0, 0)}, top_k=2)['knn']
        for uid, row in zip(m.user_ids, got):
            self.assertEqual(row.tolist(), expected[int(uid)])

    def test_blend_and_popular_fallback_filter_seen_items(self):
        for neighbors in [self.neighbors, csr_matrix((5, 5))]:
            predictions = make_predictions(self.model, neighbors, self.model.user_ids,
                                           {'knn': (0, 0), 'blend': (0.25, 0)}, top_k=2)
            for items in predictions.values():
                for row, recs in enumerate(items):
                    seen = self.model.item_ids[self.model.matrix[row].indices]
                    self.assertEqual(len(set(recs)), 2)
                    self.assertFalse(set(recs) & set(seen))
                    self.assertTrue(set(recs) <= set(self.model.item_ids))

    def test_insufficient_unseen_catalog_is_rejected(self):
        with self.assertRaises(ValueError):
            top_items(np.array([1.0, 2.0]), np.array([0]), np.array([2, 1]), 2)


if __name__ == '__main__':
    unittest.main()
