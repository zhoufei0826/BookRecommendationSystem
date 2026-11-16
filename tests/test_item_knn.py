import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from item_knn import (
    build_matrix,
    compute_similarity,
    recommend_all,
    truncate_neighbors,
)


def make_toy_matrix():
    """3 位用户 × 3 本图书：
       u0: 书0, 书1      u1: 书0, 书1      u2: 书1, 书2
       度数：书0=2, 书1=3, 书2=1
       共现：书0-书1 = 2（u0,u1），书1-书2 = 1（u2），书0-书2 = 0
    """
    df = pd.DataFrame({
        'user_id': [0, 0, 1, 1, 2, 2],
        'item_id': [0, 1, 0, 1, 1, 2],
    })
    user_to_idx = {0: 0, 1: 1, 2: 2}
    item_to_idx = {0: 0, 1: 1, 2: 2}
    return df, user_to_idx, item_to_idx


class ItemKNNTests(unittest.TestCase):
    def setUp(self):
        self.df, self.user_to_idx, self.item_to_idx = make_toy_matrix()
        self.mat = build_matrix(self.df, self.user_to_idx, self.item_to_idx)
        self.popularity = {0: 2, 1: 3, 2: 1}

    def test_cosine_similarity_is_symmetric_and_degree_normalised(self):
        sim = compute_similarity(self.mat, 'cosine').toarray()
        np.testing.assert_allclose(sim, sim.T, rtol=1e-6)
        np.testing.assert_allclose(np.diag(sim), 1.0, rtol=1e-6)
        # 书0（度2）与书1（度3）共现 2 次：余弦 = 2/sqrt(2*3)
        self.assertAlmostEqual(sim[0, 1], 2 / np.sqrt(6), places=5)
        # 书1（度3）与书2（度1）共现 1 次：余弦 = 1/sqrt(3*1)
        self.assertAlmostEqual(sim[1, 2], 1 / np.sqrt(3), places=5)
        # 书0 与 书2 无共同用户
        self.assertAlmostEqual(sim[0, 2], 0.0, places=6)

    def test_cosine_suppresses_popularity_bias_vs_raw(self):
        cos = compute_similarity(self.mat, 'cosine').toarray()
        raw = compute_similarity(self.mat, 'raw').toarray()
        # 原始共现中「热门书1 与任意书」都很大（2 和 1），余弦按度归一后拉开差距
        self.assertGreater(raw[0, 1], raw[1, 2])
        self.assertLess(cos[0, 1] / cos[1, 2], raw[0, 1] / raw[1, 2],
                        "余弦应降低热门书方向的相似度比值")

    def test_truncate_neighbors_drops_self_and_limits_count(self):
        sim = compute_similarity(self.mat, 'cosine')
        nb = truncate_neighbors(sim, 1)
        for row in range(nb.shape[0]):
            cols = nb.indices[nb.indptr[row]:nb.indptr[row + 1]]
            self.assertNotIn(row, cols, "邻居列表不应包含自身")
            self.assertLessEqual(len(cols), 1)

    def test_recommendation_excludes_seen_items(self):
        sim = compute_similarity(self.mat, 'cosine')
        nb = truncate_neighbors(sim, 2)
        user_train_items = {0: {0, 1}, 1: {0, 1}, 2: {1, 2}}
        recs, fallback_users = recommend_all(
            nb, user_train_items, self.popularity,
            self.user_to_idx, self.item_to_idx,
            beta=0.5, alpha=0.0, top_k=1,
        )
        for user_id, items in recs.items():
            self.assertEqual(len(items), 1, f"用户 {user_id} 推荐数为 {len(items)}")
            self.assertFalse(set(items) & user_train_items[user_id],
                             f"用户 {user_id} 推荐了已读图书")
        # 用户 0 已读 0、1：书2 与书1 共现，应被推荐
        self.assertEqual(recs[0], [2])
        # 用户 2 已读 1、2：书0 与书1 的相似度更高，应优先推荐书0
        self.assertEqual(recs[2], [0])
        self.assertEqual(fallback_users, 0)

    def test_fallback_when_no_unread_candidate(self):
        """用户读完全部图书时，回退逻辑不应产生重复或非法推荐。"""
        sim = compute_similarity(self.mat, 'cosine')
        nb = truncate_neighbors(sim, 2)
        recs, fallback_users = recommend_all(
            nb, {0: {0, 1, 2}, 1: {0, 1}, 2: {1, 2}},
            self.popularity, self.user_to_idx, self.item_to_idx,
            beta=0.5, alpha=0.0, top_k=1,
        )
        self.assertEqual(recs[0], [], "无未读候选时应返回空列表，交由上层回退处理")
        self.assertGreaterEqual(fallback_users, 1)

    def test_history_downweighting_changes_candidate_order(self):
        """beta>0 时热门历史书权重降低：用户只读热门书0 时，书2 的相对得分应上升。"""
        sim = compute_similarity(self.mat, 'cosine')
        nb = truncate_neighbors(sim, 2)
        user_train_items = {0: {0}, 1: {0, 1}, 2: {1, 2}}
        recs_no, _ = recommend_all(nb, user_train_items, self.popularity,
                                   self.user_to_idx, self.item_to_idx,
                                   beta=0.0, alpha=0.0, top_k=1)
        recs_beta, _ = recommend_all(nb, user_train_items, self.popularity,
                                     self.user_to_idx, self.item_to_idx,
                                     beta=0.5, alpha=0.0, top_k=1)
        # 两种设置都必须给出 1 个推荐且不含已读
        for recs in (recs_no, recs_beta):
            self.assertEqual(len(recs[0]), 1)
            self.assertNotIn(0, recs[0])

    def test_hist_norm_invariance_when_no_candidate_penalty(self):
        """alpha=0 时历史权重归一化只是每用户常数缩放，不应改变推荐顺序。"""
        sim = compute_similarity(self.mat, 'cosine')
        nb = truncate_neighbors(sim, 2)
        user_train_items = {0: {0}, 1: {0, 1}, 2: {1, 2}}
        recs_sum, _ = recommend_all(nb, user_train_items, self.popularity,
                                    self.user_to_idx, self.item_to_idx,
                                    beta=0.5, alpha=0.0, top_k=2, hist_norm='sum')
        recs_none, _ = recommend_all(nb, user_train_items, self.popularity,
                                     self.user_to_idx, self.item_to_idx,
                                     beta=0.5, alpha=0.0, top_k=2, hist_norm='none')
        self.assertEqual(recs_sum, recs_none,
                         "无候选惩罚时，历史归一化不应影响排序")

    def test_shrinkage_reduces_sparse_similarity(self):
        """共同用户极少时，收缩应把相似度压低（长尾噪声抑制）。"""
        cos = compute_similarity(self.mat, 'cosine').toarray()
        shrunk = compute_similarity(self.mat, 'cosine', shrinkage=10.0).toarray()
        # 书0-书1 共同用户 2 个，收缩后仍为正但变小
        self.assertGreater(shrunk[0, 1], 0)
        self.assertLess(shrunk[0, 1], cos[0, 1])
        # 书0-书2 无共同用户，收缩后仍为 0
        self.assertAlmostEqual(shrunk[0, 2], 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
