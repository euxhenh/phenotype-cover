import unittest

import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_allclose
from phenotype_cover._operations import pairwise_differences, group_by


class TestSelectors(unittest.TestCase):
    def test1(self):
        a = np.array([
            [1, 2, 6, 1],
            [3, 6, 7, 1],
            [3, 4, 1, 6]
        ])
        y = [0, 1, 0]
        M = group_by(a, y)
        gt = np.array([
            [2, 3, 3.5, 3.5],
            [3, 6, 7, 1]
        ])

        assert_allclose(M, gt)

    def test2(self):
        a = np.array([
            [1, 2, 6, 1],
            [3, 6, 7, 1],
            [3, 4, 1, 6]
        ])
        y = [0, 1, 0]
        M = group_by(a, y, operation=lambda x: x.sum(axis=0))
        gt = np.array([
            [4, 6, 7, 7],
            [3, 6, 7, 1]
        ])

        assert_allclose(M, gt)

    def test3(self):
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        a = csr_matrix((data, (row, col)), shape=(3, 3))
        y = [0, 1, 0]

        M = group_by(a, y, operation=lambda x: x.sum(axis=0))
        gt = np.array([
            [5, 5, 8],
            [0, 0, 3]
        ])

        assert_allclose(M, gt)

    def test4(self):
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])
        a = csr_matrix((data, (row, col)), shape=(3, 3))
        y = [0, 1, 0]

        M = group_by(a, y,
                     category_orders=[1, 0], operation=lambda x: x.sum(axis=0))
        gt = np.array([
            [0, 0, 3],
            [5, 5, 8]
        ])

        assert_allclose(M, gt)

    def test5(self):
        X = np.array([
            [0, 1, 2, 4, 1, 4],
            [2, 5, 1, 3, 4, 1],
            [3, 4, 1, 2, 1, 1],
            [4, 6, 7, 1, 3, 5],
            [6, 1, 3, 4, 5, 1]
        ], dtype=float)
        labels = np.array([0, 0, 1, 1, 2], dtype=int)
        pairmat, mapping = pairwise_differences(X, labels)

        assert_allclose(pairmat, np.array([
            [2.5, 2, 2.5, 2, 0.5, 0.5],  # 0 - 1
            [5, 2, 1.5, 0.5, 2.5, 1.5],  # 0 - 2
            [2.5, 4, 1, 2.5, 3, 2]  # 1 - 2
        ]))

    def test6(self):
        X = np.array([
            [0, 1, 2, 4, 1, 4],
            [2, 5, 1, 3, 4, 1],
            [3, 4, 1, 2, 1, 1],
            [4, 6, 7, 1, 3, 5],
            [6, 1, 3, 4, 5, 1]
        ], dtype=float)
        labels = np.array([0, 0, 1, 1, 2], dtype=int)
        pairmat, mapping = pairwise_differences(X, labels, ordered=True)

        assert_allclose(pairmat, np.array([
            [0, 0, 0, 2, 0.5, 0],  # 0 - 1
            [0, 2, 0, 0, 0, 1.5],  # 0 - 2
            [0, 4, 1, 0, 0, 2],  # 1 - 2
            [2.5, 2, 2.5, 0, 0, 0.5],  # 1 - 0
            [5, 0, 1.5, 0.5, 2.5, 0],  # 2 - 0
            [2.5, 0, 0, 2.5, 3, 0]  # 2 - 1
        ]))
