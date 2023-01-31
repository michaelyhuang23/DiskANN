import unittest

import numpy as np

import diskannpy as dap

from sklearn.neighbors import NearestNeighbors

from fixtures import random_vectors, vectors_as_temp_file


class TestFloatIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._query_vectors = random_vectors(1000, 10, dtype=np.single)
        cls._index_vectors = random_vectors(10000, 10, dtype=np.single)

    def test_recall(self):
        neighbors = NearestNeighbors(n_neighbors=100, algorithm='auto', metric="l2")
        neighbors.fit(self._index_vectors)
        distances, indices = neighbors.kneighbors(self._query_vectors)

        index = dap.DiskANNFloatIndex(dap.L2)

        self.assertEqual(True, False)  # add assertion here


class TestUInt8Index(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


class TestInt8Index(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here
