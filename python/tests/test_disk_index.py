import ctypes
import os
import unittest

import numpy as np

import diskannpy as dap

from tempfile import TemporaryDirectory

from sklearn.neighbors import NearestNeighbors

from fixtures import random_vectors, vectors_as_temp_file


class TestFloatIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # np.ctypeslib.load_library("libmkl_intel_thread", "/lib/x86_64-linux-gnu/")
        # np.ctypeslib.load_library("libmkl_intel_ilp64", "/lib/x86_64-linux-gnu/")
        # np.ctypeslib.load_library("libmkl_core", "/lib/x86_64-linux-gnu/")
        # np.ctypeslib.load_library("libiomp5", "/lib/x86_64-linux-gnu/")
        # np.ctypeslib.load_library("libmkl_avx2", "/lib/x86_64-linux-gnu/")
        # np.ctypeslib.load_library("libmkl_def", "/lib/x86_64-linux-gnu/")
        ctypes.CDLL(ctypes.util.find_library("mkl_intel_thread"))
        ctypes.CDLL(ctypes.util.find_library("iomp5"))
        ctypes.CDLL(ctypes.util.find_library("mkl_intel_ilp64"))
        ctypes.CDLL(ctypes.util.find_library("mkl_core"))
        ctypes.CDLL(ctypes.util.find_library("mkl_def"))
        ctypes.CDLL(ctypes.util.find_library("mkl_avx2"))

        cls._query_vectors = random_vectors(1000, 10, dtype=np.single)
        cls._index_vectors = random_vectors(10000, 10, dtype=np.single)

    def test_recall(self):
        neighbors = NearestNeighbors(n_neighbors=100, algorithm='auto', metric="l2")
        neighbors.fit(self._index_vectors)
        distances, indices = neighbors.kneighbors(self._query_vectors)

        index = dap.DiskANNFloatIndex(dap.L2)
        with vectors_as_temp_file(self._index_vectors) as vector_temp:
            with TemporaryDirectory() as temp_dir:

                index.build(
                    data_file_path=vector_temp,
                    index_prefix_path=os.path.join(temp_dir, "ann"),
                    R=16,
                    L=32,
                    final_index_ram_limit=0.00003,
                    indexing_ram_limit=1,
                    num_threads=1,
                    pq_disk_bytes=0
                )
            # -R 16 -L 32 -B 0.00003 -M 1 --build_PQ_bytes 5 from line 93 pr-test.yml

            self.assertEqual(True, False)  # add assertion here


class TestUInt8Index(unittest.TestCase):
    def test_something(self):
        pass
        # self.assertEqual(True, False)  # add assertion here


class TestInt8Index(unittest.TestCase):
    def test_something(self):
        pass
        # self.assertEqual(True, False)  # add assertion here
