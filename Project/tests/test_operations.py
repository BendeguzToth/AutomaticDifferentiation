"""
This file contains unit test for operations.py
"""

# Standard libraries
from unittest import TestCase

# Third-party libraries
import numpy as np

# Project files
from autodiff.tensor import Tensor, derive
import autodiff.operations as ops


class TestOps(TestCase):
    def test_sum(self):
        a = Tensor(np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ]))
        axs0 = ops.single_sum(a, axis=0)
        axs1 = ops.single_sum(a, axis=1)

        self.assertTrue(np.array_equal(axs0.value, np.array([[6, 9, 12]])))
        self.assertTrue(np.array_equal(axs1.value, np.array([[6], [9], [12]])))