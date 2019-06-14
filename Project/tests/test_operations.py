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
    def test_sum2d(self):
        a = Tensor(np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ]))
        axs0 = ops.single_sum(a, axis=0)
        axs1 = ops.single_sum(a, axis=1)

        self.assertTrue(np.array_equal(axs0.value, np.array([[6, 9, 12]])))
        self.assertTrue(np.array_equal(axs1.value, np.array([[6], [9], [12]])))

        axs0.grad = np.array([[2, 5, 20]])
        a.backprop()

        self.assertTrue(np.array_equal(a.grad, np.array([
            [2, 5, 20],
            [2, 5, 20],
            [2, 5, 20]
        ])))


        a.reset_grad()
        a.set_undone()
        a.remove_dependency(axs0)

        axs1.grad = np.array([[1], [4], [2]])
        a.backprop()

        self.assertTrue(np.array_equal(a.grad, np.array([
            [1, 1, 1],
            [4, 4, 4],
            [2, 2, 2]
        ])))

    def test_sum3d(self):
        a = Tensor(np.array([
            [
                [2, 7],
                [9, 17],
                [21, 5]],

            [
                [3, 8],
                [16, 39],
                [102, 4]]
        ]))

        self.assertEqual(a.shape, (2, 3, 2))

        axs0 = ops.single_sum(a, axis=0)
        axs1 = ops.single_sum(a, axis=1)
        axs2 = ops.single_sum(a, axis=2)

        self.assertTrue(np.array_equal(axs0.value, np.array([
            [[5, 15],
            [25, 56],
            [123, 9]]
        ])))

        self.assertTrue(np.array_equal(axs1.value, np.array([
            [[32, 29]],

            [[121, 51]]
        ])))

        self.assertTrue(np.array_equal(axs2.value, np.array([
            [[9],
             [26],
             [26]],

            [[11],
             [55],
             [106]]
        ])))

        axs0.grad = np.array([
            [
                [2, 4],
                [16, 7],
                [9, 1]
            ]
        ])

        a.backprop()

        self.assertTrue(np.array_equal(a.grad, np.array([
            [
                [2, 4],
                [16, 7],
                [9, 1]
            ],

            [
                [2, 4],
                [16, 7],
                [9, 1]
            ]
        ])))

        a.reset_grad()
        a.remove_dependency(axs0)
        a.set_undone()

        axs1.grad = np.array([
            [[12, 5]],

            [[7, 20]]
        ])

        a.backprop()

        self.assertTrue(np.array_equal(a.grad, np.array([
            [
                [12, 5],
                [12, 5],
                [12, 5]
            ],

            [
                [7, 20],
                [7, 20],
                [7, 20]
            ]
        ])))

        a.reset_grad()
        a.remove_dependency(axs1)
        a.set_undone()

        axs2.grad = np.array([
            [[2],
             [8],
             [1]],

            [[0],
             [7],
             [13]]
        ])

        a.backprop()

        self.assertTrue(np.array_equal(a.grad, np.array([
            [
                [2, 2],
                [8, 8],
                [1, 1]
            ],

            [
                [0, 0],
                [7, 7],
                [13, 13]
            ]
        ])))
