"""
This file contains unit test for operations.py
"""

# Standard libraries
from unittest import TestCase

# Third-party libraries
import numpy as np

# Project files
from autodiff.tensor import Tensor
import autodiff.tensor as tn
import autodiff.operations as ops


class TestOps(TestCase):
    def test_sum2d(self):
        a = Tensor(np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]
        ]))
        axs0 = ops.sum(a, axis=0)
        axs1 = ops.sum(a, axis=1)

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
        a.reset_done()
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

        axs0 = ops.sum(a, axis=0)
        axs1 = ops.sum(a, axis=1)
        axs2 = ops.sum(a, axis=2)

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
        a.reset_done()

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
        a.reset_done()

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

    def test_sumAxes(self):
        a = Tensor(np.array([
            [
                [5, 9],
                [2, 7],
                [4, 1]
            ],

            [
                [6, 0],
                [3, 11],
                [23, 8]
            ]
        ]))

        axs01 = ops.sum(a, axis=(0, 1))
        axs02 = ops.sum(a, axis=(0, 2))
        axs12 = ops.sum(a, axis=(1, 2))

        self.assertTrue(np.array_equal(axs01.value, np.array([
            [[43, 36]]
        ])))

        self.assertTrue(np.array_equal(axs02.value, np.array([
            [
                [20],
                [23],
                [36]
            ]
        ])))

        self.assertTrue(np.array_equal(axs12.value, np.array([
            [[28]],

            [[51]]
        ])))

        axs01.grad = np.array([
            [[7, 2]]
        ])
        a.backprop()
        self.assertTrue(np.array_equal(a.grad, np.array([
            [
                [7, 2],
                [7, 2],
                [7, 2]
            ],

            [
                [7, 2],
                [7, 2],
                [7, 2]
            ]
        ])))

        a.reset_grad()
        a.remove_dependency(axs01)
        a.reset_done()

        axs02.grad = np.array([
            [
                [1],
                [2],
                [3]
            ]
        ])
        a.backprop()
        self.assertTrue(np.array_equal(a.grad, np.array([
            [
                [1, 1],
                [2, 2],
                [3, 3]
            ],

            [
                [1, 1],
                [2, 2],
                [3, 3]
            ]
        ])))

        a.reset_grad()
        a.remove_dependency(axs02)
        a.reset_done()

        axs12.grad = np.array([
            [[2]],

            [[42]]
        ])
        a.backprop()
        self.assertTrue(np.array_equal(a.grad, np.array([
            [
                [2, 2],
                [2, 2],
                [2, 2]
            ],

            [
                [42, 42],
                [42, 42],
                [42, 42]
            ]
        ])))
        a.reset_all()
        axs012 = ops.sum(a, axis=(0, 1, 2))
        self.assertTrue(np.array_equal(axs012.value, np.array([[[79]]])))
        axs012.grad = np.array([[[42]]])
        a.backprop()
        self.assertTrue(np.array_equal(a.grad, np.array([
            [
                [42, 42],
                [42, 42],
                [42, 42]
            ],

            [
                [42, 42],
                [42, 42],
                [42, 42]
            ]
        ])))

        a.reset_all()
        axsNone = ops.sum(a)
        self.assertTrue(np.array_equal(axsNone.value, np.array([[[79]]])))
        axsNone.grad = np.array([[[42]]])
        a.backprop()
        self.assertTrue(np.array_equal(a.grad, np.array([
            [
                [42, 42],
                [42, 42],
                [42, 42]
            ],

            [
                [42, 42],
                [42, 42],
                [42, 42]
            ]
        ])))

    def test_concat2d(self):
        a = tn.tensor(
            [
                [1, 4, 5],
                [8, 9, 6]
            ]
        )

        b = tn.tensor(
            [
                [9, 7, 2],
                [4, 2, 1],
                [8, 6, 9]
            ]
        )

        c = ops.concatenate([a, b], axis=0)

        self.assertTrue(np.array_equal(c.value, np.array([
            [1, 4, 5],
            [8, 9, 6],
            [9, 7, 2],
            [4, 2, 1],
            [8, 6, 9]
        ])))

        c.grad = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]

        ])

        a.backprop()
        b.backprop()

        self.assertTrue(np.array_equal(a.grad, np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])))

        self.assertTrue(np.array_equal(b.grad, np.array([
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])))

    def test_concat_nd(self):
        a = tn.ones((4, 2, 12))
        b = tn.ones((4, 5, 12))
        c = tn.ones((4, 7, 12))

        res = ops.concatenate([a, b, c], axis=1)

        self.assertEqual(res.shape, (4, 14, 12))

        res.grad = np.random.randn(4, 14, 12)
        target_grads = np.split(res.grad, (2, 7, 14), axis=1)

        a.backprop()
        b.backprop()
        c.backprop()

        self.assertTrue(np.array_equal(a.grad, target_grads[0]))
        self.assertTrue(np.array_equal(b.grad, target_grads[1]))
        self.assertTrue(np.array_equal(c.grad, target_grads[2]))

    def test_transpose(self):
        a = tn.tensor([
            [1, 2, 3],
            [4, 5, 6]
        ])

        b = ops.transpose(a, permutation=(1, 0))

        self.assertTrue(np.array_equal(b.value, np.array([
            [1, 4],
            [2, 5],
            [3, 6]
        ])))

        a2 = tn.tensor([
            [
                [6, 8, 9],
                [3, 4, 7]
            ],

            [
                [2, 5, 8],
                [0, 1, 6]
            ]
        ])

        b2 = ops.transpose(a2, permutation=(1, 2, 0))

        self.assertTrue(np.array_equal(b2.value, np.array([
            [
                [6, 2],
                [8, 5],
                [9, 8]
            ],

            [
                [3, 0],
                [4, 1],
                [7, 6]
            ]
        ])))

        b2.grad = b2.value
        a2.backprop()

        self.assertTrue(np.array_equal(a2.value, a2.grad))

        a3 = tn.ones((8, 12, 3, 5, 7))
        b3 = ops.transpose(a3, permutation=(4, 0, 1, 3, 2))
        b3.grad = b3.value
        a3.backprop()

        self.assertTrue(np.array_equal(a3.value, a3.grad))

    def test_average(self):
        a = tn.tensor([
            [
                [3, 4, 5],
                [6, 8, 9]
            ],

            [
                [1, 2, 1],
                [2, 2, 1]
            ]
        ])

        ax0 = ops.average(a, axis=0)
        self.assertTrue(np.array_equal(ax0.value, np.array([
            [
                [2., 3., 3.],
                [4., 5., 5.]
            ]
        ])))

        ax0.grad = np.array([
            [
                [2., 6., 8.],
                [6., 12., 50.]
            ]
        ])

        a.backprop()
        self.assertTrue(np.array_equal(a.grad, np.array([
            [
                [1., 3., 4.],
                [3., 6., 25]
            ],

            [
                [1., 3., 4.],
                [3., 6., 25]
            ]
        ])))
        a.reset_done()
        a.reset_grad()
        a.remove_dependency(ax0)

        ax1 = ops.average(a, axis=1)
        self.assertTrue(np.array_equal(ax1.value, np.array([
            [[4.5, 6., 7.]],

            [[1.5, 2., 1.]]
        ])))

        ax1.grad = np.array([
            [[6., 8., 2.]],

            [[2., 12., 4.]]
        ])
        a.backprop()
        self.assertTrue(np.array_equal(a.grad, np.array([
            [
                [3., 4., 1.],
                [3., 4., 1.]
            ],


            [
                [1., 6., 2.],
                [1., 6., 2.]
            ]
        ])))
        a.reset_done()
        a.reset_grad()
        a.remove_dependency(ax1)

        ax01 = ops.average(a, axis=(0, 1))
        self.assertTrue(np.array_equal(ax01.value, np.array([
            [[3., 4., 4.]]
        ])))

        ax01.grad = np.array([
            [[4., 8., 16.]]
        ])

        a.backprop()
        self.assertTrue(np.array_equal(a.grad, np.array([
            [
                [1., 2., 4.],
                [1., 2., 4.],
            ],

            [
                [1., 2., 4.],
                [1., 2., 4.],
            ]
        ])))
