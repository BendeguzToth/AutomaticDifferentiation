"""
This file contains basic mathematical operations
for Tensor objects.
"""

# Third-party dependencies
import numpy as np

# Project files
from autodiff.tensor import Tensor
from autodiff.devtools import unstable, placeholder


def sum(tensor, axis):
    """
    Sums the array over the specified axis.
    Dimension stays the same.
    :param tensor: Tensor object.
    :param axis: int or tuple of ints, axis to
    be summed over.
    :return: New Tensor object.
    """
    result = Tensor(np.sum(tensor.value, axis=axis, keepdims=True))
    tensor.dependencies[result] = ({"shape": tensor.shape}, lambda cache, from_above: np.broadcast_to(from_above, shape=cache["shape"]))

    return result


@unstable
def batch_matmul(x, y):
    """
    Batch matmul.
    """
    result = Tensor(np.matmul(x.value, y.value))

    # print(np.swapaxes(x.value.copy(), -1, -2).shape)

    x.dependencies[result] = ({"local": np.swapaxes(y.value.copy(), -1, -2)}, lambda cache, from_above: np.sum(np.matmul(from_above, cache["local"]), axis=0))
    # y.dependencies[result] = ({"local": np.swapaxes(x.value.copy(), -1, -2)}, lambda cache, from_above: np.sum(np.matmul(from_above, cache["local"]), axis=0))

    return result


@placeholder
def average(x, axis=None):
    """
    This function calculates the average of a given Tensor
    along the specified axes.
    :param x: The Tensor object.
    :param axis: int or tuple of ints.
    :return: A Tensor object.
    """


@unstable
def reshape(x, newshape):
    """
    Reshapes the given Tensor to the specified shape.
    :param x: Tensor object.
    :param newshape: tuple of integers.
    :return: A Tensor object.
    """
    result = Tensor(np.reshape(x.value, newshape))

    x.dependencies[result] = ({"old_shape": x.shape}, lambda cache, from_above: np.reshape(from_above, cache["old_shape"]))
    return result


@placeholder
def upcast(x, *args):
    """
    Casting a 2D array to 3/4D by tiling along first dimensions.
    :param x: Tensor.
    :param args: how to upcast
    :return:
    """


@placeholder
def collapse(x, *args):
    """
    Inverse of upcast.
    :param x: Tensor
    :param args: how to collapse
    :return:
    """


@placeholder
def exp(x):
    """
    e ^ Tensor
    :param x: Tensor.
    :return:
    """


@placeholder
def squeeze(tensor):
    """
    Removes extra dimensions.
    :param tensor: Tensor object.
    :return:
    """