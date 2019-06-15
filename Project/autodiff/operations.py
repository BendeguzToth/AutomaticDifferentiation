"""
This file contains basic mathematical operations
for Tensor objects.
"""

# Third-party dependencies
import numpy as np

# Project files
from autodiff.tensor import Tensor
from autodiff.devtools import unstable, placeholder, log


@log
def sum(tensor, axis=None):
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
@log
def mean(tensor, axis=None):
    """
    This function calculates the average of a given Tensor
    along the specified axes.
    Keeps dimensions.
    :param tensor: The Tensor object.
    :param axis: int or tuple of ints.
    :return: A Tensor object.
    """
    result = Tensor(np.mean(tensor.value, axis=axis, keepdims=True))
    if axis is not None:
        fac = np.prod(np.array(tensor.shape)[np.array(list(axis))])
    else:
        fac = np.prod(np.array(tensor.shape))
    tensor.dependencies[result] = (
    {"shape": tensor.shape, "fac": fac}, lambda cache, from_above: np.broadcast_to(from_above, shape=cache["shape"]) / fac)

    return result


@unstable
@log
def matmul(tensor_a, tensor_b):
    """
    Last 2 dimensions have to be valid for dot product,
    all axis before that needs to match.
    :param tensor_a:
    :param tensor_b:
    """
    result = Tensor(np.matmul(tensor_a.value, tensor_b.value))
    tensor_a.dependencies[result] = ({"local": np.swapaxes(tensor_b.value.copy(), -1, -2)}, lambda cache, from_above: np.matmul(from_above, cache["local"]))
    tensor_b.dependencies[result] = ({"local": np.swapaxes(tensor_a.value.copy(), -1, -2)}, lambda cache, from_above: np.matmul(cache["local"], from_above))

    return result


@unstable
@log
def tile_leading_dims(tensor, leading_dims):
    """
    This function will broadcast up the specified tensor by
    tiling leading_dims dimensions in the first axes.
    example:
    tensor = [
                [1, 2, 3],
                [4, 5, 6]
            ]

            tensor.shape = (2, 3)
            ops.tile_leading_dims(tensor, leading_dims=(2))
            =
            [
            [
                [1, 2, 3],
                [4, 5, 6]
            ],

            [
                [1, 2, 3],
                [4, 5, 6]
            ]
            ]

            tensor.shape = (2, 2, 3)
    :param tensor: Tensor object to be broadcasted.
    :param leading_dims: int or tuple of ints, specifying
    the number of entries at the leading dimensions.
    :return New Tensor object.

    """
    result = Tensor(np.broadcast_to(tensor.value, (leading_dims,)+tensor.shape))
    tensor.dependencies[result] = ({"sum_axis": tuple(range(len((leading_dims,))))}, lambda cache, from_above: np.sum(from_above, axis=cache["sum_axis"]))

    return result


@unstable
@log
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
def exp(x):
    """
    e ^ Tensor
    :param x: Tensor.
    :return:
    """
