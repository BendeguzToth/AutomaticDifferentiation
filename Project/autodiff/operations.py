"""
This file contains atomic mathematical operations
for Tensor objects.
"""

# Third-party dependencies
import numpy as np

# Project files
from autodiff.tensor import Tensor
from autodiff.devtools import unstable, placeholder, logging


@logging
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
@logging
def average(tensor, axis=None):
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


@placeholder
@logging
def max(tensor, axis=None):
    """
    This function returns the maximum elements along the specified axis.
    :param tensor: Tensor object.
    :param axis: The axis to take the max along.
    :return: New Tensor object.
    """



@placeholder
@logging
def concatenate(tensor_a, tensor_b, axis):
    """
    Concatenates two tensors along the specified axis.
    :param tensor_a: Tensor object.
    :param tensor_b: Tensor object.
    :param axis: The axis to concatenate along.
    :return: New Tensor object.
    """

@placeholder
@logging
def split(tensor, axis):
    """
    Splits the tensor.
    :param tensor:
    :param axis:
    :return:
    """


@unstable
@logging
def repeat(tensor, rep, axis):
    """
    This function tiles the vector along the given axis
    'rep' times.
    :param tensor: Tensor object.
    :param rep: The number of repetition.
    :param axis: Axis to tile along.
    :return: New Tensor object.
    """
    new_shape = np.ones((tensor.rank,), dtype=int)
    new_shape[axis] = rep
    result = Tensor(np.tile(tensor.value, new_shape))
    tensor.dependencies[result] = ({"axis":axis, "rep":rep}, lambda cache, from_above: np.sum(np.array(np.split(from_above, cache["rep"], cache["axis"])), axis=0))

    return result


@unstable
@logging
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
@logging
def reshape(tensor, newshape):
    """
    Reshapes the given Tensor to the specified shape.
    :param x: Tensor object.
    :param newshape: tuple of integers.
    :return: A Tensor object.
    """
    result = Tensor(np.reshape(tensor.value, newshape))

    tensor.dependencies[result] = ({"old_shape": tensor.shape}, lambda cache, from_above: np.reshape(from_above, cache["old_shape"]))
    return result


@unstable
@logging
def exp(tensor):
    """
    e ^ tensor
    :param tensor: Tensor.
    :return: New Tensor object.
    """
    result = Tensor(np.exp(tensor.value))
    tensor.dependencies[result] = ({"local": result.value.copy()}, lambda cache, from_above: from_above * cache["local"])

    return result


@unstable
@logging
def ln(tensor):
    """
    ln(tensor) - natural logging of the tensor.
    :param tensor: Tensor object.
    :return: New Tensor object.
    """
    result = Tensor(np.log(tensor.value))
    tensor.dependencies[result] = ({"local": 1 / tensor.value}, lambda cache, from_above: cache["local"] * from_above)

    return result
