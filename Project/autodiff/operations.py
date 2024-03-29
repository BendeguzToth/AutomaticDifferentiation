"""
This file contains atomic mathematical operations
for Tensor objects.
"""

# Third-party dependencies
import numpy as np

# Project files
from autodiff.tensor import Tensor
import autodiff.constants as const


def sum(tensor, axis=None):
    """
    Sums the array over the specified axis.
    Keeps dimensions.
    :param tensor: Tensor object.
    :param axis: int or tuple of ints, axis to
    be summed over.
    :return: New Tensor object.
    """
    result = Tensor(np.sum(tensor.value, axis=axis, keepdims=True))
    tensor.dependencies[result] = ({"shape": tensor.shape}, lambda cache, from_above: np.broadcast_to(from_above, shape=cache["shape"]))

    return result


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
        if type(axis) is int:
            axis = (axis,)
        fac = np.prod(np.array(tensor.shape)[np.array(list(axis))])
    else:
        fac = np.prod(np.array(tensor.shape))
    tensor.dependencies[result] = (
    {"shape": tensor.shape, "fac": fac}, lambda cache, from_above: np.broadcast_to(from_above, shape=cache["shape"]) / fac)

    return result

def variance(tensor, axis=-1):
    """
    This function calculates the variance of the Tensor along the specified
    axis.
    :param tensor: Tensor object.
    :param axis: int.
    :return: New Tensor object.
    """
    avg = np.mean(tensor.value, axis=axis, keepdims=True)
    x_min_mean = tensor.value - np.broadcast_to(avg, tensor.shape)

    def backward(cache, from_above):
        unmean = np.full(shape=cache["input_shape"],
                         fill_value=1 / cache["input_shape"][cache["axis"]]) * np.broadcast_to(from_above, shape=cache[
            "input_shape"])
        unpow = 2 * cache["x-mean"] * unmean
        unmean2 = np.broadcast_to(np.sum(-unpow, axis=cache["axis"], keepdims=True), unpow.shape) * np.full(unpow.shape,
                                                                                                            1 /
                                                                                                            unpow.shape[
                                                                                                                cache[
                                                                                                                    "axis"]])
        ret = unmean2 + unpow
        return ret

    result = Tensor(np.var(tensor.value, axis=axis, keepdims=True))
    tensor.dependencies.append((result, ({"input_shape": tensor.shape, "axis": axis, "x-mean": x_min_mean}, backward)))
    return result


def maximum(tensor, b):
    """
    Returns element-wise maximum of either two arrays or
    an array and a numeric. First argument is preferred when
    equal.
    :param tensor: Tensor object.
    :param b: Tensor or Union[int, float]
    :return: Tensor object.
    """
    if type(b) is Tensor:
        result = Tensor(np.where(tensor.value >= b.value, tensor.value, b.value))
        tensor.dependencies[result] = ({"local": np.where(tensor.value >= b.value, 1, 0)}, lambda cache, from_above: cache["local"] * from_above)
        b.dependencies[result] = ({"local": np.where(tensor.value >= b.value, 0, 1)}, lambda cache, from_above: cache["local"] * from_above)

        return result
    elif type(b) is int or type(b) is float:
        result = Tensor(np.where(tensor.value >= b, tensor.value, b))
        tensor.dependencies[result] = ({"local": np.where(tensor.value >= b, 1, 0)}, lambda cache, from_above: cache["local"] * from_above)

        return result


def transpose(tensor, permutation):
    """
    This function transposes the tensor to have the
    specified permutation of axes.
    :param tensor: Tensor object.
    :param permutation: Int tuple with same length as rank of tensor.
    The numbers in the tuple are the axes indexes, whose order
    determines the new permutation. If we have a Tensor with shape
    (10, 8, 12, 3), and permutation=(0, 2, 3, 1), it will have shape
    (10, 12, 3, 8).
    :return: New Tensor object.
    """
    result = Tensor(np.transpose(tensor.value, axes=permutation))
    back_perm = np.zeros((tensor.rank,), dtype=int)
    back_perm[np.array(list(permutation))] = np.arange(tensor.rank)
    tensor.dependencies[result] = ({"back_perm": back_perm}, lambda cache, from_above: np.transpose(from_above, axes=cache["back_perm"]))

    return result


def swap_axis(tensor, ax1=-2, ax2=-1):
    """
    Swaps the specified axis of tensor.
    :param tensor: Tensor object.
    :param ax1: Axis to swap.
    :param ax2: Axis to swap.
    :return: New Tensor object.
    """
    result = Tensor(np.swapaxes(tensor.value, ax1, ax2))
    tensor.dependencies[result] = ({"axis": (ax1, ax2)}, lambda cache, from_above: np.swapaxes(from_above, *cache["axis"]))

    return result


def concatenate(tensors, axis):
    """
    Concatenates all tensors in 'tensors'. All of them must have the
    same dimensions, except along 'axis'.
    :param tensors: List of Tensor objects to concatenate.
    :param axis: Axis to concatenate along.
    :return: New Tensor object.
    """
    result = Tensor(np.concatenate([x.value for x in tensors], axis))
    running_sum = 0
    for tensor in tensors:
        tensor.dependencies[result] = ({"index": tuple([slice(None,) for _ in range(axis)] + [slice(running_sum,running_sum + tensor.shape[axis])])}, lambda cache, from_above: from_above[cache["index"]])
        running_sum += tensor.shape[axis]
    return result


def repeat(tensor, rep, axis, backward=const.sum):
    """
    This function tiles the vector along the given axis
    'rep' times.
    :param tensor: Tensor object.
    :param rep: The number of repetition.
    :param axis: Axis to tile along. int.
    :param backward: Callable. Way of dealing with reducing
    dimensions in backward pass. The 'real' gradient of the
    operation is sum, but in some cases it might be desirable
    to average instead.
    :return: New Tensor object.
    """
    new_shape = np.ones((tensor.rank,), dtype=int)
    new_shape[axis] = rep
    result = Tensor(np.tile(tensor.value, new_shape))
    tensor.dependencies[result] = ({"axis":axis, "rep":rep}, lambda cache, from_above: backward(np.array(np.split(from_above, cache["rep"], cache["axis"])), axis=0))

    return result


def tile_leading_dims(tensor, leading_dims, backward=const.sum):
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
    :param backward: Callable. Way of dealing with reducing
    dimensions in backward pass. The 'real' gradient of the
    operation is sum, but in some cases it might be desirable
    to average instead.
    :return New Tensor object.

    """
    if type(leading_dims) is int:
        shape = (leading_dims,)+tensor.shape
    elif type(leading_dims) is tuple:
        shape = leading_dims + tensor.shape
    else:
        raise Exception(f"Argument 'leading dims' in operations.tile_leading_dims(tensor, leading_dims) should be of type in or tuple, not type {type(leading_dims)}")
    result = Tensor(np.broadcast_to(tensor.value, shape))
    tensor.dependencies[result] = ({"sum_axis": tuple(range(len((leading_dims,))))}, lambda cache, from_above: backward(from_above, axis=cache["sum_axis"]))

    return result


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


def sqrt(tensor):
    """
    Element wise square root.
    :param tensor: Tensor object.
    :return: New Tensor object.
    """
    result = Tensor(np.sqrt(tensor.value))
    tensor.dependencies[result] = ({"local": 1 / (2 * np.sqrt(tensor.value))}, lambda cache, from_above: cache["local"] * from_above)

    return result


def exp(tensor):
    """
    e ^ tensor
    :param tensor: Tensor.
    :return: New Tensor object.
    """
    result = Tensor(np.exp(tensor.value))
    tensor.dependencies[result] = ({"local": result.value.copy()}, lambda cache, from_above: from_above * cache["local"])

    return result


def ln(tensor):
    """
    ln(tensor) - natural logging of the tensor.
    :param tensor: Tensor object.
    :return: New Tensor object.
    """
    result = Tensor(np.log(tensor.value))
    tensor.dependencies[result] = ({"local": 1 / tensor.value}, lambda cache, from_above: cache["local"] * from_above)

    return result
