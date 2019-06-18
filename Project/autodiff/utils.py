"""
This file contains more advanced functions, built
out of Tensor objects.
"""

# Third-party dependencies
import numpy as np

# Project files
from autodiff.tensor import Tensor
import autodiff.operations as ops
from autodiff.devtools import unstable, placeholder


def sigmoid(tensor):
    """
    This function implements element-wise
    sigmoid.
    :param tensor: A Tensor object to be applied sigmoid
    to.
    :return: New Tensor object.
    """
    return 1 / (1 + ops.exp(-tensor))


@unstable
def tanh(tensor):
    """
    This function implements the tanh function
    :param tensor: Tensor object.
    :return: New Tensor object.
    """
    return (ops.exp(2 * tensor) - 1) / (ops.exp(2 * tensor) + 1)


@unstable
def softmax(tensor, axis=-2):
    """
    This function implements softmax activation.
    :param tensor: Tensor object.
    :param axis: The axis to perform the softmax over.
    :return: New Tensor object.
    """
    shift = tensor - Tensor(np.max(tensor.value, axis, keepdims=True))
    exps = ops.exp(shift)
    S = ops.repeat(ops.sum(exps, axis), tensor.shape[axis], axis)
    return exps / S


@unstable
def ReLU(tensor):
    """
    This function implements the ReLU activation
    function.
    :param tensor: Tensor object.
    :return: New Tensor object.
    """
    return ops.maximum(tensor, 0)


def MSE(output, label):
    """
    This function implements mean squared error.
    Sum is over the last 2 dimensions.
    :param output: Tensor object. The expected value.
    :param label: Tensor object. The true value.
    :return: New Tensor object.
    """
    return 0.5 * ops.sum((label - output) ** 2, axis=(-2, -1))


def elementwise_cross_entropy(output, label):
    """
    This function implements neuron-wise cross-entropy loss.
    :param output: Tensor object. The expected value.
    :param label: Tensor object. The true value.
    :return: New Tensor object.
    """
    return -(label * ops.ln(output)) + (1 - label) * ops.ln(1 - output + 1e-7)


def vector_cross_entropy(output, label):
    """
    This function implements the cross-entropy loss.
    This function implements neuron-wise cross-entropy loss.
    :param output: Tensor object. The expected value.
    :param label: Tensor object. The true value.
    :return: New Tensor object.
    """
    return - label * ops.ln(output + 1e-7)