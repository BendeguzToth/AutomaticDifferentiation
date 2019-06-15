"""
This file contains more advanced functions, built
out of Tensor objects.
"""

# Third-party dependencies
import numpy as np

# Project files
from autodiff.tensor import Tensor
import autodiff.operations as ops


def sigmoid(tensor):
    """
    This function implements element-wise
    sigmoid.
    :param tensor: A Tensor object to be applied sigmoid
    to.
    :return: New Tensor object.
    """
    return 1 / (1 + ops.exp(-tensor))


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