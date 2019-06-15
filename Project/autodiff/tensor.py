"""
This file contains the implementation of the Tensor class,
the main building block of the autodiff module.
"""

# Third-party libraries
import numpy as np

# Project files
from autodiff.devtools import placeholder, unstable, log


class Tensor:
    """
    A wrapper class around np.ndarray, that keeps track
    of its own gradients. An arbitrary function built out
    of Tensors can be differentiated automatically with
    respect to any of its (Tensor) parameters.
    """
    def __init__(self, value=np.array([])):
        self.__value = value
        self.__shape = value.shape
        self.__rank = len(self.__shape)
        # dict of {Tensor: (cache, func),
        #          Tensor: (cache, func)
        #          ...
        # }
        # Where Tensor is a Tensor object, that is in the forward
        # pass dependent on this node. local_gradient is the gradient
        # of the Tensor object with respect to self, and func is a callable
        # that takes two arguments (cache, from_above) to calculate the gradient
        # of the final node with respect to self. cache is a log dict,
        # from_above is the gradient at Tensor.
        # Call self.clear_dependencies() to clear.
        self.dependencies = {}
        # The gradient of the node. Every time there is a new gradient incoming, it
        # will be added to self.grad. After the gradient is calculated, it needs to
        # be reset by calling self.reset_grad().
        # To backpropagate from a node, its gradient first needs to be set to ones
        # by calling self.start_backprop_here().
        self.grad = np.zeros(shape=self.__shape)
        # This boolean indicates whether the gradient is already calculated for this
        # Tensor. After differentiating it needs to be reset by calling self.set_undone().
        self._done = False

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        self.__value = value

    @property
    def shape(self):
        return self.__shape

    @property
    def rank(self):
        return self.__rank

    @property
    def done(self):
        """
        Returns whether the Tensor has calculated its gradient
        based on the gradients of its dependencies.
        """
        return len(self.dependencies) is 0 or self._done

    def reset_all(self):
        """
        This function clears the backprop cache, and sets everything
        back to its default state.
        -- clears dependencies.
        -- sets grad to zeros.
        -- _done to False.
        """
        self.dependencies.clear()
        self.grad = np.zeros(shape=self.shape)
        self._done = False

    def set_undone(self):
        """
        This function sets the 'done'boolean of the Tensor to False.
        It leaves 'dependencies' and 'grad' unchanged!
        """
        self._done = False

    def reset_grad(self):
        """
        Resets self.grad to all zeros.
        """
        self.grad = np.zeros(shape=self.shape)

    def clear_dependencies(self):
        """
        Clears the dependencies of the Tensor.
        """
        self.dependencies.clear()

    def remove_dependency(self, tensor):
        """
        Removes 'tensor' from the dependencies.
        If tensor not found nothing happens.
        :param tensor: The Tensor object to be removed.
        """
        try:
            self.dependencies.pop(tensor)
        except KeyError:
            pass

    def start_backprop_here(self):
        """
        This function is used to mark the final node in
        the function. The dependencies of this node will
        be cleared, and will receive a gradient of 1.
        """
        self.dependencies.clear()
        self.grad = np.ones(shape=self.shape)
        self._done = True

    def backprop(self):
        """
        Tries to resolve the gradient at the Tensor by
        fetching the gradients of all of its dependencies.
        After it has no dependencies left, it returns its value.
        """
        if not self.done:
            for key, _ in self.dependencies.items():
                self.grad += self.dependencies[key][1](self.dependencies[key][0], key.backprop())
            self._done = True
        return self.grad

    def __copy__(self):
        return Tensor(self.value.copy())

    def __add__(self, other):
        """
        Element-wise add with to another Tensor.
        :param other: Tensor object, with same shape as self.
        :return: The result of the operation.
        """
        assert type(other) is Tensor, "Tensor object can only interact with other Tensor objects."

        result = Tensor(self.value + other.value)
        self.dependencies[result] = ({"local": np.ones(shape=self.shape)}, lambda cache, from_above: cache["local"] * from_above)
        other.dependencies[result] = ({"local": np.ones(shape=other.shape)}, lambda cache, from_above: cache["local"] * from_above)

        return result

    def __sub__(self, other):
        """
        Element-wise subtraction of two Tensor objects.
        self - other
        :param other: Tensor object.
        :return: The element-wise difference of the objects.
        """
        assert type(other) is Tensor, "Tensor object can only interact with other Tensor objects."

        result = Tensor(self.value - other.value)
        self.dependencies[result] = ({"local": np.ones(shape=self.shape)}, lambda cache, from_above: cache["local"] * from_above)
        other.dependencies[result] = ({"local": -np.ones(shape=self.shape)}, lambda cache, from_above: cache["local"] * from_above)

        return result

    @log
    def __mul__(self, other):
        """
        Element-wise multiplication of two Tensor objects.
        :param other: Tensor object.
        :return: The element-wise product of the two tensor objects.
        """
        assert type(other) is Tensor, "Tensor object can only interact with other Tensor objects."

        result = Tensor(self.value * other.value)
        self.dependencies[result] = ({"local": other.value.copy()}, lambda cache, from_above: cache["local"] * from_above)
        other.dependencies[result] = ({"local": self.value.copy()}, lambda cache, from_above: cache["local"] * from_above)

        return result

    def __rmul__(self, other):
        """
        Multiply with scalar.
        :param other: int or float.
        :return: The product of the multiplication.
        """
        assert type(other) is int or type(other) is float, "Left side of multiplication operand of Tensor must be a numeric value"

        result = Tensor(other * self.value)
        self.dependencies[result] = ({"local": np.full(shape=self.shape, fill_value=other)}, lambda cache, from_above: cache["local"] * from_above)

        return result

    def __pow__(self, power):
        result = Tensor(self.value ** power)
        self.dependencies[result] = ({"local": power * self.value ** (power-1)}, lambda cache, from_above: cache["local"] * from_above)

        return result

    def __matmul__(self, other):
        """
        Implements matrix multiply for 2D tensors.
        :param other: Tensor object.
        :return: Value after matrix multiplication.
        """
        result = Tensor(np.dot(self.value, other.value))
        self.dependencies[result] = ({"local": other.value.copy().T}, lambda cache, from_above: np.dot(from_above, cache["local"]))

        return result

    def __str__(self):
        return "TENSOR with value: {}, gradient: {}".format(str(self.value), str(self.grad))

    def __repr__(self):
        return str(self.value)


@placeholder
class TensorGroup:
    """
    This class implements a group of Tensors
    with shared value and grad. Can be used
    for models with loops (e.g. RNNs)
    """


def derive(dF, dx):
    """
    Calculates the derivative of the Tensor dF with respect to dx.
    Sets Tensor.error of all the Tensor object in the list.
    IT WILL SET dF.grad() TO ONES!
    :param dF: The 'final' node.
    :param dx: List of nodes of which we want to know the error.
    """
    dF.start_backprop_here()
    for candidate in dx:
        candidate.backprop()
