"""
This file contains the implementation of the Tensor class,
the main building block of the autodiff module.
"""

# Third-party libraries
import numpy as np

# Project files
from autodiff.devtools import placeholder, unstable, logging


class Tensor:
    """
    A wrapper class around np.ndarray, that keeps track
    of its own gradients. An arbitrary function built out
    of Tensors can be differentiated automatically with
    respect to any of its (Tensor) parameters.
    """
    def __init__(self, value=np.array([]), name=""):
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
        # of the final node with respect to self. cache is a logging dict,
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
        self.name = name

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

    def __len__(self):
        return self.value.__len__()

    @unstable
    def __neg__(self):
        """
        Implements the '-' operator in front of the
        Tensor.
        :return: New Tensor object.
        """
        result = Tensor(-self.__value)
        self.dependencies[result] = ({"local": -np.ones(shape=self.shape)}, lambda cache, from_above: cache["local"] * from_above)

        return result

    @unstable
    def __add__(self, other):
        """
        Element-wise add with to another Tensor.
        :param other: Tensor object, with same shape as self.
        :return: The result of the operation.
        """
        if type(other) is Tensor:
            result = Tensor(self.value + other.value)
            self.dependencies[result] = ({"local": np.ones(shape=self.shape)}, lambda cache, from_above: cache["local"] * from_above)
            other.dependencies[result] = ({"local": np.ones(shape=other.shape)}, lambda cache, from_above: cache["local"] * from_above)

            return result
        elif type(other) is int or type(other) is float:
            result = Tensor(self.value + other)
            self.dependencies[result] = ({"local": np.ones(shape=self.shape)}, lambda cache, from_above: cache["local"] * from_above)

            return result

        else:
            raise Exception(f"Tensor.__add__(self, other) is only implemented with numeric and Tensor,"
                            f" not with {type(other)}")

    @unstable
    def __radd__(self, other):
        return self.__add__(other)

    @unstable
    def __sub__(self, other):
        """
        Element-wise subtraction of two Tensor objects, or with numeric.
        :param other: Tensor object or numeric.
        :return: The element-wise difference of the objects.
        """
        if type(other) is Tensor:
            result = Tensor(self.value - other.value)
            self.dependencies[result] = ({"local": np.ones(shape=self.shape)}, lambda cache, from_above: cache["local"] * from_above)
            other.dependencies[result] = ({"local": -np.ones(shape=self.shape)}, lambda cache, from_above: cache["local"] * from_above)

            return result

        elif type(other) is int or type(other) is float:
            result = Tensor(self.__value - other)
            self.dependencies[result] = ({"local": np.ones(shape=self.shape)}, lambda cache, from_above: cache["local"] * from_above)

            return result

        else:
            raise Exception(f"Tensor.__sub__(self, other) is only implemented with numeric and Tensor,"
                            f" not with {type(other)}")

    @unstable
    def __rsub__(self, other):
        result = Tensor(other - self.__value)
        self.dependencies[result] = ({"local": np.ones(shape=self.shape)}, lambda cache, from_above: cache["local"] * from_above)

        return result

    @unstable
    @logging
    def __mul__(self, other):
        """
        Element-wise multiplication with Tensor object or numeric.
        :param other: Tensor object or in or float.
        :return: The element-wise product.
        """
        if type(other) is Tensor:
            result = Tensor(self.value * other.value)
            self.dependencies[result] = ({"local": other.value.copy()}, lambda cache, from_above: cache["local"] * from_above)
            other.dependencies[result] = ({"local": self.value.copy()}, lambda cache, from_above: cache["local"] * from_above)

            return result
        elif type(other) is int or type(other) is float:
            result = Tensor(other * self.value)
            self.dependencies[result] = ({"local": np.full(shape=self.shape, fill_value=other)},
                                         lambda cache, from_above: cache["local"] * from_above)

            return result
        else:
            raise Exception(f"Tensor.__mul__(self, other) is only implemented with numeric and Tensor,"
                            f" not with {type(other)}")

    @unstable
    @logging
    def __truediv__(self, other):
        """
        Implements the '/' operator.
        :param other: Numeric or Tensor.
        :return: New Tensor object.
        """
        if type(other) is int or type(other) is float:
            result = Tensor(self.__value / other)
            self.dependencies[result] = ({"local": np.full(shape=self.shape, fill_value=1 / other)}, lambda cache, from_above: cache["local"] * from_above)

            return result

        elif type(other) is Tensor:
            result = Tensor(self.__value / other.value)
            self.dependencies[result] = ({"local": 1 / other.value}, lambda cache, from_above: cache["local"] * from_above)
            other.dependencies[result] = ({"local": - self.value / (other.value ** 2)}, lambda cache, from_above: cache["local"] * from_above)

            return result

        else:
            raise Exception(f"Tensor.__truediv__(self, other) is only implemented with numeric and Tensor,"
                            f" not with {type(other)}")

    @unstable
    def __rtruediv__(self, other):
        """
        Implements the '/' operator.
        Tensor / Tensor cases are handled in Tensor.__truediv__,
        here we only handle numeric / Tensor.
        :param other: Numeric.
        :return: New Tensor object.
        """
        assert type(other) is float or type(other) is int

        result = Tensor(other / self.__value)
        self.dependencies[result] = ({"local": - other / (self.__value ** 2)}, lambda cache, from_above: cache["local"] * from_above)

        return result

    @unstable
    def __rmul__(self, other):
        return self.__mul__(other)

    @unstable
    def __pow__(self, power):
        result = Tensor(self.value ** power)
        self.dependencies[result] = ({"local": power * self.value ** (power-1)}, lambda cache, from_above: cache["local"] * from_above)

        return result

    @placeholder
    def __rpow__(self, other):
        pass

    @unstable
    def __matmul__(self, other):
        """
        Last 2 dimensions have to be valid for dot product,
        all axis before that needs to match.
        :param tensor_a:
        :param tensor_b:
        """
        result = Tensor(np.matmul(self.value, other.value))
        self.dependencies[result] = ({"local": np.swapaxes(other.value.copy(), -1, -2)}, lambda cache, from_above: np.matmul(from_above, cache["local"]))
        other.dependencies[result] = ({"local": np.swapaxes(self.value.copy(), -1, -2)}, lambda cache, from_above: np.matmul(cache["local"], from_above))

        return result

    def __str__(self):
        return "TENSOR {} with value:\n{}, gradient:\n{}".format(self.name, str(self.value), str(self.grad))

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


@placeholder
def collapse_grad(tensor, nr_leading_dims):
    """
    Call this function on a Tensor object to collapse its gradient
    down by ignoring some number of leading dimensions.
    :param tensor:
    :param nr_leading_dims:
    :return:
    """


def xavier_tensor(shape):
    """
    This function initializes a Tensor with the given shape
    using the Xavier initialization. Last dim is normalized.
    :param shape: The shape of the Tensor.
    :return: Tensor object with the specified shape.
    """
    return Tensor(np.random.randn(*shape) / np.sqrt((shape[-1])))


def xavier_relu_tensor(shape):
    """
    This function initializes a Tensor with the given shape
    using the Xavier initialization for ReLU.. Last dim is
    normalized.
    :param shape: The shape of the Tensor.
    :return: Tensor object with the specified shape.
    """
    return Tensor(np.random.randn(*shape) / np.sqrt((shape[-1] / 2)))


def unit_normal_tensor(shape):
    """
    Returns a Tensor with the specified shape, where all values
    are drawn from a unit gaussian distribution.
    :param shape: Shape of the Tensor.
    :return: Unit gaussian Tensor object with the specified shape.
    """
    return Tensor(np.random.randn(*shape))