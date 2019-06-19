import autodiff.constants as const
const.devmode = False

import numpy as np
import autodiff.tensor as tn
from autodiff.tensor import Tensor
from autodiff import utils, operations as ops


class PointWiseNN:
    """
    Point wise neural network class
    for the encoder.
    """
    def __init__(self, hidden_size, io_size):
        """
        Ctor.
        :param hidden_size: Size of the hidden layer.
        :param io_size: Size of the input and the output
        layer.
        """
        self.__w1 = tn.xavier_relu((hidden_size, io_size))
        self.__b1 = tn.unit_normal((hidden_size, 1))

        self.__w2 = tn.xavier_relu((io_size, hidden_size))
        self.__b2 = tn.unit_normal((io_size, 1))

    @property
    def variables(self):
        return self.__w1, self.__w2, self.__b1, self.__b2

    def __call__(self, x):
        """
        The forward pass of the network.
        :param x: Batch of sequences of data.
        Dimensions: batch_size, sequence_length, feature_size, 1.
        :return: Batch of sequences of column vectors of the same
        dim as x.
        """
        w1, b1, w2, b2 = self.__tile_up(x.shape[:2])
        return w2 @ utils.relu(w1 @ x + b1) + b2

    def __tile_up(self, leading_dims):
        """
        This function tiles the variables along the batch and
        sequence axis, with the specified dimensions.
        :param leading_dims: Leading dimensions to tile.
        Tuple of (batch_size, seq_length)
        :return: Tuple of tiled tensors [w1, b1, w2, b2]
        """
        return (ops.tile_leading_dims(self.__w1, leading_dims), ops.tile_leading_dims(self.__b1, leading_dims),
               ops.tile_leading_dims(self.__w2, leading_dims), ops.tile_leading_dims(self.__b2, leading_dims))


def lookahead_mask(size):
    mask = np.zeros((size, size))
    mask[np.triu_indices_from(mask, 1)] = 1
    return Tensor(mask)


def attention(q, k, v):
    """
    Attention.
    :param q: (dk, 1)
    :param k: (sequence, dk, 1)
    :param v: (sequence, dm, 1)
    :return:
    """
    # Smooth k to (sequence, features).
    k = ops.reshape(k, (k.shape[0], k.shape[1] * k.shape[2]))
    similarity = k @ q
    scaled = similarity / k.shape[-2]
    raw_scores = utils.softmax(scaled, axis=-2)
    return raw_scores


# temp_k = tn.tensor([[10, 0, 0],
#                     [0, 10, 0],
#                     [0, 0, 10],
#                     [0, 0, 10]])  # (4, 3)
#
# temp_q = tn.tensor([[0, 10, 0]])  # (1, 3)

temp_q = tn.ones((3, 1))
temp_k = tn.ones((4, 3, 1))
print(attention(temp_q, temp_k, None))

# data = Tensor(np.random.randn(25, 9, 32, 1))
#
# net = PointWiseNN(64, 32)
#
#
# ops.tile_leading_dims(net.variables[0], (25, 9))
#
# out = net(data)
# print(ops.reshape(ops.sum(out), newshape=(1,)))