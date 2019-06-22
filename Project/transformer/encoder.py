import autodiff.constants as const
const.devmode = False

import numpy as np
import math
import autodiff.tensor as tn
from autodiff.tensor import Tensor
from autodiff import utils, operations as ops

np.set_printoptions(suppress=True)


class LayerNorm:
    """
    This class implements layer
    normalization.
    """
    def __init__(self):
        self.gamma = tn.tensor([1])
        self.beta = tn.tensor([0])

    def __call__(self, x):
        batch_size = x.shape[0]
        feature_len = x.shape[1]
        seq_len = x.shape[2]
        mean = ops.repeat(ops.average(x, axis=1), feature_len, axis=1)
        var = ops.repeat(ops.variance(x, axis=1), feature_len, axis=1)

        core = (x - mean) / ops.sqrt(var + const.fuzz)

        return ops.repeat(ops.tile_leading_dims(self.gamma, (batch_size, feature_len)), seq_len, axis=2) * core + ops.repeat(ops.tile_leading_dims(self.beta, (batch_size, feature_len)), seq_len, axis=2)


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


# def attention(q, k, v):
#     """
#     Attention.
#     :param q: (dk, 1)
#     :param k: (sequence, dk, 1)
#     :param v: (sequence, dm, 1)
#     :return:
#     """
#     # Smooth k to (sequence, features).
#     k = ops.reshape(k, (k.shape[0], k.shape[1] * k.shape[2]))
#     similarity = k @ q
#     scaled = similarity / k.shape[-2]
#     raw_scores = utils.softmax(scaled, axis=-2)
#     return raw_scores




class QKV:
    def __init__(self, embedding_length, qk_length, v_length):
        self.wq = tn.xavier((qk_length, embedding_length))
        self.wk = tn.xavier((qk_length, embedding_length))
        self.wv = tn.xavier((v_length, embedding_length))

        self.dk = qk_length

    def __call__(self, x):
        """
        This function calculates the q, k, v vectors for the provided
        sequence.
        :param x: Sequence of column vectors.
        :return: Tuple of (q, k, v)
        """
        # Sequence length.
        q = self.wq @ x
        k = self.wk @ x
        v = self.wv @ x
        raw_scores = ops.swap_axis(k) @ q
        scaled = raw_scores / math.sqrt(self.dk)
        final_scores = utils.softmax(scaled)
        return v @ final_scores


def attention(q, k, v):
    """
    (FxS)
    :param q:
    :param k:
    :param v:
    :return:
    """
    raw_scores = ops.swap_axis(k) @ q
    scaled = raw_scores / math.sqrt(k.shape[0])
    final_scores = utils.softmax(scaled)
    return final_scores, v @ final_scores


# k = tn.tensor([
#     [10, 0, 0, 0],
#     [0, 10, 0, 0],
#     [0, 0, 10, 10]
# ], dtype="float32")
#
#
# v = tn.tensor([
#     [1, 10, 100, 1000],
#     [0, 0, 5, 6]
# ], dtype="float32")
#
# q = tn.tensor([
#     [0, 0, 10],
#     [0, 10, 10],
#     [10, 0, 0]
# ], dtype="float32")
#
# fs, res = attention(q, k, v)
# print(ops.swap_axis(res))
# print(ops.swap_axis(fs))
#
#
# print(lookahead_mask(3))


class MHA:
    def __init__(self, n_heads, embedding_length, qk_length, v_length):
        self.n_heads = n_heads
        self.wq = tn.xavier((qk_length * n_heads, embedding_length))
        self.wk = tn.xavier((qk_length * n_heads, embedding_length))
        self.wv = tn.xavier((v_length * n_heads, embedding_length))
        self.wo = tn.xavier((embedding_length, n_heads * v_length))

    def __call__(self, x):
        """
        :param x: (BxFxS)
        :return:
        """
        batch_size = x.shape[0]
        # (B, dqkv, S)
        q = ops.tile_leading_dims(self.wq, batch_size) @ x
        k = ops.tile_leading_dims(self.wk, batch_size) @ x
        v = ops.tile_leading_dims(self.wv, batch_size) @ x

        q = self.prepare_heads(q)
        k = self.prepare_heads(k)
        v = self.prepare_heads(v)

        attention = self.attention(q, k, v) # BxHxFxS
        attention = ops.swap_axis(attention, 1, 2) # (BxSxHxF)
        reshaped = ops.reshape(attention, (attention.shape[0], attention.shape[1]* attention.shape[2], attention.shape[3])) # (BxFxS)
        # final = ops.swap_axis(reshaped, 1, 2) # (BxSxF)
        return ops.tile_leading_dims(self.wo, batch_size) @ reshaped

    def vars(self):
        """
        Returns a list of trainable variables.
        :return: List[Tensor]
        """
        return [self.wq, self.wk, self.wv, self.wo]

    def prepare_heads(self, x):
        """
        (BxFxS) -> (BxHxSxF)
        :param x:
        :return:
        """
        bsf = ops.swap_axis(x, 1, 2)
        bhsf = ops.reshape(bsf, (bsf.shape[0], bsf.shape[1], self.n_heads, -1))
        return ops.swap_axis(bhsf, 1, 2) # (BxHxSxF)

    @staticmethod
    def attention(q, k, v):
        """
        (BxHxSxF)
        :param q:
        :param k:
        :param v:
        :return:
        """
        # raw_scores = ops.swap_axis(k) @ q           # fine
        # scaled = raw_scores / math.sqrt(k.shape[-1])# should be ok
        # final_scores = utils.softmax(scaled, -1)
        # return v @ final_scores
        raw_scores = k @ ops.swap_axis(q)  # fine
        scaled = raw_scores / math.sqrt(k.shape[-1])  # should be ok
        final_scores = utils.softmax(scaled, -1)
        return ops.swap_axis(v) @ final_scores


mha = MHA(n_heads=8, embedding_length=16, qk_length=8, v_length=16)

tensor = tn.tensor( [[[2.1, 3.3, 9.5],
                    [0.6, -1.2, -0.4],
                    [0.8, -2.0, 1.1],
                    [4.5, -0.0, 2.7]]],
                    dtype="float32"
                   )

ln = LayerNorm()

res = ln(tensor)
print(res.value)
print(np.average(res.value, axis=1))
