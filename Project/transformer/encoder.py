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

    def vars(self):
        return [self.gamma, self.beta]

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

    def vars(self):
        return [self.__w1, self.__w2, self.__b1, self.__b2]

    def __call__(self, x):
        """
        The forward pass of the network.
        :param x: Batch of sequences of data. (BxFxS)
        :return: Batch of sequences of column vectors of the same
        dim as x.
        """
        w1, b1, w2, b2 = self.__tile_up(x.shape[0], x.shape[2])
        return w2 @ utils.relu(w1 @ x + b1) + b2

    def __tile_up(self, batch_dim, sequence_dim):
        """
        This function tiles the variables along the batch and
        sequence axis, with the specified dimensions.
        :param batch_dim: Dimension along batch axis.
        :param seq_dim: Dimension along sequence axis.
        :return: Tuple of tiled tensors (w1, b1, w2, b2)
        with the right shape.
        """
        return ops.tile_leading_dims(self.__w1, batch_dim), ops.repeat(ops.tile_leading_dims(self.__b1, batch_dim), sequence_dim, axis=2),\
               ops.tile_leading_dims(self.__w2, batch_dim), ops.repeat(ops.tile_leading_dims(self.__b2, batch_dim), sequence_dim, axis=2),
        # return (ops.tile_leading_dims(self.__w1, leading_dims), ops.tile_leading_dims(self.__b1, leading_dims),
        #        ops.tile_leading_dims(self.__w2, leading_dims), ops.tile_leading_dims(self.__b2, leading_dims))


def lookahead_mask(size):
    mask = np.zeros((size, size))
    mask[np.triu_indices_from(mask, 1)] = 1
    return Tensor(mask)


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
    return v @ final_scores


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
        self.d_model = embedding_length
        self.wq = tn.xavier((qk_length * n_heads, embedding_length))
        self.wk = tn.xavier((qk_length * n_heads, embedding_length))
        self.wv = tn.xavier((v_length * n_heads, embedding_length))
        self.wo = tn.xavier((embedding_length, n_heads * v_length))

        self.w_padding_mask = tn.ones(self.wk.shape)

    def __call__(self, x, padding_mask, mask=None):
        """
        :param x: (BxFxS)
        :return:
        """
        batch_size = x.shape[0]
        # (B, dqkv, S)
        q = ops.tile_leading_dims(self.wq, batch_size) @ x
        k = ops.tile_leading_dims(self.wk, batch_size) @ x
        v = ops.tile_leading_dims(self.wv, batch_size) @ x

        q = self.prepare_heads(q)  # (BxHxSxF)
        k = self.prepare_heads(k)  # (BxHxSxF)
        v = self.prepare_heads(v)  # (BxHxSxF)

        padding_mask_4d = self.prepare_heads(ops.tile_leading_dims(self.w_padding_mask, batch_size) @ padding_mask) / self.d_model  # (BxHxSxF)

        attention = self.attention(q, k, v, padding_mask_4d, mask) # BxHxFxS
        attention = ops.swap_axis(attention, 1, 2) # (BxSxHxF)
        reshaped = ops.reshape(attention, (attention.shape[0], attention.shape[1]* attention.shape[2], attention.shape[3])) # (BxFxS)

        return (1 - padding_mask) * (ops.tile_leading_dims(self.wo, batch_size) @ reshaped)

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
        bsf = ops.swap_axis(x, 1, 2) # (BxSxF)
        bshf = ops.reshape(bsf, (bsf.shape[0], bsf.shape[1], self.n_heads, -1)) # (BxSxHxF)
        return ops.swap_axis(bshf, 1, 2) # (BxHxSxF)

    @staticmethod
    def attention(q, k, v, padding_mask, mask=None):
        """
        (BxHxSxF)
        :param q:
        :param k:
        :param v:
        :param mask:
        :return: (BxHxFxS)
        """
        dqk = q.shape[3]
        mask = padding_mask @ ops.swap_axis(tn.ones(q.shape)) / dqk
        raw_scores = k @ ops.swap_axis(q)
        scaled = raw_scores / math.sqrt(k.shape[3])
        if mask is not None:
            scaled += (mask * -1e9)
        final_scores = utils.softmax(scaled, axis=2)  # S
        return ops.swap_axis(v) @ final_scores


class EncoderBlock:
    """
    NO DROPOUT YET!
    """
    def __init__(self, dmodel, n_heads, dqk, hidden_size):
        self.mha = MHA(n_heads=n_heads, embedding_length=dmodel, qk_length=dqk, v_length=dmodel)
        self.nn = PointWiseNN(hidden_size, dmodel)

        self.layernorm1 = LayerNorm()
        self.layernorm2 = LayerNorm()

    def vars(self):
        return self.mha.vars() + self.nn.vars() + self.layernorm1.vars() + self.layernorm2.vars()

    def __call__(self, x, mask):
        attention_out = self.layernorm1(self.mha(x, mask) + x)
        return self.layernorm2(self.nn(attention_out) + attention_out)


data = np.array([
    [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0]
    ],

    [
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
])

mask = np.array([
    [
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1]
    ],

    [
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1]
    ],
])

mha = MHA(2, 3, 3, 3)

res = mha(tn.tensor(data), tn.tensor(mask))

print(res)

other = tn.tensor([
    [
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 0]
    ]
])

other_mask = tn.tensor([
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
])

print(mha(other, other_mask))