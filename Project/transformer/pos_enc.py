import numpy as np
import matplotlib.pyplot as plt


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(sequence_len, d_model):
    angle_rads = get_angles(np.arange(sequence_len)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return pos_encoding

dmodel = 4
seq = 3

pos_encoding = positional_encoding(sequence_len=seq, d_model=dmodel)
print (pos_encoding.shape)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, dmodel))
plt.ylabel('Position')
plt.colorbar()
plt.show()
