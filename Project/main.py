import logging
import numpy as np
from autodiff.tensor import Tensor, derive, TensorGroup
import autodiff.operations as ops
from autodiff.devtools import unstable

logging.basicConfig(level=logging.INFO)
#
# a = Tensor(np.array([2., 4.]))
# b = Tensor(np.array([5., 9]))
#
# c = a * b
# d = ops.reshape(c, (2, 1))
# print(d)
#
# derive(d, [a, b])
#
# print(c)


lr = 0.005

gewicht  = Tensor(np.array([[0.82, 0.75]])) # shape = (2, 1)
bias = Tensor(np.array([[1.2]]))


def xy():
    x = Tensor(np.random.randn(10, 2, 1) * 3)
    y = ops.matmul(ops.tile_leading_dims(gewicht, leading_dims=10), x) + ops.tile_leading_dims(bias, leading_dims=10)
    return x, y

def test():
    w = Tensor(np.random.randn(1, 2))
    b = Tensor(np.random.rand(1, 1))

    for i in range(100):
        x, y = xy()

        out = ops.matmul(ops.tile_leading_dims(w, 10), x) + ops.tile_leading_dims(b, 10)
        print(ops.mean())
        loss = 0.5 * ops.mean(ops.sum((y - out) ** 2, axis=(1, 2)))

        print("Epoch {} Loss {}".format(i, loss))

        derive(loss, [w, b])
        w.value -= lr * w.grad
        b.value -= lr * b.grad
        w.reset_all()
        b.reset_all()
    print(w)
    print(b)

# x, y = xy()
# print(x.shape)
# print(y.shape)
test()
