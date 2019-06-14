import logging
import numpy as np
from autodiff.tensor import Tensor, derive
import autodiff.operations as ops

logging.basicConfig(level=logging.INFO)

a = Tensor(np.array([2., 4.]))
b = Tensor(np.array([5., 9]))

c = a * b
d = ops.reshape(c, (2, 1))
print(d)

derive(d, [a, b])

print(c)

