import numpy as np
from autodiff.tensor import Tensor, derive
from autodiff import utils, operations as ops

tensor = Tensor(np.array([
    [1.2],
    [0.7],
    [5.4]
]))

out = utils.softmax(tensor)
derive(out, [tensor])

print(out)