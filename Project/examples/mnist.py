"""
In this file we show an example of the autodiff package
by solving the MNIST dataset for handwritten digit classification.
"""

import numpy as np

from examples.data.mnist_loader import load_data_wrapper

import autodiff.utils as utils
from autodiff.tensor import Tensor, derive, xavier, unit_normal
import autodiff.operations as ops
from autodiff import devtools

devtools.ENABLE_DECORATORS = False


# Helper functions for loading in the data.
def loadData(file):
    raw_data = load_data_wrapper(file)
    return list(raw_data[0]), list(raw_data[1]), list(raw_data[2])


training, evaluation, testing = loadData("data/mnist.pkl.gz")

training_data = np.array(training[0])
training_labels = np.array(training[1])

test_data = np.array(testing[0])
test_labels = np.array(testing[1])


def generate_training_data(data, labels, batch_size):
    """
    This generator will yield a training batch.
    :param data: Numpy array of the input data.
    :param labels: Numpy array of the labels.
    :param batch_size: The size of the generated batch.
    :return: A tuple of (batch_data, batch_labels)
    """
    for start in range(0, len(data), batch_size):
        yield Tensor(data[start:start+batch_size, ...]), Tensor(labels[start:start+batch_size, ...])


BATCH_SIZE = 100
LEARNING_RATE = 0.01
N_EPOCH = 10


class Layer:
    def __init__(self, size, input_size, activation):
        self.w = xavier(shape=(size, input_size))
        self.b = unit_normal(shape=(size, 1))
        self.activation = activation

    def __call__(self, batch):
        batch_size = len(batch)
        z = ops.tile_leading_dims(self.w, batch_size) @ batch + ops.tile_leading_dims(self.b, batch_size)
        return self.activation(z)

    def update(self, lr):
        self.w.value -= lr*self.w.grad
        self.b.value -= lr*self.b.grad

        self.w.reset_all()
        self.b.reset_all()


hidden_layer = Layer(60, 784, activation=utils.relu)
output_layer = Layer(10, 60, activation=utils.softmax)

for epoch in range(1, N_EPOCH):
    # Training
    for data, label in generate_training_data(training_data, training_labels, BATCH_SIZE):
        loss = utils.vector_cross_entropy(output_layer(hidden_layer(data)), label)

        derive(loss, [hidden_layer.w, hidden_layer.b, output_layer.w, output_layer.b])

        hidden_layer.update(LEARNING_RATE)
        output_layer.update(LEARNING_RATE)

    # Validating
    for data, label in generate_training_data(test_data, test_labels, 10000):
        out = output_layer(hidden_layer(data))
        idxs = np.reshape(np.argmax(out.value, axis=1), newshape=(10000,))
        result = np.zeros(shape=(10000, 10, 1))
        for example in range(len(result)):
            result[example][idxs[example], 0] = 1.

        print(f"Epoch {epoch} - Eval accuracy: {np.count_nonzero(result * label.value) / 10000}")
