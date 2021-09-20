## Automatic Differentiation
This project implements a tensor class on top of the numpy ndarray that keeps track of its dependencies run-time. This dynamic computational graph can then be used to differentiate the nodes.
```python
from autodiff.tensor import Tensor, differentiate

a = Tensor([5.])
b = Tensor([2.])

c = a * b

differentiate(c, [a])

# Now a.grad = [2.0]

# Reset a, and remove it from the graph.
a.reset_all()

# If we differentiate again it will be 0, as it was
# removed from the graph.
differentiate(c, [a])

# Now a.grad = [0.]

# b however still has a gradient of [5.]
differentiate(c, [b])

```

This can be used to define and train neural networks only defining the forward pass:

```python
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
```
A full example of a mini network trained on MNIST can found in `examples`.