from autodiff.tensor import Tensor, differentiate

a = Tensor([5.])
b = Tensor([2.])

c = a * b

differentiate(c, [a])

# Now a.grad = [2.0]
print(a.grad)

# Reset a, and remove it from the graph.
a.reset_all()

# If we differentiate again it will be 0, as it was
# removed from the graph.
differentiate(c, [a])
print(a.grad)

# b however still has a gradient of [5.]
differentiate(c, [b])
print(b.grad)
