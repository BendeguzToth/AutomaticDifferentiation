import numpy as np

a = np.array([
    [
        [1, 3],
        [8, 1]
    ],

    [
        [9, 2],
        [7, 4]
    ]
])

b = np.array([
  [
      [
        [1, 0],
        [0, 1]
      ]
  ],


  [
      [
          [0, 0],
          [1, 1]
      ]
  ],


  [
      [
          [0, 1],
          [0, 1]
      ]
  ]])

print(a)
a[b] = -99
print(a)
# print(b)
# print("====")
print(np.argmax(a, axis=1))

print(np.max(a, axis=1, keepdims=True).shape)

