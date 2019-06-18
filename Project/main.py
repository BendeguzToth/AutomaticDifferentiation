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

b = np.array([[
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

# print(b)
# print("====")
# print(np.argmax(a, axis=0))

print(np.max(a, axis=1))

