import numpy as np

M = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
]

x = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]]

B = [
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20]
]


def matrix_mul(M, X, B):
    try:
        product = np.dot(M, X)
        result = product + B
        return result
    except ValueError as e:
        raise ValueError(f"Matrix operation failed: {e}")

