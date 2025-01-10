import numpy as np

M = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
]

B = [
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20]
]

def get_matrix_from_user():
    
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    print(f"Enter the elements of the matrix ({rows}x{cols}), row by row:")
    matrix = []
    for i in range(rows):
        row = list(map(float, input(f"Row {i + 1}: ").split()))
        if len(row) != cols:
            raise ValueError(f"Row {i + 1} must have exactly {cols} elements.")
        matrix.append(row)
    return np.array(matrix)

def matrix_multiply(matrix1, matrix2):
    try:
        return np.dot(matrix1, matrix2)
    except ValueError as e:
        raise ValueError(f"Matrix multiplication failed: {e}")


def plain_mat_mul(X):
    # Create result matrix
    MX = [[0] for _ in len(X[0]) for _ in len(M)]

    # Do multiplication (MX)
    for i in range(len(M)):
        for j in range(len(X[0])):
            for k in range(len(X)):
                MX[i][j] += M[i][k] * X[k][j]
    
    # Do addition ( + B)
    Y = [
        [MX[i][j] + B[i][j] for j in range(len(MX[0]))] for i in range(MX)
    ]

    # return
    return Y
