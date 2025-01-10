import numpy as np

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
