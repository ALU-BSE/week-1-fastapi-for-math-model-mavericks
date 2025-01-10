import math
from plain_multiplication import plain_mat_mul


def sigmoid(x):
    """Apply sigmoid function to an input x."""
    return 1 / (1 + math.exp(-x))

def apply_sigmoid_to_matrix(X):
    """Multiply matrices (M * X) + B and apply sigmoid to each element."""

    # Call the matrix multiplication function to get (M * X) + B
    result = plain_mat_mul(X)
    
    # Apply sigmoid function to each element in the result matrix
    result_sigmoid = [[sigmoid(result[i][j]) for j in range(len(result[0]))] for i in range(len(result))]
    
    for row in result_sigmoid:
        print(row)

    return result_sigmoid


apply_sigmoid_to_matrix([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],
                         [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
                         [21, 22, 23, 24, 25]])