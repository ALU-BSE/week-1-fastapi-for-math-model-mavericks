from typing import List
from fastapi import FastAPI
import uvicorn 
import numpy as np
from pydantic import BaseModel
import math

app = FastAPI()

M = np.ones((5, 5))  # Shape is passed as a tuple
B = np.ones((5, 5))  # Same here for the bias matrix

# Define a Matrix class using Pydantic to validate the input data
class Matrix(BaseModel):
    matrix: List[List[float]]

def matrix_mul(x):
    """
    Perform matrix multiplication using NumPy's dot function and then add matrix B.
    Args:
        x: The input matrix for multiplication.
    Returns:
        The result of the matrix multiplication and addition with matrix B.
    """
    try:
        return np.dot(M,x) + B
    except ValueError as e:
        raise ValueError(f"Matrix multiplication failed: {e}")


def matrix_mul_without_numpy(X):
    """
    Perform matrix multiplication (M * X) manually, followed by matrix addition (MX + B).
    
    Args:
        X: The input matrix to be multiplied by matrix M.
    
    Returns:
        Y: The resulting matrix after multiplication and addition.
    """
    # Create result matrix
    MX = [[0] * len(X[0]) for _ in range(len(M))]

    # Do multiplication (MX = M * X)
    for i in range(len(M)):
        for j in range(len(X[0])):
            for k in range(len(X)):
                MX[i][j] += M[i][k] * X[k][j]
                
    
        
    # Do addition (MX + B)
    Y = [
        [MX[i][j] + B[i][j] for j in range(len(MX[0]))] for i in range(len(MX))
    ]

    # Return MX + B
    return Y

def sigmoid(X):
    """
    Perform matrix multiplication (M * X) + B, then apply the sigmoid function () to each element.
    
    Args:
        X: The input matrix to be processed.
    
    Returns:
        result_sigmoid: The matrix after applying the sigmoid function element-wise.
    """    
    # Apply sigmoid function to each element in the result matrix
    result_sigmoid = [[1 / (1 + math.exp(-X[i][j])) for j in range(len(X[0]))] for i in range(len(X))]
    
    for row in result_sigmoid:
        print(row)

    return result_sigmoid

@app.post(
    "/calculate",
    description="""
    Perform matrix operations on a given 5x5 matrix.

    This endpoint accepts a JSON object containing a 5x5 matrix and performs the following:
    
    1. **Matrix Multiplication (M * X + B)**:
        - Uses predefined matrices `M` (weights) and `B` (bias).
        - Calculates the matrix multiplication using NumPy and a manual (non-NumPy) implementation.
    
    2. **Sigmoid Function**:
        - Applies the sigmoid function element-wise on the matrix multiplication result (NumPy implementation only).

    **Request Example**:
    ```json
    {
        "matrix": [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25]
        ]
    }
    ```

    **Response Example**:
    ```json
    {
        "matrix_multiplication": [
            [12.0, 13.5, 14.1, 15.6, 16.2],
            [25.1, 28.7, 31.2, 33.4, 36.5],
            [45.3, 48.9, 50.8, 52.3, 54.0],
            [67.9, 70.1, 72.5, 74.3, 77.0],
            [89.2, 90.8, 94.0, 96.2, 99.9]
        ],
        "non_numpy_multiplication": [
            [12.0, 13.5, 14.1, 15.6, 16.2],
            [25.1, 28.7, 31.2, 33.4, 36.5],
            [45.3, 48.9, 50.8, 52.3, 54.0],
            [67.9, 70.1, 72.5, 74.3, 77.0],
            [89.2, 90.8, 94.0, 96.2, 99.9]
        ],
        "sigmoid_output": [
            [0.999993, 0.999998, 0.999999, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ]
    }
    ```

    **Note**:
    - The input matrix must be a valid 5x5 matrix.
    - The `matrix_multiplication` and `non_numpy_multiplication` results should match.
    """
)
def f(matrix: Matrix):
    # Check if matrix is 5x5
    if len(matrix.matrix) != 5 or len(matrix.matrix[0]) != 5:
        return {"error": "Input matrix must be a 5x5 matrix."}

    # Perform calculations
    numpy_result = matrix_mul(matrix.matrix)  # Using NumPy
    non_numpy_result = matrix_mul_without_numpy(matrix.matrix)  # Without NumPy
    sigmoid_result = sigmoid(non_numpy_result)  # Apply sigmoid to NumPy result

    # Return results
    return {
        "matrix_multiplication": numpy_result.tolist(),
        "non_numpy_multiplication": non_numpy_result,
        "sigmoid_output": sigmoid_result,
    }

if __name__ == "__main__":
    uvicorn.run(app)
