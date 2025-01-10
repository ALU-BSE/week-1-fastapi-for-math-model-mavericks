from typing import List
from fastapi import FastAPI
import uvicorn 
import numpy as np
from pydantic import BaseModel
import math

app = FastAPI()

M = np.zeros(5, 5)
B = np.zeros(5, 1)

class Matrix(BaseModel):
    data: List[List[float]]

def matrix_mul(x):
    pass

def matrix_mul_without_numpy(X):
    # Create result matrix
    MX = [[0] * len(X[0]) for _ in range(len(M))]

    # Do multiplication (MX)
    for i in range(len(M)):
        for j in range(len(X[0])):
            for k in range(len(X)):
                MX[i][j] += M[i][k] * X[k][j]
    
    # Do addition ( + B)
    Y = [
        [MX[i][j] + B[i][j] for j in range(len(MX[0]))] for i in range(len(MX))
    ]

    # return MX + B
    return Y

def sigmoid(X):
    """Multiply matrices (M * X) + B and apply sigmoid to each element."""
    
    # Call the matrix multiplication function to get (M * X) + B
    result = matrix_mul_without_numpy(X)
    
    # Apply sigmoid function to each element in the result matrix
    result_sigmoid = [[1 / (1 + math.exp(-result[i][j])) for j in range(len(result[0]))] for i in range(len(result))]
    
    for row in result_sigmoid:
        print(row)

    return result_sigmoid

@app.post("/calculate")
def f(matrix: Matrix):
    pass

if __name__ == "__main__":
    uvicorn.run(app)