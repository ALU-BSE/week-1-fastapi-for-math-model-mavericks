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

def matrix_mul_without_numpy(x):
    pass

def sigmoid(x):
    pass

@app.post("/calculate")
def f(matrix: Matrix):
    pass

if __name__ == "__main__":
    uvicorn.run(app)