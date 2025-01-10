from typing import List
from fastapi import FastAPI
import uvicorn 
import numpy as np
from pydantic import BaseModel

app = FastAPI()

class Matrix(BaseModel):
    matrix: List[List[float]]

def matrix_mul(x, y):
    pass

def matrix_mul_without_numpy(x, y):
    pass

def sigmoid(x):
    pass

@app.post("/calculate")
def calculate(matrix: Matrix):
    pass

if __name__ == "__main__":
    uvicorn.run(app)