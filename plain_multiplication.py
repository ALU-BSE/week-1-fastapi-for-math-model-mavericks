M = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
]

B = [
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20]
]

def plain_mat_mul(X):
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