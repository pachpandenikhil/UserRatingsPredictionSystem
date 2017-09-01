import sys, math, warnings
import numpy as np


# Reads utility matrix from file
def read_utility_matrix(file, rows, cols):
    f = open(file)
    matrix = np.zeros((rows, cols), dtype=np.float)
    for line in iter(f):
        line = line.rstrip()
        if line:
            entry = line.split(',')
            row = int(entry[0]) - 1
            col = int(entry[1]) - 1
            val = float(entry[2])
            matrix[row][col] = float(val)
    return matrix


# Returns the initial U and V matrices with all elements set to 1
def get_initial_decomposition(n, m, d):
    U = np.ones((n, d), dtype=np.float)
    V = np.ones((d, m), dtype=np.float)
    return U, V


# Computes RMSE between utility matrix M and decomposition UV
def get_RMSE(M, UV, rows, cols):
    squared_error = 0.0
    count = 0
    for i in range(rows):
        squared_diff_sum = 0.0
        for j in range(cols):
            if M[i][j] != 0:
                count += 1
                diff = M[i][j] - UV[i][j]
                diff *= diff
                squared_diff_sum += diff
        squared_error += squared_diff_sum
    return math.sqrt(squared_error/count)


# Returns optimal value for element U[r][s]
def optimize_x(M, U, V, r, s, m, d):
    # calculating numerator
    numerator = 0.0
    for j in range(m):
        if M[r][j] == 0:
            continue
        Vsj = V[s][j]
        Mrj = M[r][j]
        sum_prod = 0.0
        for k in range(d):
            if k != s:
                sum_prod += U[r][k] * V[k][j]
        numerator += Vsj * (Mrj - sum_prod)

    # calculating denominator
    denominator = 0.0
    for j in range(m):
        if M[r][j] == 0:
            continue
        denominator += V[s][j] * V[s][j]

    return numerator/denominator


# Returns optimal value for element V[r][s]
def optimize_y(M, U, V, r, s, n, d):
    # calculating numerator
    numerator = 0.0
    for i in range(n):
        if M[i][s] == 0:
            continue
        Uir = U[i][r]
        Mis = M[i][s]
        sum_prod = 0.0
        for k in range(d):
            if k != r:
                sum_prod += U[i][k] * V[k][s]
        numerator += Uir * (Mis - sum_prod)

    # calculating denominator
    denominator = 0.0
    for i in range(n):
        if M[i][s] == 0:
            continue
        denominator += U[i][r] * U[i][r]

    return numerator/denominator


# main execution
if __name__ == '__main__':

    # reading utility matrix
    utility_matrix_file = sys.argv[1]
    utility_matrix_rows = int(sys.argv[2])
    utility_matrix_cols = int(sys.argv[3])
    factors = int(sys.argv[4])
    iterations = int(sys.argv[5])

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    M = read_utility_matrix(utility_matrix_file, utility_matrix_rows, utility_matrix_cols)
    U, V = get_initial_decomposition(utility_matrix_rows, utility_matrix_cols, factors)

    for iteration in range(iterations):
        # learning elements in U
        for i in range(utility_matrix_rows):
            for j in range(factors):
                U[i][j] = optimize_x(M, U, V, i, j, utility_matrix_cols, factors)

        # learning elements in V
        for i in range(factors):
            for j in range(utility_matrix_cols):
                V[i][j] = optimize_y(M, U, V, i, j, utility_matrix_rows, factors)

        UV = np.dot(U, V)
        RMSE = get_RMSE(M, UV, utility_matrix_rows, utility_matrix_cols)
        print("%.4f" % RMSE)