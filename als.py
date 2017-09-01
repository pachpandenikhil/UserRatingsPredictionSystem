from __future__ import print_function

import sys, warnings

import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark.sql import SparkSession

LAMBDA = 0.01  # regularization
np.random.seed(42)


def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))


def update(i, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]

    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu

    return np.linalg.solve(XtX, Xty)


# Reads utility matrix from file
def read_utility_matrix(file, rows, cols):
    f = open(file)
    utility_matrix = np.zeros((rows, cols), dtype=np.float)
    for line in iter(f):
        line = line.rstrip()
        if line:
            entry = line.split(',')
            row = int(entry[0]) - 1
            col = int(entry[1]) - 1
            val = float(entry[2])
            utility_matrix[row][col] = float(val)
    return matrix(utility_matrix)


# Returns the initial U and V matrices with all elements set to 1	
def get_initial_decomposition(n, m, d):
    U = np.ones((n, d), dtype=np.float)
    V = np.ones((m, d), dtype=np.float)
    return matrix(U), matrix(V)


# Writes the output to the file line by line.
def write_output(output, file_path):
    with open(file_path, 'w') as file:
        for line in output:
            file.write("%.4f" % line + '\n')


if __name__ == "__main__":
    spark = SparkSession.builder.appName("PythonALS").getOrCreate()

    sc = spark.sparkContext

    # reading utility matrix
    utility_matrix_file = sys.argv[1]
    utility_matrix_rows = int(sys.argv[2])
    utility_matrix_cols = int(sys.argv[3])
    factors = int(sys.argv[4])
    ITERATIONS = int(sys.argv[5])
    partitions = int(sys.argv[6])
    output_file = sys.argv[7]

    M = utility_matrix_rows
    U = utility_matrix_cols
    F = factors

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    R = read_utility_matrix(utility_matrix_file, utility_matrix_rows, utility_matrix_cols)
    ms, us = get_initial_decomposition(M, U, F)

    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)

    output = []
    for i in range(ITERATIONS):
        ms = sc.parallelize(range(M), partitions).map(lambda x: update(x, usb.value, Rb.value)).collect()
        # collect() returns a list, so array ends up being
        # a 3-d array, we take the first 2 dims for the matrix
        ms = matrix(np.array(ms)[:, :, 0])
        msb = sc.broadcast(ms)

        us = sc.parallelize(range(U), partitions).map(lambda x: update(x, msb.value, Rb.value.T)).collect()
        us = matrix(np.array(us)[:, :, 0])
        usb = sc.broadcast(us)

        error = rmse(R, ms, us)
        output.append(error)

    # writing the output to file
    write_output(output, output_file)
    spark.stop()
