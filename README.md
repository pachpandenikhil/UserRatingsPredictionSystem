# User Ratings Prediction System
User ratings prediction system using collaborative filtering techniques in Python.

- Performed latent factor modeling of the utility matrix (ratings matrix) using incremental UV decomposition and Spark’s example implementation of ALS.
- Performed latent factor modeling of the utility matrix (ratings matrix) using incremental UV decomposition and Spark’s example implementation of ALS.

Core Technology: Apache Spark, Python, numpy, AWS (Amazon EC2).

# Programs
## UV Decomposition
### Algorithm
- Incremental UV decomposition where initially all elements in latent factor matrix U and V are 1’s.
- The learning starts with learning elements in U row by row and then learning elements in V column by column.
- When learning an element, it uses the latest value learned for all other elements.
- The learning process stops after a specified number of iterations, where a round of learning all elements in U and V is one iteration.
- The program outputs RMSE after each iteration to the standard output.
### Input
The program takes 5 arguments
```
> python uv.py input-matrix n m f k
```
- *input-matrix* is a utility matrix in sparse format. For example,
```
1,1,5
1,2,2
1,3,4
1,4,4
1,5,3
2,1,3
```
- *n* is the number of rows (users) of the matrix.
- *m* is the number of columns (products).
- *f* is the number of dimensions/factors in the factor model.
- *k* is the number of iterations.
### Output
After each iteration, the program outputs RMSE with 4 floating points to the standard output
```
1.0019
0.9794
0.8464
…
```
## ALS (Modified Spark's Example Implementation)
### About
Modified the parallel implementation of ALS (alternating least squares) algorithm in **Spark (version 2.1.0)**, so that it takes a utility matrix as the input, and outputs the RMSE into a file after each iteration.
### Input
The program takes 7 arguments
```
> bin/spark-submit als.py input-matrix n m f k p output-file
```
- The first 5 parameters are same as that for UV.
- *p* is the number of partitions for the input-matrix.
- *output-file* is the path to the output file.
### Output
Same as that for UV with ouput written to a file.

