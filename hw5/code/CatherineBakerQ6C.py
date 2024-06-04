# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.
import numpy as np

# Define our cov matrix
covarianceX = np.array([
    [2.8, 2.8, 2.8, 0, 0],
    [2.8, 2.8, 2.8, 0, 0],
    [2.8, 2.8, 2.8, 0, 0],
    [0, 0, 0, 1.2, 1.2],
    [0, 0, 0, 1.2, 1.2],
])

# Find eigen-values and vectors and print them
eigenvalues, eigenvectors = np.linalg.eig(covarianceX)

print(eigenvalues)
print("\n\n")
print(eigenvectors)