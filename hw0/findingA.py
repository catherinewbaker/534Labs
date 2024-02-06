# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.

import math
import numpy as np

# Given vectors
vOne = np.array([(math.sqrt(3) / 2.0), 0.5, 0])
vTwo = np.array([0, 0.5, (math.sqrt(3) / 2.0)])
vHatOne = np.array([(1.0 / math.sqrt(2)), 0, (-1.0 / math.sqrt(2))])
vHatTwo = np.array([(-1.0 / math.sqrt(2)), (1.0 / math.sqrt(2)), 0])

# Arrange vectors in matrix form
v = np.column_stack((vOne, vTwo))
vHat = np.column_stack((vHatOne, vHatTwo))

# Solve for matrix A using the pseudo-inverse
a = np.dot(vHat, np.linalg.pinv(v))

# Display matrix A
print("Matrix A:")
print(a)
