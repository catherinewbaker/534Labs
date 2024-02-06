# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.

# 1 Preliminaries
#  Yes

# 2 Preliminaries
#  (a) MATH 361 Mathematical Statistics I at Emory University in the Mathematics Department
#  (b) MATH 221 Linear Algebra at Emory University in the Mathematics Department
#  (c) MATH 346 Linear Optimization at Emory University in the Mathematics Department
#  (d) No except for some machine learning topics in CS 325 Artificial Intelligence and some pattern recognition in CS 571 Natural Language Processing, both in the COmputer Science Department at Emory University

# 3 Honor Code Acknowledgement
#  I hereby affirm by writing my name that I will abide by the honor code set forth in BMI534/CS534 - Catherine Baker

# 4 Linear Algebra
import math
import numpy as np

# (a)
# Establish vectors as lists with numpy
#vOne = np.array([(math.sqrt(3) / 2.0), 0.5, 0])
#vTwo = np.array([0, 0.5, (math.sqrt(3) / 2.0)])
#vHatOne = np.array([(1.0 / math.sqrt(2)), 0, (-1.0 / math.sqrt(2))])
#vHatTwo = np.array([(-1.0 / math.sqrt(2)), (1.0 / math.sqrt(2)), 0])

# Solve for A
#aOne = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
#aTwo = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

#aOne = np.linalg.solve(vOne.reshape(3, 1), vHatOne.reshape(3, 1))
#aTwo = np.linalg.solve(vTwo.reshape(3, 1), vHatTwo.reshape(3, 1))

#aOne = np.linalg.lstsq(vOne.reshape(3, 1), vHatOne, rcond=None)[0]
#aTwo = np.linalg.lstsq(vTwo.reshape(3, 1), vHatTwo, rcond=None)[0]

#print(aOne)

# Define the vectors
vOne = np.array([(math.sqrt(3) / 2.0), 0.5, 0])
vTwo = np.array([0, 0.5, (math.sqrt(3) / 2.0)])
vHatOne = np.array([(1.0 / math.sqrt(2)), 0, (-1.0 / math.sqrt(2))])
vHatTwo = np.array([(-1.0 / math.sqrt(2)), (1.0 / math.sqrt(2)), 0])

# Additional constraint for the third column of A (can be adjusted)
third_col = np.array([0, 0, 0])

# Create matrices M1 and M2
M1 = np.column_stack((vOne, vTwo, third_col))
M2 = np.column_stack((vOne, vTwo, third_col))

# Solve for the first two columns of A using least squares
A_col1, _, _, _ = np.linalg.lstsq(M1, vHatOne, rcond=None)
A_col2, _, _, _ = np.linalg.lstsq(M2, vHatTwo, rcond=None)

# Construct matrix A
A = np.column_stack((A_col1, A_col2, third_col))

print(A)



