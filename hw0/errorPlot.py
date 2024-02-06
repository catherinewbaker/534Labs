# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.

import numpy as np
import matplotlib.pyplot as plot

# Define sample sizes and realizations
sampSizes = [10, 100, 1000, 10000]
numReals = 100

# Define mean vector and covariance matrix
mu = np.array([1, 1]) # assuming the mu and Sigma from Q6 still stands
Sigma = np.array([[1, -0.5], [-0.5, 0.5]])

# Define dictionary for error
errors = {n: [] for n in sampSizes} # tied with respective sample size

# Generate datasets and compute sample covariance
for n in sampSizes: # over all values for N
    for _ in range(numReals): # 1-100
        X = np.random.multivariate_normal(mu, Sigma, n) # create random samples from mu and Sigma distribution with n sample Sizes
        currMu = np.mean(X, axis=0) # find mean of this random sample set
        currSigma = np.cov(X, rowvar=False, bias=False) # find covariance amtrix of this random sample set
        error = np.linalg.norm(Sigma - currSigma, 'fro') # error = norm of difference between sigmas
        errors[n].append(error) # add this error to our dictionary at its given sample size

# Box plot mapping found errors
plot.boxplot([errors[n] for n in sampSizes], labels=sampSizes)
plot.title('Error in Estimating Covariance Matrix by Sample Size')
plot.xlabel('Sample Size')
plot.ylabel('Frobenius Norm of the Error')
plot.grid(True)
plot.show()
