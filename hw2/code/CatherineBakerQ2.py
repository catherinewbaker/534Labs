# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.
import numpy as np

# Class 1 Variables
N1 = 10
mu1 = np.array([1, 1])
sigma1 = np.array([[2, -1], [-1, 1]])

# Class 2 Variables
N2 = 10
mu2 = np.array([3, 3])
sigma2 = np.array([[1, 0.5], [0.5, 1]])

# Testing sample size
NTest = 1000  # Number of testing samples for each class

# Generates data samples based on given parameters
def dataGeneration(N, mu, Sigma):
    Z = np.random.randn(N, mu.shape[0])  # Generate N standard normal samples
    L = np.linalg.cholesky(Sigma)  # Compute the Cholesky decomposition of Sigma
    X = Z.dot(L.T) + mu  # Using linear transformation X = LZ + mu to ensure correct covariance
    
    return X

# Generate samples for both classes
X1 = dataGeneration(N1, mu1, sigma1)
X2 = dataGeneration(N2, mu2, sigma2)

# Generate testing samples for both classes
X1Test = dataGeneration(NTest, mu1, sigma1)
X2Test = dataGeneration(NTest, mu2, sigma2)

# Estimate the sample means
muHat1 = np.mean(X1, axis=0)
muHat2 = np.mean(X2, axis=0)

# Estimate the sample covariances
sigmaHat1 = np.cov(X1.T)
sigmaHat2 = np.cov(X2.T)

# Save the generated data and estimated parameters to files
np.save('X1.npy', X1)
np.save('X2.npy', X2)
np.save('X1Test.npy', X1Test)
np.save('X2Test.npy', X2Test)
np.save('muHat1.npy', muHat1)
np.save('muHat2.npy', muHat2)
np.save('sigmaHat1.npy', sigmaHat1)
np.save('sigmaHat2.npy', sigmaHat2)