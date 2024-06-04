# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.
import numpy as np
import matplotlib.pyplot as plt

# Load the data and estimated mean and covariance
# class 1
X1 = np.load('data/X1C.npy') # train data
muHat1 = np.load('data/muHat1C.npy') # estimated mean
sigmaHat1 = np.load('data/sigmaHat1C.npy') # estimated covariance matrix
X1Test = np.load('data/X1TestC.npy') # test data
# true class 1 mean and convariance
muTrue1 = np.array([1, 1])
sigmaTrue1 = np.array([[2, -1], [-1, 1]])

#class 2
X2 = np.load('data/X2C.npy') # train data
muHat2 = np.load('data/muHat2C.npy') # estimated mean
sigmaHat2 = np.load('data/sigmaHat2C.npy') # estimated covaraince matrix
X2Test = np.load('data/X2TestC.npy') # test data
# true class 2 mean and convariance
muTrue2 = np.array([3, 3])
sigmaTrue2 = np.array([[1, 0.5], [0.5, 1]])

# Prior probabilities
prior1 = 0.5 # 50% of training data is in class 1
prior2 = 0.5 # 50% of training data is in class 2

# Functions
# Part A: Quadratic discriminant function for delta_k
def delta_k(x, mu_k, sigma_k, pi_k):
    sigmaInv = np.linalg.inv(sigma_k)
    return (((-0.5) * np.log(np.linalg.det(sigma_k))) - (0.5 * (x - mu_k).T @ sigmaInv @ (x - mu_k)) + np.log(pi_k))

# Part A: Decision boundary function (for two classes) (where delta1 = delta2)
def decisionBoundary(X, Y, mu1, mu2, sigma1, sigma2, prior1, prior2):
    Z = np.zeros(X.shape)  # Initialize Z with the same shape as X and Y
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]): # for each x and y point...
            point = np.array([X[i, j], Y[i, j]])
            # calculate the deltas
            delta1 = delta_k(point, mu1, sigma1, prior1) 
            delta2 = delta_k(point, mu2, sigma2, prior2)
            Z[i, j] = delta1 - delta2 # want the difference between the QDA functions to be 0 for decision boundary
    return Z 

# Part B: Calculating the PDF
def guassianPDF(x, mu, sigma):
    n = mu.shape[0]
    sigma_det = np.linalg.det(sigma)
    sigmaInv = np.linalg.inv(sigma)

    # Equations (separted left and right half of the PDF)
    norm = 1 / ((2 * np.pi) ** (n / 2) * sigma_det ** 0.5)
    exponent = -0.5 * np.dot(np.dot((x - mu).T, sigmaInv), (x - mu))
    return norm * np.exp(exponent)

# Part B: For plotting the level curves
def levelCurves(X, Y, mu1, sigma1, mu2, sigma2):
    Z1 = np.zeros(X.shape)
    Z2 = np.zeros(X.shape)
    # find the pdf values of each point for each class
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            Z1[i, j] = guassianPDF(point, mu1, sigma1) # Z1 = pdf values for all points being in class 1
            Z2[i, j] = guassianPDF(point, mu2, sigma2) # Z2 = pdf values for all points being in class 2
    return Z1 - Z2

# Part B: ^g_i -> predicting class by QDA value (find k that maximizes delta_k)
def predictClass(x, mu1, mu2, sigma1, sigma2, prior1, prior2):
    delta1 = delta_k(x, mu1, sigma1, prior1)
    delta2 = delta_k(x, mu2, sigma2, prior2)
    return 1 if delta1 > delta2 else 2

# Part B: Predicting Error from given equation
def errorCalculation(X_test, trueClass, mu1, mu2, sigma1, sigma2, prior1, prior2):
    predictions = np.zeros(X_test.shape[0])
    totalError = 0
    # for the test values...
    for i, x in enumerate(X_test):
        predictions[i] = predictClass(x, mu1, mu2, sigma1, sigma2, prior1, prior2) # calculate the predicted class for each x
        totalError += np.absolute(predictions[i] - trueClass) # add the difference of that value and the true class to error 
    totalError /= len(predictions) # average results (1 / N)
    return predictions, totalError

# Part B: predict error
predictions1, error1 = errorCalculation(X1Test, 1, muHat1, muHat2, sigmaHat1, sigmaHat2, prior1, prior2)
predictions2, error2 = errorCalculation(X2Test, 2, muHat1, muHat2, sigmaHat1, sigmaHat2, prior1, prior2)
error = (error1 * len(X1Test) + error2 * len(X2Test)) / (len(X1Test) + len(X2Test)) # total error weighted by testing data per class

# Plotting
# Part A: Plot the training samples as a scatter plot
plt.scatter(X1[:, 0], X1[:, 1], label='Class 1 Train', alpha=1) # plotting class 1
plt.scatter(X2[:, 0], X2[:, 1], label='Class 2 Train', alpha=1) # plotting class 2

# Part A: get a range of x and y for a grid over the plot
xRange = np.linspace(min(np.min(X1[:,0]), np.min(X2[:,0])) - 1, max(np.max(X1[:,0]), np.max(X2[:,0])) + 1, 100) # find the minimum and maximum values for x for each class and create 100 intervals between them
yRange = np.linspace(min(np.min(X1[:,1]), np.min(X2[:,1])) - 1, max(np.max(X1[:,1]), np.max(X2[:,1])) + 1, 100) # find the minimum and maximum values for y for each class and create 100 intervals between them
X, Y = np.meshgrid(xRange, yRange) # create a grid on the plot with the above ranges, returns arrays of x and y values

# Part A: Finding decision boundary (for each X and Y) as a vector
Z = decisionBoundary(X, Y, muHat1, muHat2, sigmaHat1, sigmaHat2, prior1, prior2)

# Part B: Plot level curves where densities are equal
ZEstimated = levelCurves(X, Y, muHat1, sigmaHat1, muHat2, sigmaHat2)
ZTrueDiff = levelCurves(X, Y, muTrue1, sigmaTrue1, muTrue2, sigmaTrue2)
plt.contour(X, Y, ZTrueDiff, levels=[0], colors='orange', alpha=0.5)  # for true mean and covariance (C_1)
plt.contour(X, Y, ZEstimated, levels=[0], colors='blue', alpha=0.5)  # for estimated mean and covariance (C_2)

# Part B: Plotting the testing samples and decision boundary
#plt.scatter(X1Test[:, 0], X1Test[:, 1], label='Class 1 Test', alpha=0.1)
#plt.scatter(X2Test[:, 0], X2Test[:, 1], label='Class 2 Test', alpha=0.1)

# Part A and B: Level curves for the density estimates based on decision boundary
plt.contour(X, Y, Z, levels=[0], colors='k') # levels=[0] -> where the decision boundary (Z vector) = 0

# misc plot settings and labels
plt.title(f'QDA Samples with Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()