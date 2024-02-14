# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.

import numpy as np
import matplotlib.pyplot as plt

# Path to the dataset (in P1_dataset subfolder)
dataPath = './P1_dataset/'

# Load the dataset
XTrain = np.load(dataPath + 'X_train.npy')
YTrain = np.load(dataPath + 'Y_train.npy')
XTest = np.load(dataPath + 'X_test.npy')
YTest = np.load(dataPath + 'Y_test.npy')

# X = input features, (numSamples, numFeatures)
# y = labels/target array, (numSamples)
# rate = learning rate (5e-3 in our case)
# lamb = lamda value (1 in our case), lasso penalty weight
# iterations = # of times we run the gradient descent
# lasso regression with gradient descent
def lassoGradient(X, y, rate=5e-3, lamb=1, iterations=10000):
    numSamples, numFeatures = X.shape # get feature and sample counts from shape of input data array (90, 500)
    beta = np.zeros(numFeatures)  # initialize model coefficients to 0
    
    # for as many iterations as dictated (10,000)
    for iteration in range(iterations):
        yPred = X.dot(beta) # predicted y = x dotted with beta
        gradient = -2 * X.T.dot(y - yPred) / numSamples # gradient of loss
        # soft thresholding section (fairly identical to equation on pdf)
        for i in range(len(beta)):
            # Compute the updated value for the current element
            updatedBeta = beta[i] - rate * gradient[i]
            updatedLamb = rate * lamb
            # Apply soft thresholding
            if updatedBeta > updatedLamb:
                beta[i] = updatedBeta - updatedLamb
            elif updatedBeta < -updatedLamb:
                beta[i] = updatedBeta + updatedLamb
            else:
                beta[i] = 0

    return beta # coefficient estimates

# Train the lasso on train data
betaHat = lassoGradient(XTrain, YTrain)

# Plot the estimated coefficients after optimization
plt.figure(figsize=(10, 5))
plt.stem(betaHat, use_line_collection=True)
plt.title('Estimated Coefficients after Optimized LASSO')
plt.xlabel('Model Weight Index')
plt.ylabel('Model Weight Value')
plt.show()

# Predictions for LASSO on the test set
YPred = XTest.dot(betaHat)

# Calculate and print the MSE for LASSO
MSE = np.mean((YTest - YPred) ** 2)
print(f'LASSO Model Mean Square Error: {MSE}')
