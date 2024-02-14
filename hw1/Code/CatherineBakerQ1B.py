# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.

import numpy as np
import matplotlib.pyplot as plt

# Path to the dataset (in P1_dataset subfolder)
dataset_path = './P1_dataset/'

# Load the dataset
XTrain = np.load(dataset_path + 'X_train.npy')
YTrain = np.load(dataset_path + 'Y_train.npy')
XTest = np.load(dataset_path + 'X_test.npy')
YTest = np.load(dataset_path + 'Y_test.npy')

# Function to fit a Least Squares model to the data
def fitLeastSquares(X, y):
    XUpdated = np.hstack([np.ones((X.shape[0], 1)), X])  # add 1s column for intercept
    beta = np.linalg.inv(XUpdated.T.dot(XUpdated)).dot(XUpdated.T).dot(y) # The Normal Equation for updating beta (coefficients)
    return beta

# Fit the Least Squares model
betaLS = fitLeastSquares(XTrain, YTrain)

# Plot the estimated coefficients after fitting the model
plt.figure(figsize=(10, 5))
plt.stem(betaLS[1:], use_line_collection=True)  # Skip the intercept term for plotting
plt.title('Estimated Coefficients after Least Squares')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.show()

# Predictions for Least Squares on the test set
XTestUpdated = np.hstack([np.ones((XTest.shape[0], 1)), XTest])  # Add intercept term
YPred = XTestUpdated.dot(betaLS)

# Calculate and print the MSE for OLS
MSE = np.mean((YTest - YPred) ** 2)
print(f'Ordinary Least Squares Model Mean Square Error: {MSE}')
