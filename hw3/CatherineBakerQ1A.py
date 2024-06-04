# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
# I have completed this homework without collaborating with any classmates

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Load the data
mat = scipy.io.loadmat('/Users/catherinebaker/Desktop/MLProjects/hw3/data/HW3_logistic.mat')
X = mat['X'] 
y = np.squeeze(mat['Y']) # had some array issues with Y's shape

# normalize data to prevent overflow errors
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

# Given variables
epochs = 10000
learningRate = 0.1
trials = 10
numfeatures = X.shape[1]

# splitting the data into trianing and test sets with 25% in the test set
def dataSplit(X, y, testSize=0.25):
    indices = np.arange(X.shape[0]) # return evenly spaced indicies of X
    np.random.shuffle(indices) # shuffle (randomized sample assignment)

    split = int(X.shape[0] * (1 - testSize)) # find the index that splits the data into 75/25
    trainIndex, testIndex = indices[:split], indices[split:] #separate the indicies to training and test (before and after split)
    return X[trainIndex], X[testIndex], y[trainIndex], y[testIndex] # return the train and test values for each x and y

# given loss function
def logisticLoss(y, beta0, beta, X):
    z = beta0 + np.dot(X, beta) # first compute z
    return np.log(1 + np.exp(-y * z)).mean() # then compute the rest and take the mean

# given gradient function using partial derivatives and sigmoid function (had some overflow/runtime error w/o it)
def computeGradients(y, beta0, beta, X):
    z = -y * (beta0 + np.dot(X, beta))  # Compute z
    sigmoid = 1 / (1 + np.exp(-z))  # Compute the sigmoid function
    factor = -y * (sigmoid)  # Plug sigmoid into our equation
    
    gradBeta0 = factor.mean()  # Gradient w.r.t. beta0
    gradBeta = (np.dot(X.T, factor)) / len(y)  # Gradient w.r.t. beta
    return gradBeta0, gradBeta

# use gradient descent rule for updating model parameters
def updateBetas(beta0, beta, gradBeta0, gradBeta, learningRate):
    beta0 = beta0 - (learningRate * gradBeta0) # subtract the learning rate * gradient for each
    beta = beta - (learningRate * gradBeta)
    return beta0, beta

# Store errors for plotting
trainErrors = np.zeros((trials, epochs)) # train error per trial and epoch
testErrors = np.zeros((trials, epochs)) # test error per trial and epoch

# run experiments for as many trials (10)
for trial in range(trials):
    Xtrain, Xtest, yTrain, yTest = dataSplit(X, y) # split the dataset
    # Initialize the parameters
    beta0 = np.random.randn() # random initial value from normal distribution
    beta = np.random.randn(numfeatures) # array of random initial values from normal distribution
    
    # each epoch will calculate the gradient, update the parameters, and compute error 
    for epoch in range(epochs):
        gradBeta0, gradBeta = computeGradients(yTrain, beta0, beta, Xtrain) # compute gradients
        beta0, beta = updateBetas(beta0, beta, gradBeta0, gradBeta, learningRate) # update parameters
        
        # Compute training and testing errors
        trainErrors[trial, epoch] = logisticLoss(yTrain, beta0, beta, Xtrain)
        testErrors[trial, epoch] = logisticLoss(yTest, beta0, beta, Xtest)

# Calculate average and standard deviation of errors
avgTrainErrors = trainErrors.mean(axis=0)
stdTrainErrors = trainErrors.std(axis=0)
avgTestErrors = testErrors.mean(axis=0)
stdTestErrors = testErrors.std(axis=0)

# Plotting
epochsRange = range(1, epochs + 1)
plt.fill_between(epochsRange, avgTrainErrors - stdTrainErrors, avgTrainErrors + stdTrainErrors, alpha=0.1, color="r")
plt.fill_between(epochsRange, avgTestErrors - stdTestErrors, avgTestErrors + stdTestErrors, alpha=0.1, color="g")
plt.plot(epochsRange, avgTrainErrors, 'r-', label='Training Error')
plt.plot(epochsRange, avgTestErrors, 'g-', label='Testing Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.show()