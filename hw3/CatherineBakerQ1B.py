# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
# I have completed this homework without collaborating with any classmates
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import scipy.io

# Load the data
mat = scipy.io.loadmat('/Users/catherinebaker/Desktop/MLProjects/hw3/data/HW3_logistic.mat')
X = mat['X'] 
y = np.squeeze(mat['Y'])

# normalize data to prevent overflow errors
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std
 
# Variables
iterations = 1000 # 1000?
testErrors = []
trainingErrors = []

# splitting the data into trianing and test sets with 25% in the test set
def dataSplit(X, y, testSize=0.25):
    indices = np.arange(X.shape[0]) # return evenly spaced indicies of X
    np.random.shuffle(indices) # shuffle (randomized sample assignment)

    split = int(X.shape[0] * (1 - testSize)) # find the index that splits the data into 75/25
    trainIndex, testIndex = indices[:split], indices[split:] #separate the indicies to training and test (before and after split)
    return X[trainIndex], X[testIndex], y[trainIndex], y[testIndex] # return the train and test values for each x and y

for iter in range(iterations):
    # split data into training and testing sets
    Xtrain, Xtest, yTrain, yTest = dataSplit(X, y)
    
    # Create and train the logistic regression model
    model = LogisticRegression(max_iter=1000, solver='lbfgs') # 1000?
    model.fit(Xtrain, yTrain)
    
    # predict probabilities on test set
    predYProbsTest = model.predict_proba(Xtest)
    predYProbsTrain = model.predict_proba(Xtrain)
    
    # Calculate the log loss - using sklearn function since we only have accses to the data
    testError = log_loss(yTest, predYProbsTest)
    trainError = log_loss(yTrain, predYProbsTrain)

    testErrors.append(testError)
    trainingErrors.append(trainError)

# Calculate mean testing error and standard deviation
meanTestError = np.mean(testErrors)
stdTestError = np.std(testErrors)

meanTrainError = np.mean(trainingErrors)
stdTrainError = np.std(trainingErrors) 

# Plotting
plt.figure(figsize=(10, 10))
plt.plot(range(iterations), testErrors, label='Test Error', color='blue')
plt.plot(range(iterations), trainingErrors, label='Training Error', color='red')
plt.fill_between(range(iterations), meanTestError - stdTestError, meanTestError + stdTestError, color='lightblue', alpha=0.5)
plt.fill_between(range(iterations), meanTrainError - stdTrainError, meanTrainError + stdTrainError, color='pink', alpha=0.5)
plt.xlabel('Iteration')
plt.ylabel('Log Loss Error')
plt.title('Logistic Regression Training and Test Error Across Iterations')
plt.ylim(0, 1)
plt.legend()
plt.show()