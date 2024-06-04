from sklearn.model_selection import KFold
import numpy as np
from SVM_soft import SVM_soft
from Kernel_linear import Kernel_linear
from ssvm_classify import ssvm_classify

# Load the data
X = np.genfromtxt('/Users/catherinebaker/Desktop/MLProjects/hw3/data/HW3_svm_data_X.dat')
y = np.genfromtxt('/Users/catherinebaker/Desktop/MLProjects/hw3/data/HW3_svm_data_y.dat')
data = {'X': X, 'y': y}

def crossValidateSVM(data, C, nSplits=5):
    kf = KFold(n_splits=nSplits, shuffle=True, random_state=42) # Initialize KFold split for data
    
    X, y = data['X'], data['y'] # Extract X and y from data
    errors = [] # placeholder for errors from each fold

    # Perform 5-fold cross-validation
    for trainIndex, testIndex in kf.split(X): # one loop = one fold
        # Split data into training and test sets
        Xtrain, Xtest = X[trainIndex], X[testIndex]
        ytrain, ytest = y[trainIndex], y[testIndex]

        # Create data dictionaries for training and testing
        trainData = {'X': Xtrain, 'y': ytrain}
        testData = {'X': Xtest, 'y': ytest}

        # Train SVM model on the training set
        ssvm = SVM_soft(trainData, Kernel_linear, C)

        # Calculate Error
        errorCount = ssvm_classify(testData, ssvm) # Compute the number of misclassifications on the test set
        errorRate = errorCount / len(testIndex) # Compute error rate
        errors.append(errorRate)

    avgError = np.mean(errors) # Compute the average test error across all folds
    return avgError

C = 1.0  # Chose 1 since it performed well on question 3.c.ii
avgTestError = crossValidateSVM(data, C)
print(f"Average test error: {avgTestError}")