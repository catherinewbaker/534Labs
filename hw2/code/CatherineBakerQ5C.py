# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

# Load data
data = loadmat('data/HW2_1.mat')
X = data['X']  # 'X' is the feature matrix
y = data['y'].ravel()  # 'y' is the label vector

# Variables of interest
alpha = 0.95
lambdas = np.logspace(-4, 2, 100)  # Lambda range on a log scale

# Outer Cross Validation - will split our data into n_splits sections
outer = KFold(n_splits=5, shuffle=True, random_state=42)

testErrors = []
validationErrors = []

# starting the CV
for i, (trainValIndex, testIndex) in enumerate(outer.split(X, y)):
    # Split data into training+validation and test sets
    XTrainVal, XTest = X[trainValIndex], X[testIndex]
    yTrainVal, yTest = y[trainValIndex], y[testIndex]

    # Inner CV for hyperparameter tuning (lambda)
    inner = KFold(n_splits=5, shuffle=True, random_state=42)
    meanErrors = []
    stdErrors = []
    
    # for each lambda value
    for lamb in lambdas:
        errors = []
        for trainIndex, valIndex in inner.split(XTrainVal, yTrainVal):
            XTrain, XVal = XTrainVal[trainIndex], XTrainVal[valIndex]
            yTrain, yVal = yTrainVal[trainIndex], yTrainVal[valIndex]
            
            # Standardize features - so all features contribute equally
            scaler = StandardScaler()
            XTrainScaled = scaler.fit_transform(XTrain)
            XValScaled = scaler.transform(XVal)
            
            # Train model
            model = ElasticNet(alpha=lamb, l1_ratio=alpha, random_state=42)
            model.fit(XTrainScaled, yTrain)
            
            # Predict and calculate error
            yPredicted = model.predict(XValScaled)
            error = mean_squared_error(yVal, yPredicted)
            errors.append(error)
        
        meanErrors.append(np.mean(errors))
        stdErrors.append(np.std(errors))
    
    meanErrors = np.array(meanErrors)
    stdErrors = np.array(stdErrors)
    
    # Find lambda min and star indicies
    lambdaMinIndex = np.argmin(meanErrors) # index of smallest mean error
    possIndicies = []
    for j, val in enumerate(meanErrors + stdErrors):
        if val <= meanErrors[lambdaMinIndex] + stdErrors[lambdaMinIndex]:
            possIndicies.append(j)
    if possIndicies:
        lambdaStarIndex = np.max(possIndicies)
    else:
        lambdaStarIndex = None

    # Test Error
    # Standardize data
    scaler = StandardScaler()
    XTrainValScaled = scaler.fit_transform(XTrainVal)
    XTestScaled = scaler.transform(XTest)
    
    # Train model on all training + validation data
    optLambda = lambdas[lambdaStarIndex]
    finalModel = ElasticNet(alpha=optLambda * (1 - alpha), l1_ratio=alpha, random_state=42) # same parameters with new optimal lambda
    finalModel.fit(XTrainValScaled, yTrainVal)
    
    # Test on all test data
    yTestPredicted = finalModel.predict(XTestScaled)
    testError = mean_squared_error(yTest, yTestPredicted)
    testErrors.append(testError)
    
    # Collect validation errors for optimal lambda
    validationErrors.append(meanErrors[lambdaStarIndex])
    
# Generate box plots
plt.figure(figsize=(12, 8))
plt.boxplot([testErrors, validationErrors], labels=['Test Errors', 'Validation Errors at Optimal λ*'])
plt.title('Box Plot of Test Errors and Validation Errors at Optimal λ*')
plt.ylabel('Error')
plt.legend()
plt.show()
    