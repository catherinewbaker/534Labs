import numpy as np
import matplotlib.pyplot as plt

# Path to the dataset (in P1_dataset folder)
dataset_path = './P1_dataset/'

# Load the dataset
XTrain = np.load(dataset_path + 'X_train.npy')
YTrain = np.load(dataset_path + 'Y_train.npy')
XTest = np.load(dataset_path + 'X_test.npy')
YTest = np.load(dataset_path + 'Y_test.npy')

# X = input features, (numSamples, numFeatures)
# y = labels/target array, (numSamples)
# rate = learning rate (5e-3 in our case)
# lamd = lamda value (1 in our case), lasso penalty weight
# iterations = # of times we run the gradient descent
# lasso regression with gradient descent
def lasso_gradient_descent_optimized(X, y, rate=5e-3, lamd=1, iterations=10000):
    numSamples, numFeatures = X.shape # get feature and sample counts from shape of input data array
    print(numSamples)
    print(numFeatures)
    beta = np.zeros(numFeatures)  # initialize model coefficients to 0
    
    # for as many iterations as dictated
    for iteration in range(iterations):
        yPred = X.dot(beta) # predicted y = x dotted with beta
        gradient = -2 * X.T.dot(y - yPred) / numSamples # Gradient of the quadratic loss
        beta = np.sign(beta - rate * gradient) * np.maximum(np.abs(beta - rate * gradient) - rate * lamd, 0) # Update beta using a vectorized form of the soft thresholding
    
    return beta # coefficient estimates

# Train the LASSO model on the training data
betaHat = lasso_gradient_descent_optimized(XTrain, YTrain)

# Plot the estimated coefficients after optimization
plt.figure(figsize=(10, 5))
plt.stem(betaHat, use_line_collection=True)
plt.title('Estimated Model Coefficients after Optimized LASSO')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.show()
