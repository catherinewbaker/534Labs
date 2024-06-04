import numpy as np
import matplotlib.pyplot as plt
from perceptron.plot_class_data import plot_class_data

# Load the data
X_1 = np.genfromtxt('/Users/catherinebaker/Desktop/MLProjects/hw3/data/HW3_perceptron_data1_X.dat')
y_1 = np.genfromtxt('/Users/catherinebaker/Desktop/MLProjects/hw3/data/HW3_perceptron_data1_y.dat')
X_2 = np.genfromtxt('/Users/catherinebaker/Desktop/MLProjects/hw3/data/HW3_perceptron_data2_X.dat')
y_2 = np.genfromtxt('/Users/catherinebaker/Desktop/MLProjects/hw3/data/HW3_perceptron_data2_y.dat')

def PRCPTtrn(X, y, maxIters=1000):
    w = np.zeros(X.shape[1] + 1)  # initialize weight vector with an extra dimension for the bias as zeroes
    k = 0  # initialize iteration counter
    
    for iteration in range(maxIters):
        misclassified = 0
        for i, x in enumerate(X):
            xBias = np.insert(x, 0, 1) # Insert bias term into feature vector at the beginning
            a = np.dot(w, xBias)
            if y[i] * a <= 0:
                # Update weights for misclassified examples
                w += y[i] * xBias
                misclassified += 1
        k += 1 # +1 for every interation that takes us towards convergence
        if misclassified == 0:
            break  # Stop if all examples are correctly classified
    return w, k

def PRCPTtst(w, X, y):
    errCount = 0
    for i, x in enumerate(X):
        xBias = np.insert(x, 0, 1)  # Insert bias term
        prediction = np.sign(np.dot(w, xBias)) # make prediction
        if prediction != y[i]:
            errCount += 1 
    err = errCount / len(X)  # Compute error rate
    return err

w1, k1 = PRCPTtrn(X_1, y_1) # find w1
err1 = PRCPTtst(w1, X_1, y_1)
plot_class_data(X_1, y_1, w1[1:])

w2, k2 = PRCPTtrn(X_2, y_2)  # find w2
err2 = PRCPTtst(w2, X_2, y_2)
plot_class_data(X_2, y_2, w2[1:])

RSquared1 = np.max(np.linalg.norm(X_1, axis=1))**2
distances1 = np.abs(np.dot(X_1, w1[1:]) + w1[0]) / np.linalg.norm(w1[1:])
gammaSquared1 = np.min(distances1)**2

RSquared2 = np.max(np.linalg.norm(X_2, axis=1))**2
distances2 = np.abs(np.dot(X_2, w2[1:]) + w2[0]) / np.linalg.norm(w2[1:])
gammaSquared2 = np.min(distances2)**2

ratio1 = RSquared1 / gammaSquared1
print("\n\n\nData 1:")
print(f"R^2: {RSquared1}")
print(f"y^2: {gammaSquared1}")
print(f"Ratio of R^2 to y^2: {ratio1}")
print(f"Training completed in {k1} iterations.")
print(f"Testing error: {err1}")

ratio2 = RSquared2 / gammaSquared2
print("\n\n\nData 2:")
print(f"R^2: {RSquared2}")
print(f"y^2: {gammaSquared2}")
print(f"Ratio of R^2 to y^2: {ratio2}")
print(f"Training completed in {k2} iterations.")
print(f"Testing error: {err2}")
