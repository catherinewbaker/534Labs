import matplotlib.pyplot as plt
import numpy as np

def plot_class_data(X, y, beta):
    """
    Plot data and decision boundary.
    X is an n x 2 matrix, y is an n x 1 column vector.
    beta is a 2 x 1 vector returned by the Perceptron training algorithm.
    y=+1 points are plotted as red dots, while y=-1 points are plotted as
    blue 'x's. The decision boundary is plotted as a black line.
    """
    # Plotting the data points
    y = y.reshape((-1,))
    # print (y == 1)
    # print (X[y == 1, 0].shape)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', marker='.')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', marker='x')

    # Calculating and plotting the decision boundary
    xmax = np.max(X[:, 0])
    ymax = -xmax * beta[0] / beta[1]
    plt.plot([0, xmax], [0, ymax], 'k', linewidth=3)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Classification Data with Decision Boundary')
    plt.show()

# Example usage:
# Assuming X, y, and beta are already defined
# plot_class_data(X, y, beta)
