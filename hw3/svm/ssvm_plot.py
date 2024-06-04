import numpy as np
import matplotlib.pyplot as plt

from svm_discrim_func import svm_discrim_func
from plot_data import plot_data

# You need to implement the following functions:
# svm_discrim_func: to compute the discriminant function value for SVM

def ssvm_plot(data, ssvm):
    # Plot the data points
    plot_data(data, ssvm)  # You need to implement this function based on your data structure
    
    # Plot support vectors
    plt.scatter(ssvm['sv'][:, 0], ssvm['sv'][:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    
    # Create a grid to evaluate the model
    m = 50  # grid points for contour
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Create grid to plot
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], m+1),
                         np.linspace(ylim[0], ylim[1], m+1))
    
    # Flatten the grid to pass to svm_discrim_func
    Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Get decision values for the grid
    Z = svm_discrim_func(Xgrid, ssvm).reshape(xx.shape)  # svm_discrim_func must be implemented
    
    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    plt.show()

