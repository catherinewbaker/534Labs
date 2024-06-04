import numpy as np
from SVM_soft import SVM_soft
from ssvm_plot import ssvm_plot
from Kernel_linear import Kernel_linear

# Load the data
X = np.genfromtxt('/Users/catherinebaker/Desktop/MLProjects/hw3/data/HW3_svm_data_X.dat')
y = np.genfromtxt('/Users/catherinebaker/Desktop/MLProjects/hw3/data/HW3_svm_data_y.dat')
data = {'X': X, 'y': y}

C_values = [0.01, 0.1, 1, 10, 100] # List of C values to explore

# Iterate over the list of C values
for C in C_values:
    ssvm = SVM_soft(data, Kernel_linear, C) # Train SVM model for the current C value
    
    ssvm_plot(data, ssvm) # Plot the SVM result