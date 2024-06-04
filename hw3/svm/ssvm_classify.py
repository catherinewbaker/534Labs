import numpy as np
from svm_discrim_func import svm_discrim_func

def ssvm_classify(data, ssvm):
    #-------------------
    # Write your code here
    # use data['X'] and  svm_discrim_func to produce prediction signs
    # and to Count the number of mismatches between predicted labels and actual labels

    #-------------------
    
    X, y = data['X'], data['y']
    
    predictedValues = svm_discrim_func(X, ssvm) # predict labels
    predictedLabels = np.sign(predictedValues) # sign of the discriminant function values are the predicted labels
   
    errors = (predictedLabels != y).sum()  # Count the number of mismatches between predicted labels and actual labels
    
    return errors

# Example usage
# Assuming data is a dictionary with 'X' and 'y' as keys, and ssvm is your trained SVM model
# data = {'X': X, 'y': y}
# ssvm = your_ssvm_model
# error_count = svm_classify(data, ssvm)
# print("Number of misclassifications:", error_count)
