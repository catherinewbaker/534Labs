# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
# I have completed this homework without collaborating with any classmates
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Load the dataset
data = load_breast_cancer() # from sklearn's library
X = data.data # X features
y = data.target # y features (0 (benign) or 1)
numWeakLearners = range(1, 101) # range of weak learners to test M in {1, 2, 3, ..., 100}

# Storing the mean accuracy for each node depth
learnRates = [0.4, 1.0] # comparing rate = 0.4 with default 1.0
meanAccuracies = {rate: [] for rate in learnRates} # store mean accuracies for each learning rate

# Loop over range of weak learners
for M in numWeakLearners:
    # AdaBoost with 5-node trees and v = 0.4
    newBoost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=M, learning_rate=0.4) # initial model setup
    meanAccuracies[learnRates[0]].append(np.mean(cross_val_score(newBoost, X, y, cv=5, n_jobs=-1))) # append the mean of the cross validation scores

    # AdaBoost with 5-node trees and v = 1.0
    defaultBoost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=M, learning_rate=1.0) # initial model setup
    meanAccuracies[learnRates[1]].append(np.mean(cross_val_score(defaultBoost, X, y, cv=5, n_jobs=-1))) # append the mean of the cross validation scores

# Plotting the accuracies as a function of the number of weak learners
plt.figure(figsize=(10, 6))
plt.plot(numWeakLearners, meanAccuracies[learnRates[0]], label='Custom LR (0.4)')
plt.plot(numWeakLearners, meanAccuracies[learnRates[1]], label='Default LR (1.0)')

# misc plot settings
plt.xlabel('Number of Weak Learners')
plt.ylabel('Mean Classification Accuracy')
plt.title('AdaBoost Classification Accuracy with 5-Node Trees')
plt.legend()
plt.show()
