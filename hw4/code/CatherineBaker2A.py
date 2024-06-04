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
nodeDepth = [1, 3, 5] # 1-node (stump), 3-node, and 5-node trees 
meanAccuracies = {node: [] for node in nodeDepth} # store mean accuracies for each node

# Loop over range of weak learners
for M in numWeakLearners:
    for n in nodeDepth:
        # AdaBoost with stumps (max_depth=1)
        oneBoost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=n), n_estimators=M) # setup initial model
        oneCross = cross_val_score(oneBoost, X, y, cv=5, n_jobs=-1) # returns an array of scores from the cross balidation
        meanAccuracies[nodeDepth[int((n  - 1) / 2)]].append(np.mean(oneCross)) # save the mean of that score
  
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(numWeakLearners, meanAccuracies[nodeDepth[0]], label='Stumps')
plt.plot(numWeakLearners, meanAccuracies[nodeDepth[1]], label='3-Node Trees')
plt.plot(numWeakLearners, meanAccuracies[nodeDepth[2]], label='5-Node Trees')

# misc plot settings
plt.xlabel('Number of Weak Learners')
plt.ylabel('Mean Classification Accuracy')
plt.title('AdaBoost Classification Accuracy vs. Number of Weak Learners')
plt.legend()
plt.show()