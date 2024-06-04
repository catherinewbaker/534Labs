# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
# I have completed this homework without collaborating with any classmates
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load the data
data = load_breast_cancer() # from sklearn's library
X = data.data # X features
y = data.target # y features (0 (benign) or 1)

# Setup initial vairables
p = X.shape[1]  # Number of features
m = int(np.sqrt(p)) # m = sqrt(p)
nMinValues = [1, 50] # 1 is no control, 50 is high control
meanAccuracies = {nMin: [] for nMin in nMinValues} # store mean accuracies for 1 m
treeRange = range(10, 101, 10) # range of number of trees in the forest to evaluate

# Perform 5-fold cross-validation for each nMin setting
for tree in treeRange:
    for nMin in nMinValues:
        model = RandomForestClassifier(random_state=42, n_estimators=tree, min_samples_leaf=nMin, max_features=m) # isolate the model used
        scores = cross_val_score(model, X, y, cv=5) # returns an array of the scores calculated from cross validation
        meanAccuracies[nMin].append(scores.mean()) # save the mean of that score

# Plot results per nMin value over the range of trees
plt.figure(figsize=(10, 6))
for nMin in nMinValues:
    plt.plot(list(treeRange), meanAccuracies[nMin], label=f'nMin = {nMin}')

# misc plot settings
plt.xlabel('Number of Trees')
plt.ylabel('Classification Accuracy')
plt.title('Random Forest Classifier Accuracy by Number of Trees and nMin Value')
plt.legend()
plt.grid(True)
plt.show()