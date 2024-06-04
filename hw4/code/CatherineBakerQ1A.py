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

# Setup p and m
p = X.shape[1]  # Number of features
mValues = [1, np.sqrt(p), p] # Given m values
treeRange = range(10, 101, 10) # range of number of trees in the forest to evaluate
meanAccuracies = {m: [] for m in mValues} # stores mean accuracies for each combination of values by key m

# Split train and test data 75/25
#xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=42)

# Perform 5-fold cross-validation for each combination of tree numbers and m
for trees in treeRange:
    for m in mValues:
        model = RandomForestClassifier(random_state=42, n_estimators=trees, max_features=int(m)) # isolate the model used
        scores = cross_val_score(model, X, y, cv=5) # returns an array of the scores calculated from cross validation
        meanAccuracies[m].append(scores.mean()) # save the mean of that score

# Plot the results for each (tree number, accuracy) value and connect points with identical m values into lines
for m, accuracies in meanAccuracies.items():
    plt.plot(treeRange, accuracies, label=f'm = {int(m)}')

# misc plot settings
plt.xlabel('Number of Trees')
plt.ylabel('Classification Accuracy')
plt.legend()
plt.title('Accuracy vs Number of Trees for different values of m')
plt.show()