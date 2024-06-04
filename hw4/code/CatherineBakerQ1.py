# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
# I have completed this homework without collaborating with any classmates
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_breast_cancer() # from sklearn's library
X = data.data # X features
y = data.target # y features (0 (benign) or 1)

# Split train and test data 75/25
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=42)

# Random Forest classifier from sklearn
model = RandomForestClassifier(random_state=42, n_estimators=90) # should follow the book algorithm table 15.1
model.fit(xTrain, yTrain) # Train the classifier
predictions = model.predict(xTest) # Make predictions on the testing set

# Calculate the accuracy of the model
accuracy = accuracy_score(yTest, predictions)
print(f'Accuracy: {accuracy:.2f}') # pretty consistently 0.97