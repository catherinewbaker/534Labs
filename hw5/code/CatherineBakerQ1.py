# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate a 2D toy Dataset
numSamples = 150 # 50 points per cluster
clusters = 3  # 3 clusters each with a center
X, y = make_blobs(n_samples=numSamples, centers=clusters, random_state=None)

# Fit the KMeans model to the generated data
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
yPred = kmeans.predict(X)

# Get the cluster centroids
centers = kmeans.cluster_centers_

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=yPred, s=50, cmap='viridis', marker='o', label='Data Points')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='X', label='Centroids')

# misc plot settings
plt.title('K-Means Clustering with Toy Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()