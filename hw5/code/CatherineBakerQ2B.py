# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.
import numpy as np
import matplotlib.pyplot as plt

# Define the inequalities as functions to pass x to
def y1(x):
    return 3 - (0.5*x)

def y2(x):
    return 0.75 + (0.5*x)

def y3(x):
    return 5.25 - (1.5*x)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))

# define length of each line segment so the decision boundary is clear
x1 = np.linspace(-1, 2.25, 400)
x2 = np.linspace(2.25, 6, 400)
x3 = np.linspace(2.25, 6, 400)

# plotting the points
points = np.array([[1, 1], [3, 3], [4, 2]])
ax.scatter(points[:, 0], points[:, 1], color='black')

# plot line for each inequality
ax.plot(x1, y1(x1), label=r'$x + 2y \leq 6$', color='blue')
ax.plot(x2, y2(x2), label=r'$2x - 4y \leq -3', color='green')
ax.plot(x3, y3(x3), label=r'$6x + 4y \leq 21$', color='red')

# misc plot settings
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.title('The Points and Decision Boundary for 1-NN')
plt.grid(True)
plt.show()
