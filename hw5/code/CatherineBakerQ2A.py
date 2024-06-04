# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
#I have completed this homework without collaborating with any classmates.
import numpy as np
import matplotlib.pyplot as plt

# Define the inequalities as functions to pass x to
def y1(x):
    return 4 - x

def y2(x):
    return x - 1

def y3(x):
    return 9 - 3*x

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))

# define length of each line segment so the decision boundary is clear
x1 = np.linspace(-1, 2.5, 400)
x2 = np.linspace(2.5, 6, 400)
x3 = np.linspace(2.5, 6, 400)

# plotting the points
points = np.array([[1, 1], [3, 3], [4, 2]])
ax.scatter(points[:, 0], points[:, 1], color='black')

# plot line for each inequality
ax.plot(x1, y1(x1), label=r'$x + y \leq 4$', color='blue')
ax.plot(x2, y2(x2), label=r'$x - y \leq 1$', color='green')
ax.plot(x3, y3(x3), label=r'$3x + y \leq 9$', color='red')

# misc plot settings
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.title('The Points and Decision Boundary for 1-NN')
plt.grid(True)
plt.show()
