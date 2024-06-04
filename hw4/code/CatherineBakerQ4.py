# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS LIKE CHATGPT. Catherine Baker
# I have completed this homework without collaborating with any classmates
import matplotlib.pyplot as plt

# Given data
class1 = [(0, 4), (2.5, 6.5), (3.5, 7.5), (5, 6), (0.5, 1.5)]
class2 = [(7, 6.5), (2, 4.5), (3.5, 4), (7, 1.5), (1.5, 0)]

# Unpack the points as x values and y values
x1, y1 = zip(*class1)
x2, y2 = zip(*class2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x1, y1, 'bo', label='Class 1')  # Plot Class 1 points
plt.plot(x2, y2, 'rs', label='Class 2')  # Plot Class 2 points

# Plot classifiers
plt.plot([1, 1], [0, 5.25], color='g', linestyle='--', label='x = 1')  # Vertical line for x = 1
plt.plot([1, 6], [5.25, 5.25], color='g', linestyle='--', label='y = 5.25')  # Horizontal line for y = 5.25
plt.plot([6, 6], [5.25, max(y1 + y2) + 1], color='g', linestyle='--', label='x = 6')  # Vertical line for x = 6

# misc plot settings
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of the Given Points with Classifiers')
plt.legend()
plt.grid(True)
plt.show()