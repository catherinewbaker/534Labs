import matplotlib.pyplot as plt
import numpy as np

def plot_data(data, ssvm):
    plt.figure(figsize=(8, 6))
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    neg = data['y'] < 0
    pos = data['y'] > 0

    plt.plot(data['X'][neg, 0], data['X'][neg, 1], 'ro', markersize=5)
    plt.plot(data['X'][pos, 0], data['X'][pos, 1], 'b+', markersize=5)
    plt.grid(True)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    C = ssvm['C']
    plt.title(f'Data Distribution | C = {C}')
    # plt.show()
