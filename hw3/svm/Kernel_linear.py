import numpy as np

def Kernel_linear(u, v):
    kval = np.dot(u, v.T)
    return kval

# Example usage
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

kval = Kernel_linear(u, v)
print(kval)
