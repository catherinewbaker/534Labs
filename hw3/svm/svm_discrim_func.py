import numpy as np

def svm_discrim_func(X, ssvm):
    #-------------------
    # Write your code here
    # use ssvm['kernel'], X, ssvm['sv], ssvm['alpha_y], ssvm['w0] to implement the
    # discriminant function f
    #-------------------
    # Pull components from ssvm
    kernel = ssvm['kernel'] # Kernel function reference
    SV = ssvm['sv'] # Support vectors
    alpha_y = ssvm['alpha_y'] # Dual parameters for QP problem
    w0 = ssvm['w0'] # Offset of the hyperplane
    
    # Compute the kernel matrix between input X and support vectors
    K = kernel(X, SV)
    
    # Compute the decision values (f(x)) for each input
    f = np.dot(K, alpha_y) + w0
    return f
