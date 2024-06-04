import numpy as np
from cvxopt import matrix, solvers

def SVM_soft(data, kernel_function, C):
    y = data['y']
    y = y.astype('float')

    X = data['X']
    n = len(y)  # y is n x 1

    # Initialize A matrix
    A = kernel_function(X, X) * np.outer(y, y)
    A = A.astype('float') 

    #-------------------------
    # Solve dual problem...
    #-------------------------

    # Define linear term in the objective function
    q = -np.ones(n)
    
    # Equality constraints for Sigma_{i=1}^n
    A_eq = y.reshape(1, -1).astype('float')
    b_eq = np.zeros(1)
    
    # Inequality constraints
    G = np.vstack((-np.eye(n), np.eye(n)))
    h = np.hstack((np.zeros(n), np.ones(n) * C))
    
    # Convert to cvxopt format
    P = matrix(A)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    Y = matrix(A_eq)
    b = matrix(b_eq)
    
    # Solve QP problem
    sol = solvers.qp(P, q, G, h, Y, b)
    alpha = np.array(sol['x']).flatten()

    #-------------------------
    # Back to given code...
    #-------------------------

    # Select support vectors
    svm_eps = np.max(alpha) * 1e-8
    S = np.where(alpha > svm_eps)[0]
    NS = len(S)
    beta = alpha[S] * y[S]
    XS = X[S]

    # Calculate w0 offset parameter
    index = np.where(alpha[S] < C)[0]
    y_w_phi = np.dot(A[S[index]][:, S], alpha[S])  # sum_j y_i K(x_i, x_j) y_j
    w0_est = y[S[index]] - y[S[index]] * y_w_phi  # y_i - sum_j K(x_i, x_j) y_j
    w0 = np.median(w0_est)  # median of slightly varying estimates of w0

    # Computing the margin gamma
    theta_norm = np.sqrt(np.dot(alpha[S].T, np.dot(A[S][:, S], alpha[S])))
    gamma = 1 / theta_norm

    ssvm = {'kernel': kernel_function,
            'num_S': NS,
            'w0': w0,
            'alpha_y': beta,
            'sv': XS,
            'C': C,
            'gamma': gamma}

    return ssvm

# Example usage
# data = {'X': np.array([[1, 2], [2, 3], [3, 3]]), 'y': np.array([1, -1, 1])}
# C = 1.0
# ssvm = SVM_soft(data, linear_kernel, C)
# print(ssvm)
