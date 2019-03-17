#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
from numpy import linalg
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from svmutil import *

#X is the input data with its last column as class label.
#It is scaled by 255 
def read_and_scale_data(file):
    X = pd.read_csv(file, header=None)
    X = np.array(X)
    X_new = np.array([x for x in X if x[-1] == 5 or x[-1] == 6])
    Y = np.array([1 if x[-1]==5 else -1 for x in X_new])
    X_new = X_new/255
    X_new = np.delete(X_new, -1, axis=1)
    return X_new, Y

#Run solver
def run_solver(H, y, m):
    #Creating matrix for cvxopt solver    
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    cvxopt_solvers.options['show_progress'] = False
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
   
    return alphas

#train data on linear Kernel
def train_linear(X,y):
    m,n = X.shape
    y = y.reshape(-1,1) * 1.
    X_dash = y * X
    H = np.dot(X_dash , X_dash.T) * 1.
    
    alphas = run_solver(H, y, m)
    #Compute w,b parameters
    w = ((y * alphas).T @ X).reshape(-1,1)
    #Selecting the set of indices S corresponding to non zero parameters
    S = (alphas > w_acc).flatten()
    b = y[S] - np.dot(X[S], w)
    
    #Display results
    #print('Alphas = ',alphas[alphas > w_acc])
    #print('\t\tw = ', w.flatten())
    print('\t\tb = ', b[0][0])
    print("\t\tNo. of support vectors = ", len(alphas[alphas > w_acc]))
    return w,b

#train data on the kernel
def train_and_test_kernel(X, y, X_test, Y_test):
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)
    
    def gaussian_kernel(x, y, r=0.05):
        return np.exp(-r*linalg.norm(x-y)**2 )
    
    m,n = X.shape
    y = y.reshape(-1,1) * 1.
    
    #Creating matrix for cvxopt solver
    K = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            K[i,j] = gaussian_kernel(X[i], X[j])

    #Optimized algo to create kernel matrix via pairwise_dists 
    #from scipy.spatial.distance import pdist, squareform
    #K = squareform(pdist(X, 'euclidean'))
    #K = np.exp(-0.05* (K**2) )

    H = np.outer(y,y) * K
    alphas = run_solver(H, y, m)
    alphas = np.ravel(alphas)
  
    # Support vectors have non zero lagrange multipliers
    print("\t\tNo. of support vectors = ", len(alphas[alphas > w_acc]))
    sv = alphas > w_acc
    non_zero_indices = np.arange(len(alphas))[sv]
    sa = alphas[sv]
    sv_x = X[sv]
    y=y.flatten()
    sv_y = y[sv]
    
    # Intercept
    b = 0
    for n in range(len(sa)):
        b += sv_y[n]
        b -= np.sum(sa * sv_y * K[non_zero_indices[n],sv])
    b = b/len(sa)
    print("\t\tb = ", b)

    # test
    Z = np.zeros(len(X_test))
    for i in range(len(X_test)):
        for n in range(len(sa)):
            Z[i] += sa[n] * sv_y[n] * gaussian_kernel(X_test[i], sv_x[n])
        Z[i] += b
        
    Z = [1 if z>=0 else -1 for z in Z]
    print("\t\tTest Accuracy = ", np.sum(Z == Y_test)/len(Z))
    
#find test accuracy
def test_accuracy(X, Y, w, b):
    Z = X @ w + b[0]
    Z = [1 if z>=0 else -1 for z in Z]
    print("\t\tTest Accuracy = ", np.sum(Z == Y)/len(Z))
	
#Grab command line input
if len(sys.argv[1:]) < 4:
    print("Usage: <path_of_train_data> <path_of_test_data>  <binary_or_multi_class> <part_num>")
    sys.exit(1)
trainFile = sys.argv[1]
testFile = sys.argv[2]
multi_class = int(sys.argv[3])
part_num = sys.argv[4]

#Read the test and train data
X_train, Y_train = read_and_scale_data('train.csv')
X_test, Y_test = read_and_scale_data('test.csv')
C = 1.0  #Note the 1. to force to float type
w_acc=1e-5

#train on linear kernel
if multi_class==0 and part_num=='a':
    print("CVXOPT package")
    print("\tLinear Kernel:")
    w,b = train_linear(X_train, Y_train)
    test_accuracy(X_test, Y_test, w, b)

#train on Gaussian Kernel
if multi_class==0 and part_num=='b':
    print("CVXOPT package")
    print("\tGaussian Kernel")
    alphas = train_and_test_kernel(X_train, Y_train, X_test, Y_test)

if multi_class==0 and part_num=='c':
    print("LIBSVM package")
    #train on linear kernel
    print("\tLinear Kernel:")
    Y_train = Y_train * 1.0
    Y_test = Y_test * 1.0
    prob = svm_problem(Y_train, X_train)
    m = svm_train(prob, '-t 0 -c 1.0')
    predicted_labels, _, _ = svm_predict(Y_test, X_test, m)

    #train on Gaussian Kernel
    print("\n\tGaussian Kernel")
    n = svm_train(prob, '-t 2 -g 0.05 -c 1.0')
    predicted_labels, _, _ = svm_predict(Y_test, X_test, n)

