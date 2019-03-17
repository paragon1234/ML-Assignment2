#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
from numpy import linalg
from numpy import cumsum
import collections
from collections import Counter
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from svmutil import *

#X is the input data with its last column as class label.
#It is scaled by 255 
def read_and_scale_data(file):
    X = pd.read_csv(file, header=None)
    X = np.array(X)
    X = X[X[:,-1].argsort()]
    Y = X[:,-1]
    X = np.delete(X, -1, axis=1)
    X = X/255

    return X,Y

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

#train data on the kernel
def train_and_test_kernel(X, y, X_test, X_train, cls1, cls2):
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)
    
    def gaussian_kernel(x, y, r=0.05):
        return np.exp(-r*linalg.norm(x-y)**2 )
    
    m,n = X.shape
    y = y.reshape(-1,1) * 1.
    
    #Optimized algo to create kernel matrix via pairwise_dists 
    from scipy.spatial.distance import pdist, squareform
    K = squareform(pdist(X, 'euclidean'))
    K = np.exp(-0.05* (K**2) )

    H = np.outer(y,y) * K
    alphas = run_solver(H, y, m)
    alphas = np.ravel(alphas)
  
    # Support vectors have non zero lagrange multipliers
    #print("\t\tNo. of support vectors = ", len(alphas[alphas > w_acc]))
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
    #print("\t\tb = ", b)

    # training data prediction
    #Z_train = np.zeros(len(X_train))
    #for i in range(len(X_train)):
    #    for n in range(len(sa)):
    #        Z_train[i] += sa[n] * sv_y[n] * gaussian_kernel(X_train[i], sv_x[n])
    #    Z_train[i] += b
    #Z_train = [cls1 if z>=0 else cls2 for z in Z_train]

    # test data prediction
    Z = np.zeros(len(X_test))
    for i in range(len(X_test)):
        for n in range(len(sa)):
            Z[i] += sa[n] * sv_y[n] * gaussian_kernel(X_test[i], sv_x[n])
        Z[i] += b
    Z = [cls1 if z>=0 else cls2 for z in Z]

    return Z#, Z_train
    
def get_count_Array(y):
    ind = [[] for i in range(11)]
    ind[0] = -1
    sum = 0
    for i in range(10):
        ind[i+1] = np.count_nonzero(y == i)-1 + sum
        sum += np.count_nonzero(y == i)
    
    #print(ind)
    return ind

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
train_ind = get_count_Array(Y_train)
X_test, Y_test = read_and_scale_data('test.csv')
test_ind = get_count_Array(Y_test)
C = 1.0  #Note the 1. to force to float type
w_acc=1e-5
no_class = 10
class_combi = 45

if multi_class==1 and part_num=='a':
    print("CVXOPT package")
    #train on Gaussian Kernel
    print("\tGaussian Kernel")

    count = 0
    Y_pred = [[] for i in range(class_combi)]
    Y_train_pred = [[] for i in range(class_combi)]
    for i in range(no_class):
        for j in range(i+1,no_class,1):
            size1 = -train_ind[i] + train_ind[i+1]
            size2 = -train_ind[j] + train_ind[j+1]
            X = np.concatenate((X_train[train_ind[i]+1:train_ind[i+1]+1,:], X_train[train_ind[j]+1:train_ind[j+1]+1,:]))
            m = np.empty((size2)); m.fill(-1)
            Y = np.concatenate((np.ones(size1), m))
            #Y_pred[count], Y_train_pred[count] = train_and_test_kernel(X, Y, X_test, X_train, i, j)
            Y_pred[count] = train_and_test_kernel(X, Y, X_test, X_train, i, j)
            count += 1

# Y_train_pred = np.array(Y_train_pred)
# Y_train_pred_maj = []
# for i in range(len(Y_train)):
    # count = Counter(Y_train_pred[:, i])
    # Y_train_pred_maj.append(count.most_common(1)[0][0])
# print("\t\tTrain Accuracy = ", np.sum(Y_train_pred_maj == Y_train)/len(Y_train))

    Y_pred = np.array(Y_pred)
    Y_pred_maj = []
    for i in range(len(Y_test)):
        count = Counter(Y_pred[:, i])
        Y_pred_maj.append(count.most_common(1)[0][0])
    print("\t\tTest Accuracy = ", np.sum(Y_pred_maj == Y_test)/len(Y_test))

if multi_class==1 and part_num=='b':
    print("LIBSVM package")
    print("\tGaussian Kernel")
    #train on linear kernel
    Y_train = Y_train * 1.0
    Y_test = Y_test * 1.0
    prob = svm_problem(Y_train, X_train)
    n = svm_train(prob, '-t 2 -g 0.05 -c 1.0')
    print("\t\tTraining Accuracy")
    predicted_labels, _, _ = svm_predict(Y_train, X_train, n)
    print("\t\tTesting Accuracy")
    predicted_labels, _, _ = svm_predict(Y_test, X_test, n)