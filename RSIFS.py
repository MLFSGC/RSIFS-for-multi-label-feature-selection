import pandas as pd
import numpy as np
import torch
import math
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from utils import *
from affinity import *
min_max_scaler = preprocessing.MinMaxScaler()
from affinity import *
def S_matrix(X, sigma=1):
    dist = cdist(X,X,'euclidean')
    rbf = np.exp(-dist)
    return rbf

eps = 2.2204e-10



def gaussian_kernel(x, y, sigma=1.0):
    distance = np.linalg.norm(x-y)
    similarity = np.exp(-distance**2 / (sigma**2))
    return similarity

def A_A(X, k,sigma=1):
    n = X.shape[0]
    A = np.zeros((n,n))
    S = S_matrix(X)
    S = torch.FloatTensor(S)
    idxnbs = S.topk(k, 1, largest=True)[1][:, :]
    idxnbs = idxnbs.detach().numpy()
    for i in range(n):
        for j in range(len(idxnbs[i])):
            A[i,idxnbs[i,j]] = gaussian_kernel(X[i], X[j], sigma)
            A[idxnbs[i, j], i] = A[i,idxnbs[i,j]]
    return A
def feature_ranking(W):
    T = (W*W).sum(1)
    idx = np.argsort(T, 0)
    return idx[::-1]

def calculate_l21_norm(X):
    return (np.sqrt(np.multiply(X, X).sum(1))).sum()

def rsifs(X,Y,T=6,alpha=0.001,beta=1,gamma=100):

    eta = 0.2
    num, dim = np.shape(X)
    num, label_num = np.shape(Y)

    F = X
    S_Y = A_A(Y,10)
    # S_Y = affiY.kernel()
    L_Y, A_Y = laplacian(S_Y)
    q=0
    for i in range(T): #进入主循环
        objvalue = []
        print(i)
        W = np.random.rand(dim, label_num)
        #print(W)
        V = np.random.rand(num, label_num)
        q=q+1
        M = S_matrix(F)
        #affiX = Affinity(F)
        # S_X =affiX.kernel()
        S_X = A_A(F,10)
        L_X,A_X = laplacian(S_X)

        obj = 0
        a=0
        while 1:
            a=a+1
            row_norms = np.linalg.norm(W, axis=1, ord=2)
            row_norms[row_norms == 0] = 1e-10
            D_diag = 1 / (2 * row_norms)
            D = np.diag(D_diag)
                #print(D.shape)
            W1 = np.multiply(W,np.true_divide(np.dot(np.dot(X.T,M),V),

                                              (np.dot(np.dot(X.T,X),W)+ gamma*np.dot(D,W))+eps))
            V1 = np.multiply(V,
                       np.true_divide(np.dot(np.dot(M.T,X),W)+beta*np.dot(M.T,Y)+alpha*np.dot(np.dot(np.dot(M.T,S_Y),M),V)+np.dot(S_X,V)
                                      ,np.dot(np.dot(M,M.T),V)+beta*np.dot(np.dot(M.T,M),V)+alpha*np.dot(np.dot(np.dot(M.T,A_Y),M),V)+np.dot(A_X,V)+eps)
                       )

            obj1 = np.trace(np.dot((np.dot(X,W1)-np.dot(M,V1)).T,(np.dot(X,W1)-np.dot(M,V1)))) + beta * np.trace(np.dot((Y-np.dot(M,V1)).T,Y-np.dot(M,V1)))+alpha*np.trace(
        np.dot(np.dot(np.dot(np.dot(V1.T,M.T),L_Y),M),V1) ) + gamma*np.trace(np.dot(np.dot(W1.T,D),W1)) + np.trace(np.dot(np.dot(V1.T,L_X),V1))
            t=0
            if np.abs(obj-obj1)/obj1< 0.01 or a>100 or t>2:
                break
            else:obj = obj1
            W=W1
            V=V1
        W2 = feature_ranking(W)
        F = X[:, W2[0:int((1-q*eta)*dim)]]
        if q*eta>=0.8:
            break
        print(objvalue)

    W = np.mean(W, axis=1)

    return W
