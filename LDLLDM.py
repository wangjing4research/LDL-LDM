import os
import pickle
import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix 
from ldl_metrics import score
from scipy.special import softmax
from scipy.linalg import solve
from sklearn.cluster import KMeans
import time

#only needed if optimizing using pymanopt
#default optimizing by l-bfgs
'''
from pymanopt.manifolds import Euclidean
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
'''

eps = 1e-15

#this manifold learning colde is from sklearn
#https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/manifold/_locally_linear.py#21

def barycenter_weights(X, Z, reg=1e-3):

    n_samples, n_neighbors = X.shape[0], Z.shape[1]
    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    # this might raise a LinalgError if G is singular and has trace zero
    for i, A in enumerate(Z.transpose(0, 2, 1)):
        C = A.T - X[i]  # broadcasting
        G = np.dot(C, C.T)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[::Z.shape[1] + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B


def barycenter_kneighbors_graph(X, n_neighbors = None, reg=1e-3):
    
    #considering all neighbors
    if n_neighbors is None:
        n_neighbors = X.shape[0] - 1
    
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    X = knn._fit_X
    n_samples = knn.n_samples_fit_
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X[ind], reg=reg)
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr),
                      shape=(n_samples, n_samples)).toarray()



def weighted_ME(Y_hat, W):  
    weighted_Y_hat = Y_hat * W
    grad = np.sum(-weighted_Y_hat, 1).reshape((-1, 1)) * Y_hat + weighted_Y_hat
    return grad

def weighted_log_ME(Y_hat, W):
        return W - W.sum(1).reshape((-1, 1)) * Y_hat
    

def KL(y, y_hat, J = None):
    y = np.clip(y, eps, 1)
    y_hat = np.clip(y_hat, eps, 1)
    
    if J is None:
        loss = -1 * np.sum(y * np.log(y_hat))
        grad = y_hat - y
    else:
        loss = -1 * np.sum(J * y * np.log(y_hat))
        grad = J * (y_hat - y)
    
    return loss, grad


def append_intercept(X):
        return np.hstack((X, np.ones(np.size(X, 0)).reshape(-1, 1)))
    

#label distribution matrix is not missing
class LDLLDM_Full: 

    #each cluster
    class Cluster:
        def __init__(self, X, l, Y):
            
            self.X = X
            self.l = l
            self.I = np.eye(Y.shape[1])
            self.LDM(Y)
          
        #learn label distribution manifold
        def LDM(self, Y):
            self.Z = barycenter_kneighbors_graph(Y.T).T
            self.I_Z = self.I - self.Z 
            self.L = np.dot(self.I_Z, self.I_Z.T)
        
        
        def LDL(self, Y_hat):
            if self.l == 0:
                return 0, 0
            
            loss = (np.dot(Y_hat, self.I_Z) ** 2).sum()
            grad = np.dot(self.X.T, weighted_ME(Y_hat, 2 * np.dot(Y_hat, self.L)))
            
            return self.l * loss, self.l * grad
            
    
    def __init__(self, X, Y, l1, l2, l3, g = 0, clu_labels = None):
        
        self.X = append_intercept(X)        
        self.Y = Y
        
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        
        self.g = g
        
        self.n_examples, self.n_features, = self.X.shape
        self.n_outputs = self.Y.shape[1]
        
        #conduct K-means
        if clu_labels is None:
            kmeans = KMeans(n_clusters=g).fit(Y)
            clu_labels = kmeans.predict(Y)
            
        self.__init_clusters(clu_labels)

        
    def __init_clusters(self, clu_labels):
        self.clusters = []
        self.inds = []
        
        #global label distribution manifold 
        clu = self.Cluster(self.X, self.l2, self.Y)
        self.clusters.append(clu)
        self.inds.append(np.asarray(np.ones(self.n_examples), dtype=np.bool))
        
        if self.g > 1: 
            for i in range(self.g):
                ind = (clu_labels == i)
                X_i = self.X[ind]
                Y_i = self.Y[ind]
                clu = self.Cluster(X_i, self.l3, Y_i)
                self.clusters.append(clu)
                self.inds.append(ind)
    
    def LDL(self, W):
        W = W.reshape(self.n_features, self.n_outputs)
        Y_hat = softmax(np.dot(self.X, W), axis = 1)
        
        loss, grad = KL(self.Y, Y_hat)
        grad = np.dot(self.X.T, grad)
        
        if self.l1 != 0:
            loss += 0.5 * self.l1 * (W **2).sum()
            grad += self.l1 * W
        
        for (ind, clu) in zip(self.inds, self.clusters):
            clu_l, clu_g = clu.LDL(Y_hat[ind])
            
            loss += clu_l
            grad += clu_g
        
        return loss, grad.reshape((-1, ))
        #return loss, grad
    

    
    def LDL_loss(self, W):
        l, _ = self.LDL(W)
        return l
    
    def LDL_grad(self, W):
        _, g = self.LDL(W)
        return g
    
    #only for echo
    '''
    def fun(self, W):
        l, _ = self.LDL(W)
        print(l)
    '''
    
    #optimize using pymanopt
    def solve_gd(self, max_iters = 600):
        manifold = Euclidean(self.n_features, self.n_outputs)
        problem = Problem(manifold=manifold, cost=self.LDL_loss, grad = self.LDL_grad)
        solver = SteepestDescent(max_iters)
        
        Xopt = solver.solve(problem)
        self.W = Xopt
    
    
    #optimize using l-bfgs
    def solve(self, max_iters = 600):
        
        #optimize using pymanopt        
        #self.solve_gd()

        #optimize using l-bfgs
        weights = np.eye(self.n_features, self.n_outputs).reshape((-1, ))
        optimize_result = minimize(self.LDL, weights, method = 'l-BFGS-b', jac = True, #callback = self.fun, 
                                   options = {'gtol':1e-6, 'disp': False, 'maxiter':max_iters })
        
        weights = optimize_result.x
        self.W = weights.reshape(self.n_features, self.n_outputs)

    
    def predict(self, X_test):
        return softmax(np.dot(append_intercept(X_test), self.W), axis = 1)
    
    
    def __str__(self):
        model = "LDLLDM_"
        model += str(self.l1) + "_"
        model += str(self.l2) + "_"
        model += str(self.l3) + "_"
        model += str(self.g)
        
        return model



#label matrix is missing
class LDLLDM_Incom: 
    class Cluster:
        def __init__(self, X, l, Y):
            
            self.X = X
            self.l = l
            self.I = np.eye(Y.shape[1])
            
            
            #the ground-truth LDM, for analysis
            self.gd_Z = barycenter_kneighbors_graph(Y.T).T
            
            self.I_Z = None
            self.L = None
          
        
        def LDM(self, Y):
            if (self.l == 0):
                return 0
                
            self.Z = barycenter_kneighbors_graph(Y.T).T
            self.I_Z = self.I - self.Z 
            self.L = np.dot(self.I_Z, self.I_Z.T)
        
        
        def LDL(self, Y_hat):
            if (self.l == 0) or (self.L is None):
                return 0, 0
            
            loss = (np.dot(Y_hat, self.I_Z) ** 2).sum()
            grad = np.dot(self.X.T, weighted_ME(Y_hat, 2 * np.dot(Y_hat, self.L)))
            
            return self.l * loss, self.l * grad
            
    
    
    def __init__(self, X, Y, J, l1, l2, l3, g = 0, clu_labels = None):
        
        self.X = X        
        self.Y = Y
        self.J = J
        
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        
        self.g = g
        
        self.n_examples, self.n_features, = X.shape
        self.n_outputs = Y.shape[1]
        
        #conduct K-means
        if clu_labels is None:
            kmeans = KMeans(n_clusters=g).fit(Y)
            clu_labels = kmeans.predict(Y)
            
        
        self.__init_clusters(clu_labels)

        
    def __init_clusters(self, clu_labels):
        self.clusters = []
        self.inds = []
        
        #global label distribution manifold
        if self.l2 != 0:
            clu = self.Cluster(self.X, self.l2, self.Y)
            self.clusters.append(clu)
            self.inds.append(np.asarray(np.ones(self.n_examples), dtype=np.bool))
        
        if (self.g > 1 and self.l3 != 0): 
            for i in range(self.g):
                ind = (clu_labels == i)
                X_i = self.X[ind]
                Y_i = self.Y[ind]
                clu = self.Cluster(X_i, self.l3, Y_i)
                self.clusters.append(clu)
                self.inds.append(ind)

    
    
    def LDL(self, W):
        #W = W.reshape(self.n_features, self.n_outputs)
        Y_hat = softmax(np.dot(self.X, W), axis = 1)
        
        loss, grad = KL(self.Y, Y_hat, self.J)
        grad = np.dot(self.X.T, grad)
        
        if self.l1 != 0:
            loss += 0.5 * self.l1 * (W **2).sum()
            grad += self.l1 * W
        
        for (ind, clu) in zip(self.inds, self.clusters):
            clu_l, clu_g = clu.LDL(Y_hat[ind])
            
            loss += clu_l
            grad += clu_g
        
        #return loss, grad.reshape((-1, ))
        return loss, grad

    def LDL_loss(self, W):
        l, _ = self.LDL(W)
        return l
    
    def LDL_grad(self, W):
        _, g = self.LDL(W)
        return g
    
    def solve_gd(self, weights, max_iters = 600):
        manifold = Euclidean(self.n_features, self.n_outputs)
        problem = Problem(manifold=manifold, cost=self.LDL_loss, grad = self.LDL_grad, verbosity = 0)
        solver = SteepestDescent(maxiter=max_iters)
        
        Xopt = solver.solve(problem, x=weights)
        return Xopt
        
    
    def solve_BFGS(self, weights, max_iters = 500):
        
        optimize_result = minimize(self.LDL, weights.reshape((-1, )), method = 'l-BFGS-b', jac = True,
                                   options = {'gtol':1e-6, 'disp': False, 'maxiter':max_iters })
        
        weights = optimize_result.x
        return weights.reshape(self.n_features, self.n_outputs)
    
    
    #solving by BFGS
    def solve(self, max_iters = 100):
        
        weights = np.random.uniform(-0.1, 0.1, (self.n_features, self.n_outputs))
        loss = [self.LDL(weights)[0]]
        print("loss", loss[-1])
        t1 = time.time()
        
        for i in range(max_iters):
            #solve LDL
            #weights =  self.solve_BFGS(weights)
            weights = self.solve_gd(weights)
            
            #solve LDM
            Y_hat = softmax(np.dot(self.X, weights), axis = 1)
            
            for (ind, clu) in zip(self.inds, self.clusters):
                clu.LDM(Y_hat[ind])
            
            
            if i % 5 == 0:
                loss.append(self.LDL(weights)[0])
                print("loss", loss[-1])
            
            #if abs(loss[-2] - loss[-1]) < 0.001:
            #    break
        
        t2 = time.time()
        self.W = weights
        
        return loss, t2 - t1
    
    def predict(self, X_test):
        return softmax(np.dot(X_test, self.W), axis = 1)
    
    
    def __str__(self):
        model = "LDLLDM_"
        model += str(self.l1) + "_"
        model += str(self.l2) + "_"
        model += str(self.l3) + "_"
        model += str(self.g)
        
        return model