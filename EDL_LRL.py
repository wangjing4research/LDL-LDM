#this code is shared by https://github.com/NJUST-IDAM/

from sklearn.cluster import KMeans
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from scipy.special import softmax
import time


lambda1 = 1e-3
lambda2 = 1e-2
m = 5

# output model
def predict_func(x, m_theta, f_dim, l_dim):
    m_theta = m_theta.reshape(f_dim, l_dim)
    numerator = softmax(np.dot(x, m_theta), axis = 1)
    return numerator


# w: q*L, x: n*q, d_: n*L, z: a list which includes m arrays
def obj_func1(w, x, d_, z, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    term1 = 0.5 * np.sum((predict_func(x, w, f_dim, l_dim) - d_) ** 2)
    term2 = (w ** 2).sum()
    term3 = 0.
    for i in range(m):
        term3 += np.linalg.norm(z[i], ord='nuc')
    loss1 = term1 + lambda1 * term2 + lambda2 * term3
    return loss1


# the objective function of sub-question
# x: a list which includes the result of clustering
# d_: a list which includes the result of clustering
# z: a list which includes m arrays
# Lambda: a list of Lagrange Multiplier
# rho: a list which includes m members
def obj_func2(w, x, d_, z, Lambda, rho, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    term1 = 0.
    term2 = 0.
    term3 = 0.
    for i in range(m):
        xi_pre = predict_func(x[i], w, f_dim, l_dim)
        term1 += 0.5 * np.sum((xi_pre - d_[i]) ** 2)
        term2 += np.sum(Lambda[i] * (xi_pre - z[i]))
        term3 += (rho[i] / 2) * np.sum((xi_pre - z[i]) ** 2)
    loss2 = term1 + term2 + term3 + lambda1 * (w ** 2).sum()
    return loss2


# update w, the prime of parameter w
def fprime(w, x, d_, z, Lambda, rho, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    gradient = np.zeros_like(w)
    for i in range(m):
        #modProb = np.exp(np.dot(x[i], w))
        #sumProb = np.sum(modProb, 1)
        #disProb = modProb / (sumProb.reshape(-1, 1) + 0.000001)
        disProb = softmax(np.dot(x[i], w), axis = 1)
        disProb_2 = disProb - disProb*disProb
        
        gradient += np.dot(np.transpose(x[i]), (disProb - d_[i]) * disProb_2)
        gradient += np.dot(np.transpose(x[i]), Lambda[i] * disProb_2)
        gradient += rho[i] * np.dot(np.transpose(x[i]), (disProb - z[i]) * disProb_2)
    gradient += 2 * lambda1 * w
    return gradient.ravel()


# update z
def update_z(w, x, z, Lambda, rho, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    z_new = []
    for i in range(m):
        u, sigma, vt = np.linalg.svd(predict_func(x[i], w, f_dim, l_dim)+Lambda[i]/rho[i])
        sigma_new = [s if s-(lambda2/rho[i]) > 0 else 0 for s in sigma]
        temp = np.diag(sigma_new)
        height, width = z[i].shape
        if len(sigma) < width:
            temp = np.c_[temp, np.zeros([len(sigma), width-len(sigma)])]
        if len(sigma) < height:
            temp = np.r_[temp, np.zeros([height-len(sigma), width])]
        z_new.append(np.dot(np.dot(u, temp), vt))
    return z_new


# update Lambda
def update_Lambda(w, x, z, Lambda, rho, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    Lambda_new = []
    for i in range(m):
        Lambda_new.append(Lambda[i] + rho[i]*(predict_func(x[i], w, f_dim, l_dim)-z[i]))
    return Lambda_new

def EDL_LRL(x_train, x_test, y_train, y_test):
    
    features_dim = len(x_train[0])
    labels_dim = len(y_train[0])

    w = np.random.rand(features_dim, labels_dim)    # update
    #kmeans = KMeans(n_clusters=m).fit(y_train)
    #kmeans_result = kmeans.predict(y_train)
    kmeans = KMeans(n_clusters=m).fit(x_train)
    kmeans_result = kmeans.predict(x_train)
    x_result = []
    d_result = []
    for i in range(m):
        x_result.append([])
        d_result.append([])
    for i in range(len(x_train)):   
        x_result[kmeans_result[i]].append(list(x_train[i]))
        d_result[kmeans_result[i]].append(list(y_train[i]))
    z = []  # update
    for i in range(m):
        # z.append(np.ones_like(d_result[i]))
        z.append(np.zeros_like(d_result[i]))
        # z.append(np.ones_like(d_result[i]) / labels_dim)
    Lambda = []     # update
    for i in range(m):
        Lambda.append(np.zeros_like(d_result[i]))
    rho = np.ones(m) * (10 ** -6)     # parameter
    rho_max = 10 ** 6
    beta = 1.1  # increase factor
    # update step
    loss = obj_func1(w, x_train, y_train, z, features_dim, labels_dim)
    #print(loss)
    
    time1 = time.time()
    for i in range(50):
        #print(i)
        #print("-" * 20)
        # print(obj_func2(w, x_result, d_result, z, Lambda, rho, features_dim, labels_dim))
        result = fmin_l_bfgs_b(obj_func2, w, fprime, 
                               args=(x_result, d_result, z, Lambda, rho, features_dim, labels_dim),
                               pgtol=0.001, maxiter=10)
        w = result[0]
        z = update_z(w, x_result, z, Lambda, rho, features_dim, labels_dim)
        Lambda = update_Lambda(w, x_result, z, Lambda, rho, features_dim, labels_dim)
        loss_new = obj_func1(w, x_train, y_train, z, features_dim, labels_dim)
        if abs(loss - loss_new) < 10 ** -8 or loss_new > loss:
            break
        rho = np.min([rho[0]*beta, rho_max]) * np.ones(m)
        loss = loss_new
        #print(loss)
    time2 = time.time()

    # predict the label distributions of test set
    pre_test = predict_func(x_test, w, features_dim, labels_dim)
    
    return pre_test, time2 - time1






