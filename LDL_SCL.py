import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import time


# In[3]:


def next_batch(num, data, labels, codes):
    # Return a total of `num` random samples and labels.
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    codes_shuffle = [codes[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle), np.asarray(codes_shuffle)


# In[4]:


# cluster
def cluster_ave(labels_train, n):
    train_len = len(labels_train)
    # using k-means
    kmeans = KMeans(n_clusters=n, random_state=0).fit(labels_train)
    predict = kmeans.predict(labels_train)
    # using spectral clustering
    # predict = spectral_clustering(labels_train, n)
    classification = []
    for i in range(n):
        classification.append([])
    c = np.zeros([train_len, n]) + 10 ** -6
    for i in range(train_len):
        c[i][predict[i]] = 1
        classification[predict[i]].append(labels_train[i])
    p = []
    for i in range(n):
        p.append(np.average(classification[i], 0))
    p = np.array(p)
    return c, p


# In[5]:


# x: matrix of feature, n * d
# theta: weight matrix of feature, d * l, l is the number of labels
# c: matrix of code, n * m, m is the number of clusters
# w: weight matrix of code matrix, m * l
def predict_func(x, theta, c, w):
    matrix = np.dot(x, theta) + np.dot(c, w)
    matrix1 = matrix - np.max(matrix, 1).reshape(-1, 1)
    numerator = np.exp(matrix1)
    denominator = np.sum(np.exp(matrix1), 1).reshape(-1, 1)
    return numerator / denominator


# In[6]:


# label_real: real label of instance, n * l
# p: the average vector of cluster, number of clusters * l
def optimize_func(x, theta, c, w, label_real, p, lambda1, lambda2, lambda3, mu):
    label_predict = predict_func(x, theta, c, w)
    
    label_real = np.clip(label_real, 10 ** -6, 1)
    label_predict = np.clip(label_predict, 10 ** -6, 1)
    
    term1 = np.sum(label_real * np.log(label_real / label_predict))
    
    
    term2 = np.sum(theta ** 2)
    term3 = np.sum(w ** 2)
    dist = []
    for i in range(len(p)):
        dist.append(np.sum((label_predict - p[i]) ** 2, 1))
    dist = np.array(dist).T
    term4 = np.sum(c * dist)
    term5 = np.sum(1. / c)
    return term1 + lambda1 * term2 + lambda2 * term3 + lambda3 * term4 + mu * term5


# In[7]:


def gradient_theta(x, theta, c, w, label_real, P, m, n, lambda1, lambda2, lambda3):
    gradient1 = x.T.dot(predict_func(x, theta, c, w) - label_real)
    gradient2 = 2 * lambda1 * theta
    
    p_tmp = np.exp(np.dot(x, theta) + np.dot(c, w))
    p = (p_tmp.T / np.sum(p_tmp, axis=1)).T
    
    
    gradient3 = x.T.dot(np.multiply((len(P)*c.sum(axis=1)*p.T).T - c.dot(P), p-p**2))
    gradient3 *= 2*lambda3
    return gradient1 + gradient2 + gradient3



# @exeTime
def gradient_w(x, theta, c, w, label_real, P, m, n, lambda1, lambda2, lambda3):
    gradient1 = c.T.dot(predict_func(x, theta, c, w) - label_real)
    gradient2 = 2 * lambda2 * w
    
    p_tmp = np.exp(np.dot(x, theta) + np.dot(c, w))
    p = (p_tmp.T / np.sum(p_tmp, axis=1)).T
    
    
    gradient3 = c.T.dot(np.multiply((len(P)*c.sum(axis=1)*p.T).T - c.dot(P), p-p**2))
    gradient3 *= 2*lambda3
    return gradient1 + gradient2 + gradient3

# @exeTime
def gradient_c(x, code_len, theta, c, w, label_real, P, m, n, lambda1, lambda2, lambda3, mu):
    gradient1 = -label_real.dot(w.T)
    
    p_tmp = np.exp(np.dot(x, theta) + np.dot(c, w))
    p = (p_tmp.T / np.sum(p_tmp, axis=1)).T
    
    
    numerator = p_tmp.dot(w.T)
    denominator = np.sum(p_tmp, axis=1)
    gradient2 = (numerator.T/denominator).T

    '''
    gradient3 = np.zeros((len(x), code_len))
    for m in range(len(x)):
        for n in range(code_len):
            grad = 0.
            for l in range(len(label_real[0])):
                grad += (p[m][l] - P[n][l]) * p[m][l] * (w[n][l] - gradient2[m][n])
            grad *= 2 * lambda3 * c[m][n]
            gradient3[m][n] = grad
    '''

    
    gradient3 = opt_for(label_real, p, P, gradient2, w, c, len(x), code_len, lambda3)
    a = np.sum(p * p, 1)
    b = np.sum(P * P, 1)
    ab = p.dot(P.T)
    gradient4 = lambda3 * np.abs(np.repeat(a.reshape(-1, 1), len(P), 1) + np.repeat(np.array([b]), len(p), 0) - 2 * ab)
    gradient5 = - mu * c ** (-2)
    return gradient1 + gradient2 + gradient3 + gradient4 + gradient5


@njit    
def opt_for(label_real, p, P, gradient2, w, c, lenx, code_len, lambda3):
    gradient3 = np.zeros((lenx, code_len))
    for m in range(lenx):
        for n in range(code_len):
            grad = 0.
            for l in range(len(label_real[0])):
                grad += (p[m][l] - P[n][l]) * p[m][l] * (w[n][l] - gradient2[m][n])
            grad *= 2 * lambda3 * c[m][n]
            gradient3[m][n] = grad
    return gradient3
    



# In[37]:


def LDL_SCL(x_train, y_train, x_test, y_test, lambda1, lambda2, lambda3, c, reg = None):
    iters = 300
    batch = 50
    rho1 = 0.9
    rho2 = 0.999
    delta = 10 ** -8    # smoothing term
    epsilon = 0.001     # learning rate
    
    code_len = c+1
    features_dim = x_train.shape[1]
    labels_dim = y_train.shape[1]
    
    s1 = r1 = np.zeros([features_dim, labels_dim])
    s2 = r2 = np.zeros([code_len, labels_dim])
    
    mu = 1
    theta1 = np.ones([features_dim, labels_dim])
    w1 = np.ones([code_len, labels_dim])
    
    c1, p1 = cluster_ave(y_train, code_len)
    s3 = r3 = np.zeros_like(c1)
    
    loss1 = optimize_func(x_train, theta1, c1, w1, y_train, p1, lambda1, lambda2, lambda3, mu)
    loss = []
    loss.append(loss1)
    
    #for comparing runing time
    time1 = time.time()
    for i in range(iters):
        x_batch, y_batch, c_batch = next_batch(batch, x_train, y_train, c1)

        gradient1 = gradient_theta(x_batch, theta1, c_batch, w1, y_batch, p1, 0, 0, lambda1, lambda2, lambda3)
        s1 = rho1 * s1 + (1 - rho1) * gradient1
        s1_hat = s1 / (1 - rho1 ** (i+1))
        r1 = rho2 * r1 + (1 - rho2) * gradient1 * gradient1
        r1_hat = r1 / (1 - rho2 ** (i+1))

        gradient2 = gradient_w(x_batch, theta1, c_batch, w1, y_batch, p1, 0, 0, lambda1, lambda2, lambda3)
        s2 = rho1 * s2 + (1 - rho1) * gradient2
        s2_hat = s2 / (1 - rho1 ** (i + 1))
        r2 = rho2 * r2 + (1 - rho2) * gradient2 * gradient2
        r2_hat = r2 / (1 - rho2 ** (i + 1))

        gradient3 = gradient_c(x_train, code_len, theta1, c1, w1, y_train, p1, 0, 0, lambda1, lambda2, lambda3, mu)
        s3 = rho1 * s3 + (1 - rho1) * gradient3
        s3_hat = s3 / (1 - rho1 ** (i + 1))
        r3 = rho2 * r3 + (1 - rho2) * gradient3 * gradient3
        r3_hat = r3 / (1 - rho2 ** (i + 1))

        theta1 = theta1 - epsilon * s1_hat / (np.sqrt(r1_hat)+delta)
        w1 = w1 - epsilon * s2_hat / (np.sqrt(r2_hat)+delta)
        c1 = c1 - epsilon * s3_hat / (np.sqrt(r3_hat)+delta)

        loss2 = optimize_func(x_train, theta1, c1, w1, y_train, p1, lambda1, lambda2, lambda3, mu)
        if i > 5:
            loss.append(loss2)
        if np.abs(loss2 - loss1) < 0.0001:
            break
        else:
            mu = mu * 0.1
            loss1 = loss2    
    time2 = time.time()
    
    regression = []
    if reg is None:
        for i in range(code_len):
            lr = LinearRegression()
            lr.fit(x_train, c1[:, i])
            regression.append(lr)
    else:
        for i in range(code_len):
            lr = Ridge(alpha = reg)
            lr.fit(x_train, c1[:, i])
            regression.append(lr)
    
    codes = []
    for lr in regression:
        codes.append(lr.predict(x_test))
    codes = np.array(codes).T
    label_pre = predict_func(x_test, theta1, codes, w1)
    
    return label_pre, time2 - time1

