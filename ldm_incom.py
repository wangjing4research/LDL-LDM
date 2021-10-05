import os
import pickle
import numpy as np
from ldl_metrics import score
from LDLLDM import LDLLDM_Incom
import sys

def save_dict(dataset, scores, name):
    with open(dataset + "//" + name + ".pkl", 'wb') as f:
        pickle.dump(scores, f)

def load_dict(dataset, name):
    if not os.path.exists(dataset + "//" + name):
        name += ".pkl"
        
    with open(dataset + "//" + name, 'rb') as f:
        return pickle.load(f)


def do_LDLLDM_Incom(param):
    train_x, train_y, test_x, test_y, J, l1, l2, l3, g, c_label = param
    ldlldm = LDLLDM_Incom(train_x, train_y, J, l1, l2, l3, g = g, clu_labels = c_label)
    ldlldm.solve()
    val = score(test_y, ldlldm.predict(test_x))
    print(val)
    return (str(ldlldm), val)


def run_LDLLDM_Incom(train_x, train_y, test_x, test_y, J, scores):
    L1 = [0.0001]
    L2 = [1]
    L3 = [1]
    groups = [10]
    
    params = [(train_x, train_y, test_x, test_y, J, l1, l2, l3, g, None) 
              for l1 in L1 for l2 in L2 for l3 in L3 for g in groups]
      
    for (key, val) in map(do_LDLLDM_Incom, params):

        if not key in scores.keys():
            scores[key] = []
        scores[key].append(val)


def run_KF(dataset):
    print(dataset)
    X, Y = np.load(dataset + "/feature.npy"), np.load(dataset + "/label.npy")
    train_inds = load_dict(dataset, "train_fold")
    test_inds = load_dict(dataset, "test_fold")

    rhos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for rho in rhos:
        print("incomplete:", rho)
        Js = load_dict(dataset, "train_J_" + str(rho))
        scores = dict()
        for i in range(10):
            print("fold", i)
            train_x, train_y = X[train_inds[i + 1]], Y[train_inds[i + 1]]
            test_x, test_y = X[test_inds[i + 1]], Y[test_inds[i + 1]]
            run_LDLLDM_Incom(train_x, train_y, test_x, test_y, Js[i], scores)
        
        save_dict(dataset, scores, "LDLLDM_Incom_" + str(rho))
            

if __name__ == "__main__":
    run_KF("Scene")
        
