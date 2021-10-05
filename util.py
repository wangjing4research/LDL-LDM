import os
import pickle
import numpy as np

def save_dict(dataset, scores, name):
    with open (dataset + "//" + name + ".pkl", 'wb') as f:
        pickle.dump(scores, f)

def load_dict(dataset, name):
    if not os.path.exists(dataset + "//" + name):
        name += ".pkl"
        
    with open(dataset + "//" + name, 'rb') as f:
        return pickle.load(f)
    
    
def dic_combine(data_set, name):
    dic = {}
    
    for i in range(10):
        print(i)
        dic_tmp = load_dict(data_set, name + str(i))
        
        if len(dic) == 0:
            dic.update(dic_tmp)
        else:
            for key in dic_tmp.keys():
                if key in dic.keys():
                    dic[key].append(dic_tmp[key][0])
                    
    return dic



def make_incomplete(Y, rho):
    J = np.ones_like(Y)
    if rho == 1:
        print("full matrix")
        return J
    missing = np.random.choice(Y.size, replace = False, size = int((1 - rho) * Y.size))
    J.flat[missing] = 0
    return J


def best_parameters(dic):
    keys = []
    results = []
    for key in dic.keys():
        keys.append(key)
        results.append(np.array(dic[key]).mean(0))
    results = np.array(results)

    best_keys = set()
    for i in range(6):
        if i < 4:
            ind = results[:, i].argmin()
        else:
            ind = results[:, i].argmax()
        best_keys.add(keys[ind])
    return list(best_keys)


def dic_combine(data_set, name):
    dic = {}
    
    for i in range(10):
        if not os.path.exists(data_set + "//" + name + str(i) + ".pkl"):
            continue
        print(i)
        dic_tmp = load_dict(data_set, name + str(i))
        
        if len(dic) == 0:
            dic.update(dic_tmp)
        else:
            for key in dic_tmp.keys():
                if key in dic.keys():
                    dic[key].append(dic_tmp[key][0])
    return dic


#analyze the dic
def analyze(dic):
    results = []
    keys = []
    for key in dic.keys():
        results.append(np.array(dic[key]).mean(0))
        keys.append(key)
    
    results = np.array(results)
    keys = np.array(keys)
    
    optimal = np.array([0] * results.shape[0])
    
    for l in range(6):
        tmp = results[:, l]
        if l < 4:
            optimal[tmp <= tmp.min()] += 1
        else:
            optimal[tmp >= tmp.max()] += 1
            
    return keys[optimal >= optimal.max()]


#analyze the dic
def analyze1(dic):
    results = []
    keys = []
    for key in dic.keys():
        results.append(np.array(dic[key]).mean(0))
        keys.append(key)
    
    results = np.array(results)
    keys = np.array(keys)
    
    optimal = np.array([0] * results.shape[0])
    
    for l in range(6):
        tmp = results[:, l]
        if l < 4:
            optimal[tmp <= tmp.min()] += 1
        else:
            optimal[tmp >= tmp.max()] += 1
            
    return results[optimal >= optimal.max()]

