
# coding: utf-8

# In[1]:

from code.IntersectionSimulator import IntersectionSimulator


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from time import time
from itertools import product

def plot_policy(net):
        s1 = np.linspace(0, 1200, 300)
        gt = net.predict(s1.reshape(-1,1)/1200.0)
        
        plt.plot(s1, gt[:,0])
        
        plt.title('Final Policy - 1200 throughtput')
        plt.xlabel('Demand on Link 1')
        plt.ylabel('Proportion to Link 1')
        
        plt.show()

def evaluate(env, gt, demand):
    # estimate reward 
    n_samp = 25
    q = np.zeros(12)
    
    demand = [int(d) for d in demand]
    env.set_demands(demand)
    
    for i in range(n_samp):
        env.run(gt)
        q += env.get_queue()

    return np.max(q / n_samp)


def evaluate2(env, net):
    #estimate current policie performance
    ns = 100
    demands = np.random.dirichlet([1.0]*4, size=ns)  # uniform sample of positive numbers summing to 1
    
    gts = net.predict(demands)
    
    r = 0
    for i in range(ns):
        r += evaluate(env, gts[i], 1200 * demands[i])
    
    return r/ns

def cross_entropy_method(env, net, max_iter=300, n_samples=50, elite_prop=0.1, eps=0.01):
    # parameters
    sizes = [p.shape for p in net.get_weights()]
    dims = [len(p.flatten()) for p in net.get_weights()]
    dimsum = np.cumsum(dims)
    dim = sum(dims)
    
    #mu = np.zeros(dim)
    mu = np.random.uniform(-1,1,dim)
    var = np.ones(dim)

    elite_size = int(ceil(elite_prop * n_samples))  # number of points in elite set

    # algorithm
    for _ in range(max_iter):
        samples = np.random.multivariate_normal(mu, np.diag(var), n_samples)
        
        # estimate value function for each parameter sample
        value = np.empty(n_samples)
        
        t = time()
        # estimate value function
        for i in range(n_samples):
            weights = [w.reshape(s) for w,s in zip(np.split(samples[i], dimsum), sizes)]
            net.set_weights(weights)
            
            value[i] = evaluate2(env, net)
        
        # select the p% lowest values
        elite_index = np.argsort(value)[:elite_size]
        elite_samples = samples[elite_index]

        # update parameters
        mu = np.mean(elite_samples, axis=0)  # mean of new parameters
        var = np.var(elite_samples, axis=0)+0.0001  # variance of new parameters
        
        net.set_weights([w.reshape(s) for w,s in zip(np.split(mu, dimsum), sizes)])
        print var.max(),
        print 'iter = {0}, rew = {1}, iter time = {2}'.format(_+1, evaluate2(env, net), time()-t)

        if max(var) < eps:
            break
        
    return mu


# In[3]:

from keras.models import Sequential
from keras.layers import Dense

# create a neural network
net = Sequential()
net.add(Dense(100, input_dim=4, activation='tanh'))
net.add(Dense(4, activation='softmax'))


# In[4]:

sources_id, eff_cycle_time = [2,4, 6, 8], 100
env = IntersectionSimulator(sources_id, eff_cycle_time)

theta = cross_entropy_method(env, net, max_iter=200, n_samples=100, elite_prop=0.20, eps=0.01)


# In[ ]:

plot_policy(net)


# In[ ]:



