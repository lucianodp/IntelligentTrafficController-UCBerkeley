
# coding: utf-8

# In[1]:

from code.IntersectionSimulator import IntersectionSimulator


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from time import time
from itertools import product

def evaluate(env, green_times):
    ns = 25
    observation = np.zeros(12)
    
    for _ in range(ns):
        env.run(green_times)
        observation += env.get_queue()
    
    return np.max(observation/ns)

def cross_entropy_method(env, dim, max_iter=300, n_samples=50, elite_prop=0.1, eps=0.01):
    # parameters
    mu = np.zeros(dim)
    var = np.ones(dim)

    elite_size = int(ceil(elite_prop * n_samples))  # number of points in elite set
    gt=0
    # algorithm
    for _ in range(max_iter):
        # collect samples from current distribution
        samples = np.random.multivariate_normal(mu, np.diag(var), n_samples)
        green_times = softmax(samples)

        # estimate value function for each parameter sample
        value = np.empty(n_samples)

        # estimate value function
        for i in range(n_samples):
            value[i] = evaluate(env, green_times[i])

        # select the p% lowest values
        elite_index = np.argsort(value)[:elite_size]
        elite_samples = samples[elite_index, :]

        # update parameters
        mu = np.mean(elite_samples, axis=0)  # mean of new parameters
        var = np.var(elite_samples, axis=0)  # variance of new parameters
        
        error = np.abs(softmax(mu.reshape(1,-1)) - gt)
        gt = softmax(mu.reshape(1,-1))
        
        if error.max() <= eps:
            print 'final iter: {0}'.format(_+1)
            print 'mu = {0}, var = {1}, gt = {2}, rew = {3}'.format(
                np.round(mu, 2), 
                np.round(var, 3), 
                np.round(gt[0], 3), 
                evaluate(env, gt.ravel())
            )
            break
            
    return mu


# auxiliary functions
def softmax(x):
    """Compute softmax values for each line in x."""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1).reshape(-1, 1)


# In[3]:

def generate_partition(k, n, disp):
    return [(n1, n2, n3, n4) 
             for n1 in range(0,k+1, disp)
             for n2 in range(0,k+1, disp)
             for n3 in range(0,k+1, disp)
             for n4 in range(0,k+1, disp)
             if n1+n2+n3+n4 == n
            ]


# In[5]:

throughput = 1200
k = 120
sources_id, eff_cycle_time = [2,4,6,8], 100
env = IntersectionSimulator(sources_id, eff_cycle_time)
mus = []
times = []

for d in generate_partition(throughput, throughput, k):
    env.set_demands(d)
    print d
    t0 = time()
    mus.append(cross_entropy_method(env, 4, max_iter=300, n_samples=50, elite_prop=0.1, eps=0.01))
    times.append(time()-t0)
    print "total time: {0}".format(time()-t0)
    print 


# In[10]:

x = np.array(generate_partition(throughput, throughput, k))
y = np.array([softmax(mu.reshape(1,-1)).ravel() for mu in mus])
data = np.hstack([x,y])
np.savetxt('complexIntersectionInitialCondition.txt', data, fmt='%.5f', delimiter=' ', newline='\n')

