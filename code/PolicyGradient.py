
# coding: utf-8

# In[1]:

from code.IntersectionSimulator import IntersectionSimulator


# In[2]:

# import keras - Neural Network library
from keras.models import Sequential
from keras.layers import Dense

import theano
import theano.tensor as T

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from time import time
from itertools import izip


# In[3]:

import math

def sample_von_mises(mu, kappa):
    # auxiliary parameters
    d = len(mu)
    
    # sample v uniformily from S^(d-2)
    normal_samples = np.random.normal(size=(d-1,))
    v = normal_samples/np.linalg.norm(normal_samples)
    
    # sample w
    b = (d-1)/(2*kappa + math.sqrt(4*kappa*kappa + (d-1)*(d-1)))
    x = (1-b)/(1+b)
    c = kappa*x + (d-1)*math.log(1-x*x)
    
    while True:
        z = np.random.beta((d-1)/2.0, (d-1)/2.0)
        u = np.random.uniform(0,1)
        w = (1 - (1+b)*z)/(1 - (1-b)*z)
        
        if kappa*w + (d-1)*math.log(1-x*w) - c > math.log(u):
            break
    
    # von mises for mu = (1, 0, 0,...)
    von_mises_sample = np.hstack([w, math.sqrt(1-w*w) * v])
    
    # rotate to correct position
    P = get_rotation_matrix(mu)  # takes (1,0,0,...) to mu
    return np.dot(P, von_mises_sample)
    
def get_rotation_matrix(x):
    n = len(x)
    A = np.zeros([n,n])
    A[:,0] = x

    i = 0
    while x[i] == 0:
        i += 1

    if i > 0:
        A[:i, 1:i+1] = np.eye(i)

    A[i+1:, i+1:] = np.eye(n-i-1)

    A[i,1:] = [-x[j]/x[i] for j in range(n) if j != i]

    return gram_schmidt_ortonormalization(A)
    
def gram_schmidt_ortonormalization(A):

    Q = np.zeros(A.shape)

    for k in range(A.shape[1]):
        q = A[:, k]
        for j in range(k):
            q = q - np.dot(q, Q[:,j])*Q[:,j]

        Q[:, k] = q/np.linalg.norm(q)
    
    return Q


# In[4]:

# create a neural network
net = Sequential()
net.add(Dense(100, input_dim=4, activation='tanh'))
net.add(Dense(4, activation='softmax'))

net.set_weights([np.random.uniform(low=-1, high=1, size=p.shape) for p in net.get_weights()])


# In[5]:

params = []
for elem in net.layers:
    params += [elem.W, elem.b]

mean_var = theano.tensor.sqrt(net.output)
    
compute_green_times = theano.function([net.input], net.output, allow_input_downcast=True)
compute_means = theano.function([net.input], mean_var, allow_input_downcast=True)

jacobs = [theano.gradient.jacobian(mean_var[0], p) for p in params]
compute_jacob = theano.function([net.input], jacobs, allow_input_downcast=True)


# In[6]:

def plot_policy():
        """
        Plots the actions taken at each possible state (position,velovity) 
        """
        s1 = np.linspace(0, 1200, 300)
        gt = compute_green_times(s1.reshape(-1,1)/1200.0)
        
        plt.plot(s1, gt[:,0])
        
        plt.title('Final Policy - 1200 throughtput')
        plt.xlabel('Demand on Link 1')
        plt.ylabel('Proportion to Link 1')
        
        plt.show()

def plot_reward(env):
    
    s1 = np.linspace(0, 1200, 300)
    gt = compute_green_times(s1.reshape(-1,1)/1200.0)
    
    y = []
    for s, g in zip(s1, gt):
        y.append(evaluate(env, g, [s, 1200-s]))
    plt.plot(s1, y)
    plt.show()


# In[7]:

def evaluate(env, green_times, state):
    ns = 25
    observation = np.zeros(12)
    
    state = [int(s) for s in state]
    env.set_demands(state)
    for _ in range(ns):
        env.run(green_times)
        observation += env.get_queue()
    
    return np.max(observation/ns)

def test_controller(env, ns):
    # sample state
    sample_states = state_sampling(ns)
    
    # retrieve mean and green times
    gt = compute_green_times(sample_states/1200.0)
    
    # get mean cost
    r=0
    for i in range(ns):
        r += evaluate(env, gt[i], sample_states[i])

    return r/ns


# In[8]:

def state_sampling(ns):
    #sd = np.random.randint(low=0, high=1200, size=(ns,1))
    #return np.hstack([sd, 1200-sd])
    return 1200 * np.random.dirichlet([1]*4, size=ns)
    
def cost_function_gradient(env, ns, kappa):
    # initialize cost gradient to zero
    cost_gradient = [np.zeros(p.shape) for p in net.get_weights()]
    
    # sample states (uniform distribution)
    sample_states = state_sampling(ns)

    # get mean of von_mises distribution
    means = compute_means(sample_states/1200.0)

    # calculate cost function gradient
    for i in range(ns):
        # sample action (von mises sampling) and compute green times
        sample_action = sample_von_mises(means[i], kappa)
        green_time = sample_action**2

        # get reward/cost
        r = evaluate(env, green_time, sample_states[i])

        # compute gradient of mean
        grads = compute_jacob(sample_states[i].reshape(1,-1)/1200.0)

        # gradients of cost function
        for cg, mg in izip(cost_gradient, grads):
            cg += r * kappa * (sample_action[0] * mg[0] + sample_action[1] * mg[1] + 
                               sample_action[2] * mg[2] + sample_action[3] * mg[3])/ns
            
    return cost_gradient


# In[9]:

# create simulator
sources_id, eff_cycle_time = [2,4,6,8], 100
env = IntersectionSimulator(sources_id, eff_cycle_time)

# simulation parameters
kappa = 100
max_iter = 100
alpha = 0.005
ns = 500

# auxiliary parameters
t_mean=0

t_tot = time()
for _ in range(max_iter):
    t0 = time()
    
    # estimate gradient
    cost_gradient = cost_function_gradient(env, ns, kappa)
    
    # gradient descent step
    net.set_weights([val - alpha*cg for val,cg in izip(net.get_weights(), cost_gradient)])

    # printing
    grad_mod = max([np.max(np.abs(a)) for a in cost_gradient])
    net_mod = np.sqrt(sum([(a**2).sum() for a in net.get_weights()]))
    
    #if gradient too small, break
    if grad_mod < 0.1:
        print "Final iteration. Last reward: {0}, grad mod: {1}, net_mod: {2}".format(
            test_controller(env, 400),
            np.round(grad_mod, 2),
            np.round(net_mod, 2))
        
        break
    
    # print
    t_mean += (time()-t0 - t_mean)/(1+_)
    if _ % 1 == 0:
        #print cost_gradient
        print "iter: " + str(1+_) + ",",
        print "mean time: {0}, remaining_time: {1} min, last reward: {2}, grad mod: {3}, net_mod: {4}".format(
            np.round(t_mean, 2), 
            np.round(t_mean*(max_iter-_)/60.0, 2), 
            test_controller(env, 400),
            np.round(grad_mod, 3),
            np.round(net_mod, 3))
        
    
print "total simulation time: {0} s = {1} min".format(time()-t_tot, (time()-t_tot)/60)


# In[ ]:

plot_policy()


# In[ ]:

plot_reward(env)


# In[ ]:



