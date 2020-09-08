
# coding: utf-8

# In[ ]:


import numpy as np
import random
import math
import matplotlib.pyplot as plot
import collections
import time


# In[ ]:


# setting 1
class abrupt2:
    '''
    this class contains the non-stationary contextual bandit data, when there is a change point, 
    random number of arms have changed mean reward
    
    Input:
    T : total time horizon
    K : an integer, number of arms
    n_cp : total number of change points
    ub_reward: upper bound of the true reward
    lb_reward: lower bound of the true reward
    dim : the dimension of feature vectors
    fv : the preset feature vectors for all arms at all times
    '''
    def __init__(self, n_arm, n_cp, lb_reward, ub_reward, T, dim, fv):
        '''
        cp : a list of change points, including the start time and end time
        reward: a list of np.array, each element of the list is an array of length T, with true reward of an arm in [1,T]
        optimal: a list of length T, with optimal rewards during [1,T]
        '''
        self.K = n_arm   
        self.d = dim
        self.ub = ub_reward  
        self.lb = lb_reward
        self.T = T
        self.fv = fv
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.gamma = n_cp
        # theta is random generated in the beginning
        self.theta_pool = [np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0, -1])]
        # there should be no change points when t<dim
        cut = T//5
        self.cp = [0, cut, cut*2, cut*3, T]
        self.vt = 6

    def change(self):
        """
        this function makes change to the contextual bandit data after initialization
        p is the probability of change each time
        """
        for i in range(1, len(self.cp)):
            for t in range(self.cp[i-1], self.cp[i]):  
                self.theta = self.theta_pool[i-1] 
                self.reward[t] = [self.fv[t][i].dot(self.theta) for i in range(self.K)]
                self.optimal[t] = max(self.reward[t])      
        
    def random_sample(self, t, i):
        '''
        random sample bernoulli reward for arm i #, normal with variance 1, t is current time
        ''' 
        return np.random.normal(self.reward[t][i], 1)


# In[ ]:



# setting 2
class abrupt_fix:
    '''
    this class contains the non-stationary contextual bandit data, when there is a change point, 
    random number of arms have changed mean reward
    
    Input:
    T : total time horizon
    K : an integer, number of arms
    n_cp : total number of change points
    ub_reward: upper bound of the true reward
    lb_reward: lower bound of the true reward
    dim : the dimension of feature vectors
    fv : the preset feature vectors for all arms at all times
    '''
    def __init__(self, n_arm, n_cp, lb_reward, ub_reward, T, dim, fv):
        '''
        cp : a list of change points, including the start time and end time
        reward: a list of np.array, each element of the list is an array of length T, with true reward of an arm in [1,T]
        optimal: a list of length T, with optimal rewards during [1,T]
        '''
        self.K = n_arm   
        self.d = dim
        self.ub = ub_reward  
        self.lb = lb_reward
        self.T = T
        self.fv = fv
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.gamma = n_cp
        # theta is random generated in the beginning
        self.theta = np.random.uniform(self.lb, self.ub, dim)
        #self.theta_pool = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])]
        # there should be no change points when t<dim
        self.cp = np.linspace(0,T,n_cp+1)
        self.vt = 0
        

    def change(self):
        """
        this function makes change to the contextual bandit data after initialization
        p is the probability of change each time
        """
        for t in range(self.T):  
            old = self.theta
            self.reward[t] = [self.fv[t][i].dot(self.theta) for i in range(self.K)]
            self.optimal[t] = max(self.reward[t])      
            if t+1 in set(self.cp):
                self.theta = np.random.uniform(self.lb, self.ub, self.d)
                self.vt += np.linalg.norm(old-self.theta)
                old = self.theta
        
    def random_sample(self, t, i):
        '''
        random sample bernoulli reward for arm i #, normal with variance 1, t is current time
        ''' 
        return np.random.normal(self.reward[t][i], 1)
    
    




# In[ ]:


class context_random:
    '''
    this class contains the non-stationary contextual bandit data, when there is a change point, 
    random number of arms have changed mean reward
    
    Input:
    T : total time horizon
    K : an integer, number of arms
    n_cp : total number of change points
    ub_reward: upper bound of the true reward
    lb_reward: lower bound of the true reward
    dim : the dimension of feature vectors
    fv : the preset feature vectors for all arms at all times
    '''
    def __init__(self, n_arm, n_cp, lb_reward, ub_reward, T, dim, fv):
        '''
        cp : a list of change points, including the start time and end time
        reward: a list of np.array, each element of the list is an array of length T, with true reward of an arm in [1,T]
        optimal: a list of length T, with optimal rewards during [1,T]
        '''
        self.K = n_arm   
        self.d = dim
        self.ub = ub_reward  
        self.lb = lb_reward
        self.T = T
        self.fv = fv
        self.reward = [None] * self.T
        self.optimal = [None] * self.T
        self.gamma = n_cp
        # theta is random generated in the beginning
        self.theta = np.random.uniform(self.lb, self.ub, self.d)
        # there should be no change points when t<dim
        for t in range(self.d):
            self.reward[t] = [self.fv[t][i].dot(self.theta) for i in range(self.K)] 
            self.optimal[t] = max(self.reward[t])  # max reward
        self.cp = [0]
        self.vt = 0

    def change(self, p):
        """
        this function makes change to the contextual bandit data after initialization
        p is the probability of change each time
        """
        for t in range(self.d, self.T):  
            old = self.theta
            if np.random.uniform(0,1) <= p:
                self.theta = np.random.uniform(self.lb, self.ub, self.d)
                self.cp += [t]
                self.vt += np.linalg.norm(self.theta-old)
                old = self.theta
            self.reward[t] = [self.fv[t][i].dot(self.theta) for i in range(self.K)]
            self.optimal[t] = max(self.reward[t])      
        self.cp.append(self.T)
        
    def random_sample(self, t, i):
        '''
        random sample reward for arm i, normal with variance 1, t is current time
        ''' 
        return np.random.normal(self.reward[t][i], 1)
   


