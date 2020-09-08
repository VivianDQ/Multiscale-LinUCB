
# coding: utf-8

# In[ ]:



# coding: utf-8

# In[29]:


import numpy as np
import random
import math
import matplotlib.pyplot as plot
import collections
import time


# In[32]:


class DLinUCB:
    '''
    this class contains the algorithm of LinUCB and Multiscale_LinUCB
    '''
    def __init__(self, class_context, T):
        self.data = class_context
        self.T = T
        self.gamma = self.data.gamma
        self.d = self.data.d
        
    def dlinucb(self, lamda, L, S, gamma):
        '''
        this is the LinUCB algorithm
        Input: explore is the exploration rate
        Output: regret is an array of length T, each element is the cumulative regret up to time t
        '''
        K = self.data.K
        d = self.data.d
        regret = np.zeros(self.T)
        feature = self.data.fv
        
        A = lamda*np.identity(d) 
        b = np.zeros(d) 
        A_inverse = np.identity(d) / lamda
        A_hat = np.identity(d)*lamda
        
        theta_hat = np.zeros(d)
        linucb_idx = [0] * K
        for t in range(self.T):
            beta = math.sqrt(lamda) * S + math.sqrt(2*math.log(self.T) + d*math.log(1+ (L**2*(1-gamma**(2*t))) / (lamda*d*(1-gamma**2)) ))
            fv = feature[t]
            for arm in range(K):
                linucb_idx[arm] = fv[arm].dot(theta_hat) + beta * math.sqrt(fv[arm].dot(A_inverse).dot(A_hat).dot(A_inverse).dot(fv[arm].T))
            choose_arm = np.argmax(linucb_idx)
            rs = self.data.random_sample(t, choose_arm)
            
            A = gamma*A + np.outer(fv[choose_arm], fv[choose_arm]) + (1-gamma)*lamda*np.identity(d)
            A_hat = gamma**2*A_hat + np.outer(fv[choose_arm], fv[choose_arm]) + (1-gamma**2)*lamda*np.identity(d)
            A_inverse = np.linalg.inv(A)
            b = gamma*b + rs*fv[choose_arm]
            theta_hat = A_inverse.dot(b)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][choose_arm]
        return regret
  
    
 


# In[ ]:


class SW_LinUCB:
    '''
    this class contains the algorithm of LinUCB and Multiscale_LinUCB
    '''
    def __init__(self, class_context, T):
        self.data = class_context
        self.T = T
        self.gamma = self.data.gamma
        self.d = self.data.d
        
    def sw_linucb(self, lamda, L, S, w):
        '''
        this is the LinUCB algorithm
        Input: explore is the exploration rate
        Output: regret is an array of length T, each element is the cumulative regret up to time t
        '''
        T = self.T
        K = self.data.K
        d = self.data.d
        regret = np.zeros(self.T)
        feature = self.data.fv
        
        A = lamda*np.identity(d) 
        b = np.zeros(d) 
        A_inverse = np.identity(d) / lamda
        
        theta_hat = np.zeros(d)
        linucb_idx = [0] * K
        window = collections.deque()
        
        for t in range(self.T):
            beta = math.sqrt(lamda) * S + math.sqrt(d*math.log(T+T*w*L**2/lamda))
            fv = feature[t]
            for arm in range(K):
                linucb_idx[arm] = fv[arm].dot(theta_hat) + beta * math.sqrt(fv[arm].dot(A_inverse).dot(fv[arm].T))
            choose_arm = np.argmax(linucb_idx)
            rs = self.data.random_sample(t, choose_arm)
            window.append((t, fv[choose_arm], rs))
            
            while window and window[0][0] <= t-w:
                window.popleft()
            
            A = lamda*np.identity(d) 
            b = np.zeros(d)
            for _,x,y in window:
                #A += np.outer(fv[choose_arm], fv[choose_arm]) 
                A += np.outer(x,x)
                b += y*x
                #b += rs*fv[choose_arm]
            A_inverse = np.linalg.inv(A)
            theta_hat = A_inverse.dot(b)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][choose_arm]
        return regret
  
    
 

