
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


class Algos_context:
    '''
    this class contains the algorithm of LinUCB and Multiscale_LinUCB
    '''
    def __init__(self, class_context, T):
        self.data = class_context
        self.T = T
        self.gamma = self.data.gamma
        self.d = self.data.d
        
    def linUCB(self, explore):
        '''
        this is the LinUCB algorithm
        Input: explore is the exploration rate
        Output: regret is an array of length T, each element is the cumulative regret up to time t
        '''
        K = self.data.K
        d = self.data.d
        regret = np.zeros(self.T)
        feature = self.data.fv
        
        A = [np.identity(d) for _ in range(K)]
        b = [np.zeros(d) for _ in range(K)]
        A_inverse = [np.identity(d) for _ in range(K)]
        
        theta_hat = [None] * K
        linucb_idx = [0] * K
        for t in range(self.T):
            fv = feature[t]
            for arm in range(K):
                theta_hat[arm] = A_inverse[arm].dot(b[arm])
                linucb_idx[arm] = fv[arm].dot(theta_hat[arm]) + explore * math.sqrt(fv[arm].dot(A_inverse[arm]).dot(fv[arm].T))
            choose_arm = np.argmax(linucb_idx)
            rs = self.data.random_sample(t, choose_arm)
            A[choose_arm] += np.outer(fv[choose_arm], fv[choose_arm])
            A_inverse[choose_arm] = np.linalg.inv(A[choose_arm])
            b[choose_arm] += rs*fv[choose_arm]
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][choose_arm]
        return regret
    
    def linUCB_joint(self, explore):
        '''
        this is the LinUCB algorithm
        Input: explore is the exploration rate
        Output: regret is an array of length T, each element is the cumulative regret up to time t
        '''
        K = self.data.K
        d = self.data.d
        regret = np.zeros(self.T)
        feature = self.data.fv
        
        A = np.identity(d) 
        b = np.zeros(d) 
        A_inverse = np.identity(d) 
        
        linucb_idx = [0] * K
        for t in range(self.T):
            fv = feature[t]
            theta_hat = A_inverse.dot(b)
            for arm in range(K):
                linucb_idx[arm] = fv[arm].dot(theta_hat) + explore * math.sqrt(fv[arm].dot(A_inverse).dot(fv[arm].T))
            choose_arm = np.argmax(linucb_idx)
            rs = self.data.random_sample(t, choose_arm)
            A += np.outer(fv[choose_arm], fv[choose_arm])
            A_inverse = np.linalg.inv(A)
            b += rs*fv[choose_arm]
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][choose_arm]
        return regret
    
    def linucb_oracle_joint(self, explore):
        K = self.data.K
        d = self.data.d
        cp = self.data.cp
        #print(cp)
        regret = np.zeros(self.T)
        feature = self.data.fv
        
        '''
        A = np.identity(d) 
        b = np.zeros(d) 
        A_inverse = np.identity(d) 
        '''
        linucb_idx = [0] * K
        for i in range(1, len(cp)):
            A = np.identity(d) 
            b = np.zeros(d) 
            A_inverse = np.identity(d) 
            linucb_idx = [0] * K
            for t in range(int(cp[i-1]), int(cp[i])):  
                fv = feature[t]
                theta_hat = A_inverse.dot(b)
                for arm in range(K):
                    linucb_idx[arm] = fv[arm].dot(theta_hat) + explore * math.sqrt(fv[arm].dot(A_inverse).dot(fv[arm].T))
                choose_arm = np.argmax(linucb_idx)
                rs = self.data.random_sample(t, choose_arm)
                A += np.outer(fv[choose_arm], fv[choose_arm])
                A_inverse = np.linalg.inv(A)
                b += rs*fv[choose_arm]
                regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][choose_arm]
        return regret
  
    def CP(self, X, y, t, u):
        '''
        this is the multiscale change point detection algorithm for contextual bandit
        Input: X is the sequence of feature vectors of the choosen arm
               y is the sequence of observed sample reward of the choose arm
               u is the threshold to break, see the original paper for the choice of u
        Output: a boolen variable of whether this algorithm detects a change point
                and the time when the change point is detected, if there is no detected change point, then return -1
        '''
        length = len(y)
        if length<=2*self.d: return False, -1
        
        X = np.array(X)
        y = np.array(y)
        H = X.T.dot(X)
        theta_total = np.linalg.inv(H).dot(X.T).dot(y)
        y_hat = X.dot(theta_total)
        
        X1 = np.array(X[:(self.d-1)])
        A1 = X1.T.dot(X1)
        for s in range(self.d, length-self.d):
            X1 = np.array(X[:s])
            X2 = np.array(X[s:])
            y1 = np.array(y[:s])
            y2 = np.array(y[s:])
            
            A1 += np.outer(np.array(X[s]), np.array(X[s]))
            A2 = H-A1
            y1_hat = X1.dot(np.linalg.inv(A1).dot(X1.T).dot(y1))
            y2_hat = X2.dot(np.linalg.inv(A2).dot(X2.T).dot(y2))
            if np.linalg.norm(y_hat - np.concatenate((y1_hat,y2_hat),axis=0)) >= u:
                return True, s
        return False, -1
    
    def Multiscale_linUCB(self, explore, u, alpha):
        '''
        this is multiscale_linucb algorithm
        input: explore is the exploration rate 
               see the original paper for the choice of u
               alpha is set to sqrt(logT/T) in the original paper
        output: regret is an array of length T, each element is the cumulative regret up to time t
                count is the number of detected change points, not necessarily equal to the true number of change points
        '''
        K = self.data.K
        p = self.data.gamma/self.T
        d = self.data.d
        regret = np.zeros(self.T)
        feature = self.data.fv
        
        A = [np.identity(d) for _ in range(K)]
        b = [np.zeros(d) for _ in range(K)]
        A_inverse = [np.identity(d) for _ in range(K)]
        X_matrix = [[] for _ in range(K)]
        y = [[] for _ in range(K)]
        theta_hat = [None] * K
        linucb_idx = [0] * K
        
        count = 0
        for t in range(self.T):
            fv = feature[t]
            # random sampling
            if np.random.uniform(0,1) <= alpha:   
                choose_arm = np.random.randint(0,K)
            # choose the arm with maximum linucb index
            else:
                for arm in range(K):
                    theta_hat[arm] = A_inverse[arm].dot(b[arm])
                    linucb_idx[arm] = fv[arm].dot(theta_hat[arm]) + explore * math.sqrt(fv[arm].dot(A_inverse[arm]).dot(fv[arm].T))
                choose_arm = np.argmax(linucb_idx)   
            rs = self.data.random_sample(t, choose_arm)
            X_matrix[choose_arm].append(fv[choose_arm])
            y[choose_arm].append(rs)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][choose_arm]
            
            # check whether there is a change point for the choosen arm
            change, change_idx = self.CP(X_matrix[choose_arm], y[choose_arm], t, u)
            if change:
                count += 1
                X_matrix[choose_arm] = X_matrix[choose_arm][change_idx:]
                y[choose_arm] = y[choose_arm][change_idx:]
                A[choose_arm] = np.array(X_matrix[choose_arm]).T.dot(np.array(X_matrix[choose_arm]))
                
                b[choose_arm] = np.zeros(d)
                for i,ye in enumerate(y[choose_arm]):
                    b[choose_arm] += X_matrix[choose_arm][i]*ye
            else:
                A[choose_arm] += np.outer(fv[choose_arm], fv[choose_arm])
                b[choose_arm] += rs*fv[choose_arm]
            A_inverse[choose_arm] = np.linalg.inv(A[choose_arm])
        return regret, count
    
    def Multiscale_joint_linUCB(self, explore, u, alpha):
        '''
        this is multiscale_linucb algorithm
        input: explore is the exploration rate 
               see the original paper for the choice of u
               alpha is set to sqrt(logT/T) in the original paper
        output: regret is an array of length T, each element is the cumulative regret up to time t
                count is the number of detected change points, not necessarily equal to the true number of change points
        '''
        K = self.data.K
        p = self.data.gamma/self.T
        d = self.data.d
        regret = np.zeros(self.T)
        feature = self.data.fv
        
        A = np.identity(d) 
        b = np.zeros(d) 
        A_inverse = np.identity(d) 
        X_matrix = [] 
        y = [] 
        theta_hat = np.zeros(d)
        linucb_idx = [0] * K
        
        count = 0
        for t in range(self.T):
            fv = feature[t]
            # random sampling
            theta_hat = A_inverse.dot(b)
            if np.random.uniform(0,1) <= alpha:   
                choose_arm = np.random.randint(0,K)
            # choose the arm with maximum linucb index
            else:
                for arm in range(K):
                    #explore_ad = math.sqrt(d*math.log(1+t)) + 1
                    linucb_idx[arm] = fv[arm].dot(theta_hat) + explore * math.sqrt(fv[arm].dot(A_inverse).dot(fv[arm].T))
                choose_arm = np.argmax(linucb_idx)   
            rs = self.data.random_sample(t, choose_arm)
            X_matrix.append(fv[choose_arm])
            y.append(rs)
            regret[t] = regret[t-1] + self.data.optimal[t] - self.data.reward[t][choose_arm]
            
            # check whether there is a change point for the choosen arm
            change, change_idx = self.CP(X_matrix, y, t, u)
            if change:
                count += 1
                X_matrix = X_matrix[change_idx:]
                y = y[change_idx:]
                A = np.array(X_matrix).T.dot(np.array(X_matrix))
                
                b = np.zeros(d)
                for i,ye in enumerate(y):
                    b += X_matrix[i]*ye
            else:
                A += np.outer(fv[choose_arm], fv[choose_arm])
                b += rs*fv[choose_arm]
            A_inverse = np.linalg.inv(A)
        return regret, count


