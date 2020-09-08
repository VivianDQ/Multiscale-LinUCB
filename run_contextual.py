import numpy as np
import random
import math
import collections
import time
import argparse
import os
from data_generator import context_random, abrupt2, abrupt_fix
from Multiscale_LinUCB import Algos_context
from DLINUCB import DLinUCB, SW_LinUCB


parser = argparse.ArgumentParser(description='Non-stationary bandit')
parser.add_argument('-s', '--setting', type=str, help = 'specify the setting')
parser.add_argument('-k', '--K', type=int, help = 'number of arms')
parser.add_argument('-d', '--dim', type=int, help = 'dimension of feature vectors')
parser.add_argument('-cp', '--n_cp', type=int, default = 4, help = 'number of change points')

parser.add_argument('-t', '--T', type=int, default = 10000, help = 'total time')
parser.add_argument('-rep', '--rep', type=int, default = 10, help = 'repeat experiments for how many times')
args = parser.parse_args()


def run_contextual(T, K, n_cp, dim, ub_reward, lb_reward, rep, setting):
    if setting == 'dim2':
        n_cp = 4
    print('{}: T={}, K={}, n_cp={}, dim={}, ub_reward={}, lb_reward={}'.format(setting, T, K, n_cp, dim, ub_reward, lb_reward))
    if not os.path.exists('result/'):
        os.mkdir('result/')
    if not os.path.exists('result/contextual/'):
        os.mkdir('result/contextual/')
    path = 'result/contextual/' + 'k' + str(K) + '_cp' + str(n_cp) + '_dim' + str(dim) + '_s' + setting + '/' 
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)
    result = {
        'dis': np.zeros(T),
        'sw': np.zeros(T),
        'joint_multi_linucb': np.zeros(T),
        'joint_linucb': np.zeros(T),
    }
        

    u_prime = 3*math.log(T) + math.log(K)
    u_con = dim + 2*math.sqrt(dim*(u_prime)) + 2*u_prime 
    alpha_con = math.sqrt(math.log(T)/T) 
    explore_con = math.sqrt(math.log(2*K*T**2)/2) 
    window_size = int((dim*T)**(2/3))

    lamda = 1
    mx = 10
    L = math.sqrt(mx**2*dim)
    S = math.sqrt(dim)
    
    seed = 0
    count = 0
    while count<rep:
        np.random.seed(seed)
        t1 = time.time()
        seed += 1
        fv = np.random.uniform(0, mx, (T, K, dim))
        if setting == 'dim2':
            bandit = abrupt2(K, n_cp, lb_reward, ub_reward, T, dim, fv)
            bandit.change()
        elif setting == 'fix':
            bandit = abrupt_fix(K, n_cp, lb_reward, ub_reward, T, dim, fv)
            bandit.change()
            smax = 1000
            smin = 1000
        elif setting == 'random':
            bandit = context_random(K, n_cp, lb_reward, ub_reward, T, dim, fv)
            bandit.change(n_cp/T)
            smax = 1000
            smin = 1000
            if len(bandit.cp)<n_cp:
                continue
        count += 1
        
        # dis
        dis = DLinUCB(bandit,T)
        result['dis'] += dis.dlinucb(lamda, L, S, 1- (1/dim/T)**(2/3))
        
        # sw
        sw = SW_LinUCB(bandit, T)   
        result['sw'] += sw.sw_linucb(lamda, L, S, window_size)

        # LinUCB and Multiscale_LinUCB
        solve = Algos_context(bandit, T) 
        # joint LinUCB and joint Multiscale_LinUCB
        tmp, _ = solve.Multiscale_joint_linUCB(explore_con, u_con, alpha_con)
        result['joint_multi_linucb'] += tmp
        result['joint_linucb'] += solve.linUCB_joint(explore_con)

        print(count, 'jointmulti: {}, dis: {}, linucb: {}, sw: {}'.format(result['joint_multi_linucb'][-1], result['dis'][-1], result['joint_linucb'][-1], result['sw'][-1]))
        
    for key, value in result.items():
        result[key] /= rep
        np.savetxt(path+key, result[key])
    return result

rep = args.rep
T = args.T
K = args.K
dim = args.dim
n_cp = args.n_cp
ub_reward, lb_reward = 1, -1
setting = args.setting # choice of algorithms in {dim2, fix, random}

run_contextual(T, K, n_cp, dim, ub_reward, lb_reward, rep, setting)


