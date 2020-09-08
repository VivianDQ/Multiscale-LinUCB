import numpy as np
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from matplotlib import pylab
import collections
import time
import argparse
import os

def draw_figure(vt = True):
    plot_style = {
            'joint_multi_linucb': ['-', 'red', 'Multiscale-LinUCB'],
            'joint_linucb': ['-.', 'green', 'LinUCB'],
            'dis': [':', 'purple', 'D-LinUCB'],
            'sw': ['--', 'blue', 'SW-LinUCB'],
        }
    plot_prior = {
            'joint_multi_linucb': 2,
            'joint_linucb': 1,
            'dis': 0,
            'sw': 0,
        }
    if not os.path.exists('plots/'):
        os.mkdir('plots/')
    
    root = 'result/'
    cat = os.listdir(root)
    paths = []
    for c in cat:
        if c == '.DS_Store' or '.zip' in c: continue
        folders = os.listdir(root+c)
        for folder in folders:
            if folder == '.DS_Store' or '.zip' in folder: continue
            paths.append(root + c + '/' + folder + '/')
    
    for path in paths:
        folder = path.split('/')[-2]
        name = folder.split('_')
        K = int(name[0][1:])
        n_cp = int(name[1][2:])
        dim = int(name[2][3:])
        setting = name[3][1:]
        if 'contextual' in path:
            title = "Regret for contextual setting (K={}, D={}, dim={})".format(K, n_cp, dim)
            fn = 'contextual_' + 's_' + setting + 'k' + str(K) + '_D' + str(n_cp) + '_dim' + str(dim)
        fig = plot.figure()
        matplotlib.rc('font',family='serif')
        params = {'font.size': 18, 'axes.labelsize': 18, 'font.size': 12, 'legend.fontsize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12, 'axes.formatter.limits':(-8,8)}
        pylab.rcParams.update(params)
        
        keys = os.listdir(path)
        keys = sorted(keys, key=lambda kv: plot_prior[kv], reverse = True)
        leg = []
        ymax = 0
        for key in keys:
            if not vt and key == 'dis': continue
            if key not in plot_style.keys(): continue
            leg += [plot_style[key][-1]]
            data = np.loadtxt(path+key)
            T = len(data)
            ymax = max(ymax, data[-1])
            plot.plot((list(range(T))), data, linestyle = plot_style[key][0], color = plot_style[key][1], linewidth = 2)
        plot.legend((leg), loc='best', fontsize=16, frameon=False)
        plot.xlabel('T')
        plot.ylabel('Cumulative Regret')
        fig.savefig("plots/" + fn + '.pdf', dpi=300, bbox_inches = "tight")

draw_figure()

