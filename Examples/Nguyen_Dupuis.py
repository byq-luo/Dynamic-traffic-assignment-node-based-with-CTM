'''
@author: Sai Kiran Mayakuntla
'''

import sys
sys.path.append('../')

import numpy as np
from Codes import *

connections = [[1,2],[6],[10,16],[4,5],[6],[9],[7,8],[9],[11],[12,13],[11],[14,15],[19],[25],[17,18],[19],[23],[20],[23],[20],[21,22],[25],[24],[24],[],[]]
lengths = np.array([2,3,2,2,1,2,2,2,1,2,2,2,1,5,3,2,4,2,2,3,2,2,1,3,1,1])
nlinks = len(lengths)
nsources = 2
nsinks = 2
demands = np.array([[1,1],[1,1]])*1800

Qs = np.array([72]*nlinks)
ws = np.array([0.8]*nlinks)
Ns = np.array([240]*nlinks)

t_star = np.array([60]*nsinks).astype(int)
alpha = np.array([1]*nsinks).astype(float)
beta = np.array([0.5]*nsinks).astype(float)
gamma = np.array([2]*nsinks).astype(float)

Tm = 80
T = 132

connections = [np.array(conn) for conn in connections]
penalty = 10*T    # Penalty

netstr = get_network_structure(connections)
for key,val in netstr.items():
    exec(key + '=val')
    
minTimes = get_minTimes(lengths, sinks, incmat)

lr = 1e-4
savefilename = 'NUE_4'
solve_DUE(savefilename, lr, incmat, T, Tm, penalty, nlinks, lengths, Qs, ws, Ns, sources, sinks, merges, diverges, demands, t_star, alpha, gamma, beta, niter=200)

'''
savefilename = 'NSO_4'
solve_DSO(savefilename, lr, T, Tm, penalty, nlinks, lengths, Qs, ws, Ns, sources, sinks, merges, diverges, demands, deptimechoices, turnchoices, t_star, alpha, gamma, beta, niter=200)
'''
