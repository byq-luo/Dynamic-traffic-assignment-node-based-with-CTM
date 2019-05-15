'''
@author: Sai Kiran Mayakuntla
'''

import sys
sys.path.append('../')

import numpy as np
from Codes import *

connections = [[1,2], [3,4], [5], [5], [6], [6], []]
lengths = np.array([2, 1, 1, 1, 1, 2, 1]) # in sink_links, sink cells not counted; assuming at least 1 non-sink cell in each sink link
nlinks = len(lengths)
nsources = 1
nsinks = 1
demands = np.array([[1]])*1800 # (nsources, nsinks)

Qs = np.array([72]*nlinks)
ws = np.array([0.8]*nlinks)
Ns = np.array([240]*nlinks)

t_star = np.array([110]*nsinks)  # len(t_star) == nsinks
alpha = np.array([1]*nsinks)  # len(alpha) == nsinks
beta = np.array([0.5]*nsinks)  # len(beta) == nsinks
gamma = np.array([2]*nsinks)  # len(gamma) == nsinks

Tm = 160
T = 170

connections = [np.array(conn) for conn in connections]
penalty = 10*T    # Penalty

netstr = get_network_structure(connections)
for key,val in netstr.items():
    exec(key + '=val')
    
minTimes = get_minTimes(lengths, sinks, incmat)

lr = 1e-4
savefilename = 'ZUE_4'
solve_DUE(savefilename, lr, incmat, T, Tm, penalty, nlinks, lengths, Qs, ws, Ns, sources, sinks, merges, diverges, demands, t_star, alpha, gamma, beta, niter=200)

'''
savefilename = 'ZSO_4'
solve_DSO(savefilename, lr, incmat, T, Tm, penalty, nlinks, lengths, Qs, ws, Ns, sources, sinks, merges, diverges, demands, t_star, alpha, gamma, beta, niter=200)
'''
