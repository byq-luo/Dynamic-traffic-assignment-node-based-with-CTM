'''
@author: Sai Kiran Mayakuntla
'''

import numpy as np
from scipy.optimize import minimize

# Network representation functions
def get_incmat(connections):
    nlinks = len(connections)
    incmat = np.zeros((nlinks, nlinks))
    for i in np.arange(nlinks):
        for j in connections[i]:
            incmat[i,j] = 1
    return incmat

def get_sources(incmat):
    sources = np.where(np.sum(incmat, axis=0)==0)[0]
    return sources

def get_sinks(incmat):
    sinks = np.where(np.sum(incmat, axis=1)==0)[0]
    return sinks

def get_merges(incmat):
    merge_entry_links = np.where(np.sum(incmat, axis=0)>1)[0]
    merges = []
    for m in merge_entry_links:
        merges.append([m, np.where(incmat[:, m]>0)[0]])
    return merges

def get_diverges(incmat):
    diverge_exit_links = np.where(np.sum(incmat, axis=1)>1)[0]
    diverges = []
    for d in diverge_exit_links:
        diverges.append([d, np.where(incmat[d, :]>0)[0]])
    return diverges

def get_network_structure(connections):
    incmat = get_incmat(connections)
    nlinks = incmat.shape[0]
    sources = get_sources(incmat)
    sinks = get_sinks(incmat)
    merges = get_merges(incmat)
    diverges = get_diverges(incmat)
    
    return {
        'incmat': incmat,
        'nlinks': nlinks,
        'sources': sources,
        'sinks': sinks,
        'merges': merges,
        'diverges': diverges
    }

def get_minTimes(lengths, sinks, incmat):
    assert len(lengths) == incmat.shape[0]
    nlinks = len(lengths)
    minTimes = np.ones((nlinks,len(sinks)))*np.Inf
    for sid,sink in enumerate(sinks):
        minTimes[sink,sid] = lengths[sink]

    for i in np.arange(nlinks):
        for lid in np.arange(nlinks):
            if lid not in sinks:
                minTimes[lid,:] = np.min(lengths[lid]+minTimes[np.where(incmat[lid,:]>0)[0],:],axis=0)
                
    return minTimes

# Optimization functions
MIN_PROP_VALUE = 1e-7
def opt_fn(x, choices, costs, lr):
    return 0.5*np.sum(np.power(x-(choices-lr*costs), 2))

def jac(x, choices, costs, lr):
    return (x-(choices-lr*costs))

def ones_jac(x):
    return np.ones_like(x)

def cons_fn(x):
    return np.sum(x)-1

def solveqp(choices, costs, lr):
    x = choices.copy()
    bounds = [(0,1)]*len(choices)
    constraint = {
        'type': 'eq',
        'fun': cons_fn,
        'jac': ones_jac
    }
    
    new_choices = minimize(opt_fn, x, args=(choices, costs, lr), jac=jac, method='SLSQP', bounds=bounds, constraints=[constraint], options={'ftol':1e-09})
    return new_choices
