'''
@author: Sai Kiran Mayakuntla
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle

from .utils import *

NPINT = np.int64
NPFLT = np.float64
NPARR = np.ndarray

class Link:
    linkno = 0
    def __init__(self, T, length, nsinks, Q, w, N, isSink=False):
        self.no = Link.linkno
        Link.linkno += 1
        self.T = T
        self.length = length
        self.nsinks = nsinks
        self.Q = Q
        self.w = w
        self.N = N
        self.isSink = isSink
        
        self.occs = np.zeros((T+1,length))
        self.flows = np.zeros((T,length+1))
        self.entflows_full = np.zeros((T,nsinks))
        self.extflows_full = np.zeros((T,nsinks))
        self.cumflows = np.zeros((T,length+1))
        self.entfracs = np.ones((T, nsinks))/nsinks # since their rowsums must be equal to 1
        
        # (times, length)
        self.A = np.stack([np.concatenate((np.ones(c)*-1, np.arange(T-c)), axis=0).astype(int) for c in np.arange(1,length+1)], axis=1)
        self.a = np.stack([np.concatenate((np.arange(c,T), np.ones(c)*T), axis=0).astype(int) for c in np.arange(1,length+1)], axis=1)
        
        self.t = 0
    
    def calc_midflows(self): # calculates flows between intermediate cells (i.e., non-exit & non-entry) of a link
        co = self.occs[self.t,:]
        if len(co)<=1:
            return np.array([])
        midflows = np.minimum(np.minimum(co[:-1],self.Q),self.w*(self.N-co[1:]))
        # midflows.shape == (self.length-1,)
        return midflows
    
    def update_occs(self, flows):
        assert flows.shape == (self.length+1,)
        new_occs = self.occs[self.t,:]+flows[:-1]-flows[1:]
        self.occs[self.t+1,:] = new_occs
        assert np.all((self.occs[self.t+1, :]>0) | np.isclose(self.occs[self.t+1, :], 0))
    
    def update(self, entflows, extflows):
        assert entflows.shape == extflows.shape == (self.nsinks,)
        
        midflows = self.calc_midflows()
        flows = np.insert(midflows, 0, np.sum(entflows), 0)
        flows = np.insert(flows, len(flows), np.sum(extflows))
        self.update_occs(flows)
        self.flows[self.t,:] = flows
        self.entflows_full[self.t,:] = entflows
        self.extflows_full[self.t,:] = extflows
        
        if self.t==0:
            self.cumflows[0,:] = flows
        else:
            self.cumflows[self.t,:] = self.cumflows[self.t-1,:]+flows
        
        if flows[0]>0:
            self.entfracs[self.t,:] = entflows/flows[0]
        
        # Now A @ time (t+1) will be determined, since it only depends on cumflows until time (t)
        for c in np.arange(1, self.length+1):
            if self.t>=c: # c is the min time to exit cell c
                yex = self.cumflows[self.t,c]
                if ~np.isclose(yex, self.cumflows[self.t-c,0]): # NOT free-flowing
                    tmpA = self.A[self.t, c-1] # A's are non-decreasing
                    while (yex>self.cumflows[tmpA,0] or np.isclose(yex,self.cumflows[tmpA,0])) and tmpA<self.t+1-c: # conditions
                        tmpA += 1
                    
                    # Note that atleast one of the conditions is violated now; so, no need to add +1, as given in the formula
                    if self.t<self.T-1: # Vehicles entering @ (T-1) can't exit!
                        self.A[self.t+1,c-1] = tmpA
        self.t += 1
    
    def determine_a(self):
        assert self.t == self.T # Can only start after the simulation, when all A's are determined
        for c in np.arange(1,self.length+1):
            for tau in np.arange(self.T-c): # c is the min time to exit cell c
                if self.A[tau+c,c-1]==tau: # free-flow
                    self.a[tau,c-1] = tau+c
                else:
                    tmpa = self.a[tau-1,c-1] # a's are non-decreasing
                    if tmpa>self.T-2: # a can't be (T-1), checking if tmpa==T
                        self.a[tau,c-1] = self.T
                    else:
                        while tau>self.A[tmpa+1,c-1]: # (or) until tau <= self.A[tmpa+1,c-1] where tmpa in {(c-1),...,(T-2)}
                            tmpa+=1
                            if tmpa==self.T-1: # vehicles entering @ (T-1) can't exit!
                                tmpa = self.T
                                break
                        self.a[tau,c-1] = tmpa
    
    def get_disagg_flows(self, flow, cellno):
        #assert type(a[-1])==np.float64
        if cellno < 0: # negative indexing
            cellno += self.length+1
        assert 1 <= cellno <= self.length
        
        # initializing with zeros
        y = np.zeros((self.nsinks,))
        # zero outflow from the cell
        if np.isclose(flow,0.):
            return y
        
        # here flow != 0
        t_tmp = self.A[self.t,cellno-1] # earliest vehicles exiting the cell @ self.t come from self.A[self.t,cellno-1]
        # since flow != 0, t_tmp != -1
        
        cf_1 = self.cumflows[self.t-1, cellno] # cumflow at the beginning of t; note that self.cumflows.shape == (T, length+1)
        if (cf_1+flow < self.cumflows[t_tmp,0]) or np.isclose(cf_1+flow, self.cumflows[t_tmp,0]): # np.isclose used cuz '==' doesn't work with float values
            # must happen for free-flow (np.isclose); and can for congested (<)
            y = self.entfracs[t_tmp,:]*flow
        
        else:
            # here, flow != 0 & not free-flow
            # and flows[t_tmp, 0] > 0, cuz t_tmp = self.A[self.t,cellno-1]
            y += self.entfracs[t_tmp,:]*(self.cumflows[t_tmp,0]-cf_1)
            
            # starting iteration
            t_tmp += 1
            while ((cf_1+flow > self.cumflows[t_tmp,0]) or np.isclose(cf_1+flow, self.cumflows[t_tmp,0])) and (t_tmp <= self.t-cellno): # refer to self.A[t+1,:] determination in update()
                y += self.entflows_full[t_tmp,:]
                t_tmp += 1
            
            # here, self.cumflows[t_tmp,0] > (cf_1+flow); so the remaining flow comes from the cohort of t_tmp
            y += (flow - np.sum(y))*self.entfracs[t_tmp,:]
        
        # validity check
        assert np.isclose(np.sum(y), flow)
        
        # Correction for floating point errors
        tmp = np.maximum(y,0)
        y = tmp*(flow/np.sum(tmp))
        
        return y

class Network:
    def __init__(self, T, Tm, penalty, nlinks, lengths, Qs, ws, Ns, sources, sinks, merges, diverges, demands, deptimechoices, turnchoices, t_star, alpha, gamma, beta):
        self.T = T
        self.Tm =Tm
        self.penalty = penalty
        self.nlinks = nlinks
        self.lengths = lengths
        self.Qs = Qs
        self.ws = ws
        self.Ns = Ns
        self.sources = sources
        self.sinks = sinks
        self.merges = merges
        self.diverges = diverges
        self.demands = demands
        self.deptimechoices = deptimechoices
        self.turnchoices = turnchoices
        self.t_star = t_star
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        
        self.nsinks = sinks.shape[0]
        self.nsources = sources.shape[0]
        
        # Checks
        assert len(lengths) == len(Qs) == len(ws) == len(Ns)
        assert demands.shape == (len(sources), len(sinks))
        assert deptimechoices.shape == (T, len(sources), len(sinks))
        assert len(turnchoices) == len(diverges)
        for div,tc in zip(diverges, turnchoices):
            assert tc.shape == (T, len(div[1]), len(sinks))
            assert len(div)==2
            assert (type(div[0]) == int) or (type(div[0]) == NPINT)
        
        # derived variables
        self.nsources = len(sources)
        self.nsinks = len(sinks)
        self.nmerges = len(merges)
        self.ndiverges = len(diverges)
        self.srcflows = demands*deptimechoices # (T, nsources, nsinks)
        self.links = [Link(T, lengths[i], self.nsinks, Qs[i], ws[i], Ns[i], isSink = (i in self.sinks)) for i in np.arange(nlinks)]
        
        # initializing travel costs (TC), total system costs (TSC) & their derivatives
        self.TCs = np.zeros((T, nlinks, self.nsinks), dtype=NPFLT)
        self.TSC_drv = [np.zeros((T+1, lengths[i], self.nsinks)) for i in np.arange(nlinks)]
        self.TSC_drv_dtc = np.zeros_like(deptimechoices)
        self.TSC_drv_tc = [np.zeros_like(tc) for tc in turnchoices]
        self.TSC_drv_tc_proxy = [np.zeros_like(tc) for tc in turnchoices]
        
        # initializing time step
        self.t = 0
    
    def get_div_flows(self, divid):
        # divid -- diverge id
        div = self.diverges[divid]
        inid = div[0]
        outids = div[1]
        inlink = self.links[inid] # Incoming link
        outlinks = [self.links[outid] for outid in outids] # Outgoing links
        
        # Calculating all supplies
        R = np.array([np.min((outlink.Q, outlink.w*(outlink.N - outlink.occs[self.t,0]))) for outlink in outlinks])
        
        if self.t < inlink.length: # min time to travel a link == its length; so, zero flows till then
            return np.zeros((len(outids), self.nsinks))
        
        A_t = inlink.A[self.t, -1]
        cf_right = inlink.cumflows[self.t-1, -1] # cf @ right boundary of the div cell
        cf_left = cf_right + inlink.occs[self.t, -1] # == inlink.cumflows[self.t-1, -2]; cf @ left boundary of the div cell
        tmpA = A_t
        while (tmpA <= (self.t - inlink.length)) and ((cf_left > inlink.cumflows[tmpA, 0]) or np.isclose(cf_left, inlink.cumflows[tmpA, 0])):
            # 1st condition takes care of free-flow cases
            # 2nd & 3rd -- basically checking for A @ (t+1) of the penultimate cell of the div link
            # could be interpreted as the time of entry of the latest vehicles exiting the cell at step t (or) the latest vehicles in the cell at t
            # 3rd condition is probably non-critical always, as free-flow is already taken care of
            tmpA += 1
        
        S = np.zeros_like(outids, dtype=float)
        total_outflow = 0
        for iterA in np.arange(A_t, tmpA+1):
            if A_t == tmpA: # entire occupancy of the div cell at t entered @ A_t
                x = inlink.occs[self.t, -1]
            elif iterA == A_t:
                x = inlink.cumflows[A_t,0] - cf_right
            elif iterA < tmpA:
                x = inlink.flows[iterA, 0]
            else: # iterA == tmpA
                x = cf_left - inlink.cumflows[tmpA-1,0]
            
            if np.isclose(x, 0.):
                f = np.zeros_like(outids)
                continue
            else:
                f = x*np.sum((self.turnchoices[divid][self.t,:,:])*inlink.entfracs[iterA, :],axis=-1) # summed over the sink axis
                assert np.isclose(np.sum(f), x)
            
            if np.all((S+f < R) | np.isclose(S+f, R)) and ((np.sum(S+f) < inlink.Q) | np.isclose(np.sum(S+f),inlink.Q)): # free-flow
                total_outflow += x
                S += f
            else:
                # could happen if one or either of the conditions violated
                # only a fraction of f is moved
                tmpfrac1 = 1. # placeholder
                tmpfrac2 = 1.
                
                mask = ~np.isclose(f, 0.)
                if np.sum(mask)>0:
                    if np.any((S+f)[mask] > R[mask]):
                        tmpfrac1 = min(tmpfrac1, np.min((R-S)[mask]/f[mask]))
                if (np.sum(S+f) > inlink.Q) and (~np.isclose(np.sum(S+f), inlink.Q)):
                    tmpfrac2 = min((tmpfrac2, ((inlink.Q-np.sum(S))/np.sum(f))))
                    
                if tmpfrac2 < tmpfrac1:
                    total_outflow = inlink.Q
                else:
                    total_outflow += np.sum(f*tmpfrac1)
                    
                #assert (tmpfrac > 0.) or np.isclose(tmpfrac, 0.)
                #assert (tmpfrac < 1.) or np.isclose(tmpfrac, 1.)
                #S += tmpfrac*f
                break # since vehicles from iterA couldn't fully exit the div cell
            
        y = inlink.get_disagg_flows(total_outflow, -1)*self.turnchoices[divid][self.t, :, :]
        
        # verification
        assert (np.sum(y) > 0.) or np.isclose(np.sum(y), 0.)
        assert (np.isclose(np.sum(y), min(inlink.Q, inlink.occs[self.t, -1])) or np.any(np.isclose(np.sum(y, axis=-1), R)) or np.isclose(np.sum(y), 0.))
            
        return y
    
    def get_mrg_flows(self, mrgid):
        # mrgid -- merge id
        mrg = self.merges[mrgid]
        outid = mrg[0]
        inids = mrg[1]
        outlink = self.links[outid]
        inlinks = [self.links[inid] for inid in inids]
        
        # calculating demands
        S = np.array([np.min((inlink.occs[self.t, -1], inlink.Q)) for inlink in inlinks])
        
        # initial supply assignment
        R_init = (self.Qs[inids]/np.sum(self.Qs[inids]))*np.min((outlink.Q, outlink.w*(outlink.N - outlink.occs[self.t, 0])))
        if (np.sum(S) < np.sum(R_init)) or np.isclose(np.sum(S), np.sum(R_init)): # free-flow
            y_total = S
        else:
            y_total = np.zeros_like(inids, dtype=float)
            tmpS = S.copy()
            tmpR = R_init.copy()
            while ~np.isclose(np.sum(tmpS*tmpR), 0):
                tmpy = np.minimum(tmpS, tmpR)
                tmpS -= tmpy
                tmpR -= tmpy
                y_total += tmpy
                
                moved_mask = np.isclose(tmpS, 0.)
                surplus = np.sum(tmpR[moved_mask])
                tmpR[~moved_mask] = (self.Qs[inids[~moved_mask]]/np.sum(self.Qs[inids[~moved_mask]]))*surplus
                tmpR[moved_mask] = 0.
        
        y = np.zeros((len(inids), self.nsinks))
        for iterid, inid in enumerate(inids):
            y[iterid, :] = self.links[inid].get_disagg_flows(y_total[iterid], -1)
        
        assert np.all((y_total < S) | np.isclose(y_total, S))
        assert (np.sum(y_total) < np.sum(R_init)) or np.isclose(np.sum(y_total), np.sum(R_init))
        return y
    
    def calc_flows(self):
        # sends only disaggregated flows
        tsrcflows = self.srcflows[self.t, :, :] # (nsources, nsinks)
        divflows = [self.get_div_flows(divid) for divid in np.arange(self.ndiverges)]
        mrgflows = [self.get_mrg_flows(mrgid) for mrgid in np.arange(self.nmerges)]
        snkflows_total = [self.links[snkid].occs[self.t, -1] for snkid in self.sinks] # since the flow and space capacities of a sink are inf., all occ from the upstream cell moved
        
        # disaggregated entry flow variable; computed later
        entflows = [None]*self.nlinks
        extflows = [None]*self.nlinks
        
        for srcid,src in enumerate(self.sources):
            entflows[src] = tsrcflows[srcid]
            
        for divid, div in enumerate(self.diverges):
            extflows[div[0]] = np.sum(divflows[divid], axis=0)
            for outid, outlnkid in enumerate(div[1]):
                entflows[outlnkid] = divflows[divid][outid, :]
                
        for mrgid, mrg in enumerate(self.merges):
            entflows[mrg[0]] = np.sum(mrgflows[mrgid], axis=0)
            for inid, inlnkid in enumerate(mrg[1]):
                extflows[inlnkid] = mrgflows[mrgid][inid, :]
                
        for snkid, snk in enumerate(self.sinks):
            extflows[snk] = self.links[snk].get_disagg_flows(snkflows_total[snkid], -1)
        
        # returning aggregate disaggregated entry flows, & disaggregated exit flows
        return (entflows, extflows)
    
    def run(self, problem='both'):
        while self.t < self.T:
            enflows, extflows = self.calc_flows()
            for lnkid, lnk in enumerate(self.links):
                lnk.update(enflows[lnkid], extflows[lnkid])
            self.t += 1
        
        for lnk in self.links:
            lnk.determine_a()
        
        if problem=='DUE':
            self.calc_TCs()
        elif problem=='DSO':
            self.calc_TSC_drvs()
        elif problem=='both':
            self.calc_TCs()
            self.calc_TSC_drvs()
        elif problem=='none':
            pass
        else:
            raise 'only DUE or DSO problems solved'
            
    def calc_TCs(self):
        # boundary conditions for the recursive computation
        ## end of simulation
        self.TCs[self.T-1, :, :] = self.gamma*(self.penalty+self.T-self.t_star) + self.alpha*(self.penalty+1) # Cuz minimum of 1 timestep to exit the link (link.length>=1)
        
        ## SinkExitCosts -- to be used in get_extTCs
        sinkExitCosts = np.zeros((self.T, self.nsinks, self.nsinks))
        for snkid, snk in enumerate(self.sinks):
            tmptimes = np.arange(self.T)[:, np.newaxis] # (T, 1)
            sinkExitCosts[:, snkid, :] = self.gamma*(self.penalty+self.T-self.t_star) + self.alpha*(self.penalty+self.T-tmptimes)
            # overwriting costs for the current sink
            sinkExitCosts[:, snkid, snkid] = ((tmptimes < self.t_star[snkid])*self.beta[snkid]*(self.t_star[snkid]-tmptimes) + (self.t_star[snkid] < tmptimes)*self.gamma[snkid]*(tmptimes - self.t_star[snkid])).reshape((-1,)) # reshape to match the array shape
        # boundary conditions -- done!!
        
        def get_ext_details(timestep): # TCs for vehicles exiting @ timestep
            extTCs = [None]*self.nlinks
            a_bounds = [(lnk.a[tau, -1], lnk.a[tau+1, -1]) for lnkid, lnk in enumerate(self.links)]
            
            for snkid, snk in enumerate(self.sinks):
                if a_bounds[snk][0] == self.T:
                    extTCs[snk] = (self.gamma*(self.penalty+self.T-self.t_star) + self.alpha*self.penalty)[np.newaxis, :]
                elif a_bounds[snk][0] == a_bounds[snk][1]:
                    extTCs[snk] = (sinkExitCosts[a_bounds[snk][0], snkid, :])[np.newaxis, :]
                    
                elif a_bounds[snk][1] == self.T:
                    extTCs[snk] = sinkExitCosts[a_bounds[snk][0]:self.T, snkid, :]
                    extTCs[snk] = np.concatenate((extTCs[snk], (self.gamma*(self.penalty+self.T-self.t_star) + self.alpha*self.penalty)[np.newaxis,:]), axis=0)
                    
                else:
                    extTCs[snk] = sinkExitCosts[a_bounds[snk][0]:(a_bounds[snk][1]+1), snkid, :]
                
            for divid, div in enumerate(self.diverges):
                inid = div[0]
                outids = div[1]
                if a_bounds[inid][0] == self.T:
                    extTCs[inid] = (self.gamma*(self.penalty+self.T-self.t_star) + self.alpha*self.penalty)[np.newaxis, :]
                    
                elif a_bounds[inid][0] == a_bounds[inid][1]:
                    extTCs[inid] = np.sum(self.turnchoices[divid][a_bounds[inid][0], :, :]*self.TCs[a_bounds[inid][0], outids, :], axis=-2)[np.newaxis, :]
                    
                elif a_bounds[inid][1] == self.T:
                    extTCs[inid] = np.sum(self.turnchoices[divid][a_bounds[inid][0]:self.T, :, :]*self.TCs[a_bounds[inid][0]:self.T, outids, :], axis=-2)
                    extTCs[inid] = np.concatenate((extTCs[inid], (self.gamma*(self.penalty+self.T-self.t_star) + self.alpha*self.penalty)[np.newaxis, :]), axis=0)
                else:
                    extTCs[inid] = np.sum(self.turnchoices[divid][a_bounds[inid][0]:(a_bounds[inid][1]+1), :, :]*self.TCs[a_bounds[inid][0]:(a_bounds[inid][1]+1), outids, :], axis=-2)
            
            for mrgid, mrg in enumerate(self.merges):
                outid = mrg[0]
                inids = mrg[1]
                for initer, inid in enumerate(inids):
                    if a_bounds[inid][0] == self.T:
                        extTCs[inid] = (self.gamma*(self.penalty+self.T-self.t_star) + self.alpha*self.penalty)[np.newaxis, :]
                    
                    elif a_bounds[inid][0] == a_bounds[inid][1]:
                        extTCs[inid] = (self.TCs[a_bounds[inid][0], outid, :])[np.newaxis, :]
                    
                    elif a_bounds[inid][1] == self.T:
                        extTCs[inid] = self.TCs[a_bounds[inid][0]:self.T, outid, :]
                        extTCs[inid] = np.concatenate((extTCs[inid], (self.gamma*(self.penalty+self.T-self.t_star) + self.alpha*self.penalty)[np.newaxis, :]), axis=0)
                        
                    else:
                        extTCs[inid] = self.TCs[a_bounds[inid][0]:(a_bounds[inid][1]+1), outid, :]
            
            return (extTCs, a_bounds)
        
        # starting from (T-2), cuz (T-1) is one of the boundary conditions
        for tau in np.arange(self.T-2, -1, -1):
            extTCs, a_bounds = get_ext_details(tau) # gets expected exit costs for the entry flow to each link and the duration of its exit
            
            for lnkid, lnk in enumerate(self.links):
                if a_bounds[lnkid][0] == a_bounds[lnkid][1]: # entire flow exit together; only happens under during congestion
                    self.TCs[tau, lnkid, :] = self.alpha*(a_bounds[lnkid][0] - tau) + extTCs[lnkid][0, :]
                
                elif np.isclose(lnk.flows[tau, 0], 0):
                    assert a_bounds[lnkid][0] == tau+lnk.length # only possible under free-flow, cuz lnk.a[tau, -1] != lnk.a[tau+1, -1]
                    assert a_bounds[lnkid][0] == a_bounds[lnkid][1]-1
                    self.TCs[tau, lnkid, :] = self.alpha*lnk.length + extTCs[lnkid][0, :]
                    
                else:
                    if tau == 0: # cuz cumflow at the beginning of the simulation is 0.
                        cf_t_1 = 0
                    else:
                        cf_t_1 = lnk.cumflows[tau-1, 0]
                        
                    # lnk.flows[tau, 0] != 0; so should not be a problem with it as denominator
                    self.TCs[tau, lnkid, :] = (lnk.cumflows[a_bounds[lnkid][0], -1] - cf_t_1)*(self.alpha*(a_bounds[lnkid][0] - tau) + extTCs[lnkid][0, :])/lnk.flows[tau, 0] # first timestep of exiting
                    self.TCs[tau, lnkid, :] += (lnk.cumflows[tau, 0] - lnk.cumflows[a_bounds[lnkid][1]-1, -1])*((self.alpha*(a_bounds[lnkid][1] - tau)) + extTCs[lnkid][-1, :])/lnk.flows[tau, 0] # final timestep of exiting
                    if a_bounds[lnkid][0]+1 < a_bounds[lnkid][1]: # now, all the exit flows that entirely came from the entry cohort of tau
                        self.TCs[tau, lnkid, :] += np.sum((lnk.flows[(a_bounds[lnkid][0]+1):a_bounds[lnkid][1], -1])[:, np.newaxis]*\
                                                          (self.alpha*(np.arange(a_bounds[lnkid][0]+1, a_bounds[lnkid][1])[:, np.newaxis] - tau) + \
                                                          extTCs[lnkid][1:-1, :]), axis=0)/lnk.flows[tau, 0]
    
    def calc_TSC_drvs(self):
        # First note that unlink TCs, TSC_drv calculated for cell occupancies from times [0, T]
        # boundary conditions
        for lnkid, lnk in enumerate(self.links):
            ## end of the simulation
            self.TSC_drv[lnkid][self.T, :, :] = self.gamma*(self.penalty+self.T-self.t_star)+self.alpha*(self.penalty+1)
            
            ## sink link exits
            if lnkid in self.sinks:
                snkid = int(np.where(self.sinks==lnkid)[0])
                # vehicles in the upstream cell of a sink at time t will move into the sink @ (t+1)
                self.TSC_drv[lnkid][:self.t_star[snkid], -1, snkid]  = self.alpha[snkid] + self.beta[snkid]*(self.t_star[snkid] - (np.arange(self.t_star[snkid])))#+1))
                
                # since vehicles entering this penultimate cell in (T-1) can't exit!!
                self.TSC_drv[lnkid][self.t_star[snkid]:self.T, -1, snkid] = self.alpha[snkid] + self.gamma[snkid]*((np.arange(self.t_star[snkid], self.T)) - self.t_star[snkid])#+1) - t_star[snkid])
                
                snk_mask = np.where(np.arange(self.nsinks) != snkid)[0]
                self.TSC_drv[lnkid][:self.T, -1, snk_mask] = self.alpha[snk_mask] + (self.gamma[snk_mask]*(self.penalty+self.T-self.t_star[snk_mask])) + self.alpha[snk_mask]*(self.penalty+self.T-np.arange(self.T)[:, np.newaxis])
        
        # Starting the iteration from (T-1)
        for tau in np.arange(self.T-1, -1, -1):
            # for midcells on all links
            for lnkid, lnk in enumerate(self.links):
                # for those that can't leave the cell in tau
                self.TSC_drv[lnkid][tau, :(lnk.length-lnk.isSink), :] = self.alpha + self.TSC_drv[lnkid][tau+1, :(lnk.length-lnk.isSink), :]
                
                if lnk.length - lnk.isSink> 1: # For sink links, the derivatives of the exit cells are already computed above
                    # inflow component -- all cells except the exit
                    self.TSC_drv[lnkid][tau, :(-1-lnk.isSink), :] += np.isclose(lnk.flows[tau, 1:(lnk.length-lnk.isSink)], lnk.occs[tau, :(-1-lnk.isSink)])[:, np.newaxis]*(self.TSC_drv[lnkid][tau+1, 1:(lnk.length-lnk.isSink), :] - self.TSC_drv[lnkid][tau+1, :(-1-lnk.isSink), :])
                    # -- remember that flows.shape == (T, length+1) whereas TSC_drv.shape == occs.shape == (T+1, length)
                    
                    # outflow component -- all cells except the entry
                    self.TSC_drv[lnkid][tau, 1:(lnk.length-lnk.isSink), :]  += np.isclose(lnk.flows[tau, :(lnk.length-1-lnk.isSink)], lnk.w*(lnk.N - lnk.occs[tau, 1:(lnk.length-lnk.isSink)]))[:, np.newaxis]*(-lnk.w)*np.sum(lnk.entfracs[lnk.A[np.minimum(tau+1, self.T-1), 0:(lnk.length-lnk.isSink-1)], :]*(self.TSC_drv[lnkid][tau+1, 1:(lnk.length-lnk.isSink), :] - self.TSC_drv[lnkid][tau+1, :(-1-lnk.isSink), :]), axis=-1)[:, np.newaxis]
                    
                    ## A@(tau+1) used as the entrytime of the latest vehicles exiting the cells @ t; correct only when the flow is NOT free-flow; and above flow is not free-flow!!
                    ## Strictly speaking, the equation wrong @ tau = (T-1); since A@T not computed, A@(T-1) used as a proxy
                    
            for divid, div in enumerate(self.diverges):
                inid = div[0]
                outids = div[1]

                inlink = self.links[inid]
                outlinks = [self.links[outid] for outid in outids]

                # outflow from the diverge cell -- >0, when demand is critical (free-flow)
                tmp = np.stack([self.TSC_drv[outid][tau+1, 0, :] for outid in outids], axis=0) - self.TSC_drv[inid][tau+1, -1, :]
                self.TSC_drv[inid][tau, -1, :] += np.isclose(inlink.flows[tau, -1], inlink.occs[tau, -1])*np.sum(self.turnchoices[divid][tau, :, :]*tmp, axis=0) # summed over all outlinks
                
                # inflow to the downstream cells -- >0, when supply of one them is critical; note that the criticality of one of these cells affects all
                for outid, outlink in zip(outids, outlinks):
                    if np.isclose(outlink.w*(outlink.N - outlink.occs[tau, 0]), outlink.flows[tau, 0]):
                        self.TSC_drv[outid][tau, 0, :] += (-outlink.w)*np.sum(inlink.entfracs[inlink.A[np.minimum(tau+1, self.T-1), -1], :]*self.turnchoices[divid][tau, :, :]*tmp)
                        ## cuz the perturbing traffic stays in the same cell; only the traffic entering that cell changes
            
            for mrgid, mrg in enumerate(self.merges):
                outid = mrg[0]
                inids = mrg[1]

                outlink = self.links[outid]
                inlinks = [self.links[inid] for inid in inids]
                
                adjcell_sum = 0
                Q_sum = 0
                for inid, inlink in zip(inids, inlinks):
                    if inlink.flows[tau, -1] < np.minimum(inlink.occs[tau, -1], inlink.Q):
                        adjcell_sum += inlink.Q*np.sum(inlink.entfracs[inlink.A[np.minimum(tau+1, self.T-1), -1], :]*(self.TSC_drv[outid][tau+1, 0, :] - self.TSC_drv[inid][tau+1, -1, :]))
                        Q_sum += inlink.Q
                if ~np.isclose(Q_sum, 0):
                    adjcell_sum /= Q_sum
                
                self.TSC_drv[outid][tau, -1, :] += (-outlink.w)*adjcell_sum*np.isclose(outlink.flows[tau, 0], outlink.w*(outlink.N - outlink.occs[tau, 0]))
                    
                for inid, inlink in zip(inids, inlinks):
                    if np.isclose(inlink.occs[tau, -1], inlink.flows[tau, -1]):
                        self.TSC_drv[inid][tau, -1, :] += (self.TSC_drv[outid][tau+1, 0, :] - self.TSC_drv[inid][tau+1, -1, :]) - adjcell_sum
                        
        for srcid, src in enumerate(self.sources):
            self.TSC_drv_dtc[:, srcid, :] = self.demands[srcid, :]*self.TSC_drv[src][1:, 0, :]
            
        for divid, div in enumerate(self.diverges):
            inid = div[0]
            outids = div[1]
            for nout,outid in enumerate(outids):
                self.TSC_drv_tc[divid][:, nout, :] = self.links[inid].flows[:, -1][:, np.newaxis]*(self.TSC_drv[outid][1:, 0, :] - self.TSC_drv[inid][1:, -1, :])
                self.TSC_drv_tc_proxy[divid][:, nout, :] = self.TSC_drv[outid][1:, 0, :]

def init_choices(T, Tm, nsources, nsinks, diverges, minTimes):
    dtc = np.zeros((T, nsources, nsinks))
    '''#deptimes_ff = (t_star-minTimes[sources,:]).astype(int)
    for i in np.arange(nsources):
        for j in np.arange(nsinks):
            #deptimechoices[deptimes_ff[i,j],i,j] = 1'''
    dtc[Tm-1,...] = 1

    tc = [np.zeros((T,len(d[1]),nsinks)) for d in diverges]
    for did,d in enumerate(diverges):
        minTime = np.min(minTimes[d[1],:],axis=0)
        minpaths = np.isclose(minTimes[d[1],:]-minTime,0.0)

        for i,tmp in enumerate(minpaths.T):
            tc[did][:,:,i] = tmp/np.sum(tmp)
            
    return dtc, tc

def proj_iteration_DUE(net, lr):
    new_deptimechoices = np.zeros_like(net.deptimechoices)
    new_turningchoices = [np.zeros_like(tc) for tc in net.turnchoices]
    
    for snkid, snk in enumerate(net.sinks):
        for srcid, src in enumerate(net.sources):
            tmp = solveqp(net.deptimechoices[:net.Tm, srcid, snkid], net.TCs[:net.Tm, src, snkid], lr=lr)['x']
            tmp[tmp<MIN_PROP_VALUE] = 0
            new_deptimechoices[:net.Tm, srcid, snkid] = tmp/np.sum(tmp)
            assert np.isclose(np.sum(new_deptimechoices[:net.Tm, srcid, snkid]), 1.)

    for divid, div in enumerate(net.diverges):
        for t in np.arange(net.T):
            for snkid, snk in enumerate(net.sinks):
                tmp = solveqp(net.turnchoices[divid][t, :, snkid], net.TCs[t, div[1], snkid], lr=lr)['x']
                tmp[tmp<MIN_PROP_VALUE] = 0
                new_turningchoices[divid][t, :, snkid] = tmp/np.sum(tmp)
                assert np.isclose(np.sum(new_turningchoices[divid][t, :, snkid]), 1.)
    
    return new_deptimechoices, new_turningchoices

def proj_iteration_DSO(net, lr):
    new_deptimechoices = np.zeros_like(net.deptimechoices)
    new_turningchoices = [np.zeros_like(tc) for tc in net.turnchoices]
    
    for snkid, snk in enumerate(net.sinks):
        for srcid, src in enumerate(net.sources):
            tmp = solveqp(net.deptimechoices[:net.Tm, srcid, snkid], net.TSC_drv_dtc[:net.Tm, srcid, snkid]/np.mean(net.demands), lr=lr)['x']
            tmp[tmp<MIN_PROP_VALUE] = 0
            new_deptimechoices[:net.Tm, srcid, snkid] = tmp/np.sum(tmp)
            assert np.isclose(np.sum(new_deptimechoices[:net.Tm, srcid, snkid]), 1.)

    for divid, div in enumerate(net.diverges):
        for t in np.arange(net.T):
            for snkid, snk in enumerate(net.sinks):
                tmp = solveqp(net.turnchoices[divid][t, :, snkid], net.TSC_drv_tc[divid][t, :, snkid]/np.mean(net.demands), lr=lr)['x']
                tmp[tmp<MIN_PROP_VALUE] = 0
                new_turningchoices[divid][t, :, snkid] = tmp/np.sum(tmp)
                assert np.isclose(np.sum(new_turningchoices[divid][t, :, snkid]), 1.)
    
    return new_deptimechoices, new_turningchoices

def solve_DUE(savefilename, lr, incmat, T, Tm, penalty, nlinks, lengths, Qs, ws, Ns, sources, sinks, merges, diverges, demands, t_star, alpha, gamma, beta, niter=200):
    minTimes = get_minTimes(lengths, sinks, incmat)
    nsources = sources.shape[0]
    nsinks = sinks.shape[0]
    deptimechoices, turnchoices = init_choices(T, Tm, nsources, nsinks, diverges, minTimes)
    dtc_stack = [deptimechoices.copy()]
    tc_stack = [turnchoices.copy()]
    net = Network(T, Tm, penalty, nlinks, lengths, Qs, ws, Ns, sources, sinks, merges, diverges, demands, deptimechoices, turnchoices, t_star, alpha, gamma, beta)
    net.run(problem='DUE')
    TC_stack = [net.TCs]
    for iterno in np.arange(niter):
        deptimechoices, turnchoices = proj_iteration_DUE(net, lr)
        dtc_stack.append(deptimechoices)
        tc_stack.append(turnchoices)
        TC_stack.append(net.TCs)

        net = Network(T, Tm, penalty, nlinks, lengths, Qs, ws, Ns, sources, sinks, merges, diverges, demands, deptimechoices, turnchoices, t_star, alpha, gamma, beta)
        net.run(problem='DUE')

        print(iterno)
    
    save2file = {
        'dtc_stacks': dtc_stack,
        'tc_stacks': tc_stack,
        'TC_stacks': TC_stack
    }
    
    with open(savefilename, 'wb') as writefile:
        pickle.dump(save2file, writefile)

def solve_DSO(savefilename, lr, incmat, T, Tm, penalty, nlinks, lengths, Qs, ws, Ns, sources, sinks, merges, diverges, demands, t_star, alpha, gamma, beta, niter=200):
    minTimes = get_minTimes(lengths, sinks, incmat)
    nsources = sources.shape[0]
    nsinks = sinks.shape[0]
    deptimechoices, turnchoices = init_choices(T, Tm, nsources, nsinks, diverges, minTimes)
    dtc_stack = [deptimechoices.copy()]
    tc_stack = [turnchoices.copy()]
    net = Network(T, Tm, penalty, nlinks, lengths, Qs, ws, Ns, sources, sinks, merges, diverges, demands, deptimechoices, turnchoices, t_star, alpha, gamma, beta)
    net.run(problem='both')
    TC_stack = [net.TCs]
    TSC_drv_stack = [net.TSC_drv]
    TSC_drv_dtc_stack = [net.TSC_drv_dtc]
    TSC_drv_tc_stack = [net.TSC_drv_tc]
    
    for iterno in np.arange(niter):
        deptimechoices, turnchoices = proj_iteration_DSO(net, lr)
        dtc_stack.append(deptimechoices)
        tc_stack.append(turnchoices)
        
        net = Network(T, Tm, penalty, nlinks, lengths, Qs, ws, Ns, sources, sinks, merges, diverges, demands, deptimechoices, turnchoices, t_star, alpha, gamma, beta)
        net.run(problem='both')
            
        TC_stack.append(net.TCs)
        TSC_drv_stack.append(net.TSC_drv)
        TSC_drv_dtc_stack.append(net.TSC_drv_dtc)
        TSC_drv_tc_stack.append(net.TSC_drv_tc)

        print(iterno)

    save2file = {
        'dtc_stacks': dtc_stack,
        'tc_stacks': tc_stack,
        'TC_stacks': TC_stack,
        'TSC_drv_stacks': TSC_drv_stack,
        'TSC_drv_dtc_stacks': TSC_drv_dtc_stack,
        'TSC_drv_tc_stacks': TSC_drv_tc_stack
    }
    
    with open(savefilename, 'wb') as writefile:
        pickle.dump(save2file, writefile)

