'''
Created on May 1, 2016

@author: aplinge


This module implements the geometry calibration algorithm described in [4] with extensions.

[4] Plinge, A., & Fink, G. A., Gannot, S. (2016). 
    Passive online geometry calibration of acoustic sensor networks

'''

import time
import numpy as np
import multiprocessing as mp
import calau

# calibrator has to be global here for multiprocessing
calibrator = calau.Calibrator()
calibrator.set_room_bounds(-3.0, 3.0, -3.0, 3.0)
calibrator.set_span(180.0)
calibrator.set_ostep(0.5)

def calibrate_set(dataentry):
    [rms, x0, _] = calibrator.find_all_pairs_evo(dataentry['deltas'],dataentry['doas'],None,popsize=dataentry['popsize'])
    [ex1,ey1,ex2,ey2,o1,o2] = x0
    return [np.array([0,0,ex1,ey1,ex2,ey2,0,o1,o2]), rms**2]
    
class OnlineCalibration(object):

    def __init__(self,num_arrays,num_cores=2,num_host=4,pop=25,tol=0.01,iterations=None):
        '''
        Create calibrator object.
        @param num_arrays: number of microphone arrays
        @param num_cores: number of cores per node
        @param num_host: number of cores to use on this machine
        @param pop: population size for evolutionary algorithm
        @param tol: tolerance for  termination of evolutionary algorithm
        @param iterations: iteration limit for evolutionary algorithm      
        '''
        import platform
        # somehow, the multiprocessing does not work under windows?!
        if platform.system() == 'Windows':
            self.num_host = 1
        else:
            self.num_host = num_host
            
        self.num_arrays = num_arrays        
        # number of nodes for distributed processing
        self.num_vnodes = num_arrays
        # number of cores for parallel processing per node
        self.node_cores = num_cores
        self.total_cores = self.num_vnodes*self.node_cores
        self._popsize=pop
        self._tol=tol
        self._maxIter=iterations
        return
    
    def get_popsize(self):
        return  self._popsize
    
    def get_total_vcores(self):
        return self.total_cores
            
    def calibrate_sets(self,datasets,maxtime=None):
        """
        Perform calibration with a new datasets.
        
        @param: datasets: list of new measurement sets        
        @param: maxtime: maximum time available for computation, None for no limit
        """
        if len(datasets)<1:
            return [None,None,0]
            
        print 'running parallel computation on',len(datasets),'sets',
        
        aargs=[]
        for dataentry in datasets:
            setp = dataentry['set']
            print str.join(',',[str(s) for s in setp]),
            entry={}
            entry['deltas']  = dataentry['deltas']
            entry['doas']    = dataentry['doas']
            entry['popsize'] = self._popsize
            aargs.append(entry)
        print
        starttime = time.time()
        if self.num_host>1 and len(aargs)>1:
            pool = mp.Pool(processes=self.num_host)
            ests = pool.map(calibrate_set, aargs)
        else:
            ests = map(calibrate_set, aargs)    

        usedtime = time.time() - starttime 
        num_cores_used = min(len(ests),self.num_host)
        num_cores_virtual = min(len(ests),self.total_cores)
        comptime = usedtime * float(num_cores_used) / num_cores_virtual
        loadtime = usedtime * self.num_host / num_cores_virtual        
        if (maxtime is not None) and (comptime>maxtime):
            print  'new data', len(ests),
            if len(ests)>1:
                ests2=[]
                for (i,x) in enumerate(ests):
                    if (i%self.total_cores) < (self.num_vnodes*(self.node_cores-1)):
                        ests2.append(x)
                ests=ests2
            print '->',len(ests),'ests'
        print '%.2fs walltime /%d = %.2fs on target' % (usedtime, num_cores_virtual/num_cores_used, comptime)        
        return ests, comptime, loadtime
    
    def computemeanestimate(self,allests):
        '''
        compute weighted mean estimate from the given estimates.
        
        @param: allests: current set of estimates
        '''
        
        print 'computing consensus over',len(allests)        
        sumx=np.zeros(3*self.num_arrays,)
        sume=0
        for [x,rms] in allests:
            y = x * 1.0 / (rms+1e-9)
            e = 1.0 / (rms+1e-9)
            sume += e
            sumx += y        
        est = sumx / sume
        rs = est[:2*self.num_arrays].reshape(self.num_arrays,2)
        os = est[2*self.num_arrays:]        
        return [rs,os]
