import numpy as  np
import itertools
import math
import sys
import scipy.optimize

'''
This module implements the geometry calibration algorithm described in [1] with extensions.

[1] Plinge, A., & Fink, G. A. (2014). 
    Geometry Calibration of Multiple Microphone Arrays in Highly Reverberant Environments
    In International Workshop on Acoustic Signal Enhancement, Antibes -- Juan les Pins, France.

[3] Plinge, A., & Fink, G. A. (2014). 
    Multi-Speaker Tracking using Multiple Distributed Microphone Arrays. 
    In IEEE International Conference on Acoustics, Speech, and Signal Processing, Florence, Italy.
'''

NO_INTERSECTION_PENALTY = 10.0
NO_INTERSECTION_PENALTY2 = (NO_INTERSECTION_PENALTY**2)

from pointalg import avect, seg_intersect, dist2d, goodness

def findit(func, slices, args):
    '''
    brute force exhaustive search for a minimum value
    
    @param slices: a tuple of slices that gives the search grid
    '''
    mi = None
    xm = None
    nfev = 0
    for x in itertools.product(*slices):
        v = func(x,*args)
        nfev += 1
        if v < mi or xm is None:
            xm = x; mi = v
    return xm, mi, nfev

def listpairs(num_arrays):
    '''
    simple helper function listing all pairs of indices.
    '''
    return [(k,a0,a1) for k,(a0,a1) in enumerate(itertools.combinations(range(num_arrays),2))]

def rms_single_pair(x, *args):    
    '''
    rms for single pair configuration according to [1]    
    @param ds:    numpy array (positions,num_arrays) with the TDoAs 
    @param ts:    numpy array (positions,num_arrays) with the DoAs
    @param index: int zero based index o
    @param pos0:  position of the 0th array [x,y,o]
    '''
    ds = np.array(args[0])
    ts = np.array(args[1])
    index = args[2]
    pos0 = args[3]
    pcnt = len(ds)
    r0 = pos0[:2]
    rm = x[:2]
    o0 = pos0[-1]
    om = x[2]
    error = 0
    #ss = []
    for i in xrange(pcnt):
        t0 = o0+ts[i][0]
        tm = om+ts[i][index]
        s = seg_intersect(r0, avect(t0), rm, avect(tm))
        if s is None:
            error += NO_INTERSECTION_PENALTY2
            #s = line_intersect(r0, avect(t0), rm, avect(tm))
            #d0m = ds[i,index-1]
            #e0m = calc_dist(r0,s) - calc_dist(rm,s)
            #e = 1e2 - (e0m - d0m)
        else:                        
            d0m = ds[i,index-1]
            e0m = dist2d(r0,s) - dist2d(rm,s)
            e = e0m - d0m
            error += e*e
    return np.sqrt(error/float(pcnt))

     
def rms_all_pairs(x, ds, ts, num_arrays, pos0, pen=None):
    '''
    rms for all pair configurations according to [1]    
    @param ds:    numpy array (positions,num_arrays) with the TDoAs 
    @param ts:    numpy array (positions,num_arrays) with the DoAs
    @param num_arrays: int 
    @param pos0:  position of the 0th array [x,y,...,o]
    '''    
    ds = np.array(ds)
    ts = np.array(ts)
    num = num_arrays-1
    
    penalty = NO_INTERSECTION_PENALTY2 if pen is None else pen**2         
    pcnt = len(ds)    
    rs = pos0[:2]
    rs.extend(x[:2*num])
    rs = np.array(rs)
    rs = rs.reshape(-1,2)
    os = [ pos0[-1] ]
    os.extend(x[2*num:2*num + num])
    error = 0
    for i in xrange(pcnt):
        ps = []        
        for _,a0,a1 in listpairs(num_arrays):
            t0 = os[a0]+ts[i][a0]
            t1 = os[a1]+ts[i][a1]
            si = seg_intersect(rs[a0], avect(t0), rs[a1], avect(t1))
            if si is None:
                error += penalty
            else:                                
                ps.append(si)
            
        #if None in ps:
        #    error += 1e2*len(PAIRS[num-1])
        #else:
        if len(ps)>0:
            s = np.mean(ps,0)        
            for index,[_,a0,a1] in enumerate(listpairs(num_arrays)):            
                di = ds[i,index]
                ei = (dist2d(rs[a0],s) - dist2d(rs[a1],s))                
                e = di - ei                
                error += e*e
                      
    return np.sqrt(error/float(pcnt*num_arrays))    

def rms_all_pairs_g(x, *args):    
    ds = np.array(args[0])
    ts = np.array(args[1])
    num_arrays = args[2]
    pos0 = args[3]
    num = num_arrays-1
    pcnt = len(ds)    
    rs = pos0[:2]
    rs.extend(x[:2*num])
    rs = np.array(rs)
    rs = rs.reshape(-1,2)
    os = [ pos0[-1] ]
    os.extend(x[2*num:2*num + num])
    error = 0
    ss = []
    for i in xrange(pcnt):
        ps = []; gs=[]    
        for _,a0,a1 in listpairs(num_arrays):
            t0 = os[a0]+ts[i][a0]
            t1 = os[a1]+ts[i][a1]
            gi = goodness(t0,t1)
            si = seg_intersect(rs[a0], avect(t0), rs[a1], avect(t1))
            if si is None:
                error += NO_INTERSECTION_PENALTY2
            else:                                                
                ps.append(si*gi); gs.append(gi)
                
        if len(ps)>0:
            s = np.sum(ps,axis=0) / np.sum(gs)
            for index,aaa in enumerate(listpairs(num_arrays)):
                _,a0,a1 = aaa
                di = ds[i,index]
                ei = (dist2d(rs[a0],s) - dist2d(rs[a1],s))
                e = di - ei
                error += e*e
    return np.sqrt(error/float(pcnt*num_arrays))    


def sourcepos_single_pair(x,ts,index,pos0):    
    '''
    rms for single pair configuration according to [1]    
    @param ts:    numpy array (positions,num_arrays) with the DoAs
    @param index: int zero based index o
    @param pos0:  position of the 0th array [x,y,o]
    '''
    pcnt = len(ts)
    r0 = pos0[:2]
    rm = x[:2]
    o0 = pos0[-1]
    om = x[2]
    ss = []
    for i in xrange(pcnt):
        t0 = o0+ts[i][0]
        tm = om+ts[i][index]
        s = seg_intersect(r0, avect(t0), rm, avect(tm))            
        #print r0 ,'+k (',o0,'+',ts[i][0],')',
        #print ' x ',
        #print rm ,'+k (',om,'+',ts[i][index],')',
        #print ' => ',s
        ss.append(s)
    return ss
    
def sourcepos_all_pairs(x,ts,num_arrays,pos0,weigth=True):
    '''
    source positions for all pair configuration according to [1]
    using the weighting proposed in [3] 
    @param ts:    numpy array (positions,num_arrays) with the DoAs
    @param count: number of arrays
    @param pos0:  position of the 0th array [x,y,o]
    '''

    num = num_arrays-1
    pcnt = len(ts)    
    rs = pos0[:2]
    rs.extend(x[:2*num])
    rs = np.array(rs)
    rs = rs.reshape(-1,2)
    os = [ pos0[-1] ]
    os.extend(x[2*num:2*num + num])
    ss = []
    for i in xrange(pcnt):
        ps = []; gs=[]    
        for _,a0,a1 in listpairs(num_arrays):
            t0 = os[a0]+ts[i][a0]
            t1 = os[a1]+ts[i][a1]
            if weigth:
                gi = goodness(t0,t1)
            else:
                gi=1.0    
            si = seg_intersect(rs[a0], avect(t0), rs[a1], avect(t1))                
            ps.append(si*gi); gs.append(gi)
        s = np.sum(ps,axis=0) / np.sum(gs)
        ss.append(s)
    return ss


class Calibrator(object):
    
    def __init__(self,r0=[0,0,0],span=180.0):
        '''
        Construct calibrator object.
        
        @param r0: position and orientation of node number 0
        @param span: span of angels to search (180 for full circle)  
        '''
        self.p0 = []
        for v in r0:
            self.p0.append(v)
        self.span = span
        self.xmi=0.1
        self.xma=4
        self.ymi=0.1
        self.yma=4
        self.rstep=0.1
        self.ostep=2
        
    def set_rstep(self,r):
        self.rstep=r

    def set_ostep(self,o):
        self.ostep=o
        
    def set_room_bounds(self,xmi,xma,ymi,yma):
        self.xmi = self.rstep * (int(xmi / self.rstep)-1)
        self.xma = self.rstep * (int(xma / self.rstep)+1)
        self.ymi = self.rstep * (int(ymi / self.rstep)-1)
        self.yma = self.rstep * (int(yma / self.rstep)+1)
    
    def set_span(self, o):
        self.span=o
        
    def find_single_pair_gradient(self,x0,ds,ts,index,dx=None):
        """
        Gradient descent to find configuration, cf. [1]
        
        @param ds:    numpy array (positions,num_arrays) with the TDoAs 
        @param ts:    numpy array (positions,num_arrays) with the DoAs
        @return: fval, solution, nfev
        """
        
        o1 = x0[-1]
        if dx is None:
            bnds = ((self.xmi, self.xma),(self.ymi, self.yma))
        else:
            bnds = ((max(self.xmi,x0[0]-dx),min(x0[0]+dx,self.xma)),(max(self.ymi,x0[1]-dx),min(x0[1]+dx,self.yma)))
        bnds = bnds + ((o1-self.span,o1+self.span),)       
        x0, fval, d = scipy.optimize.fmin_l_bfgs_b(rms_single_pair, np.array(x0), args=(ds,ts,index,self.p0), bounds=bnds, approx_grad=True)
        if d['warnflag']==2:
            print >> sys.stderr, d['task']
        nfev = d['funcalls']
        if nfev<0:
            return None,None                
        return [fval,x0,nfev]  
    
    def find_single_pair_grid(self,x0,ds,ts,index,dx=0.5):
        """
        Grid search to find configuration, cf. [1]
        
        @param ds:    numpy array (positions,num_arrays) with the TDoAs 
        @param ts:    numpy array (positions,num_arrays) with the DoAs
        @return: fval, solution, nfev
        """
        pcnt = len(ds)
        o1 = x0[-1]
        slices = (np.arange(max(self.xmi,x0[0]-dx), min(x0[0]+dx,self.xma), self.rstep), np.arange(max(self.ymi,x0[1]-dx), min(x0[1]+dx,self.yma), self.rstep))
        slices = slices + (np.arange(o1-self.span, o1+self.span, self.ostep),)
        x0, fval,nfev  = findit(rms_single_pair, slices, args=(ds,ts,index,self.p0))
        return [fval,x0,nfev]
               
    def find_single_pair_from(self,x0,ds,ts,index,dx=None,gridsearch=False):
        """
        @param ds:    numpy array (positions,num_arrays) with the TDoAs 
        @param ts:    numpy array (positions,num_arrays) with the DoAs
        @return: fval, solution, nfev
        """        
        if gridsearch:
            fval,x0,nfev  = self.find_single_pair_grid(x0,ds,ts,index,dx)
        fval, x0, nfev2 = self.find_single_pair_gradient(x0,ds,ts,index,dx)                        
        return [fval,x0,nfev+nfev2]  
            
    def find_single_pair(self,ds,ts,index,gradient_descent=True):
        """
        Find configuration of a pair, cf. [1]
        
        @param ds:    numpy array (positions,num_arrays) with the TDoAs 
        @param ts:    numpy array (positions,num_arrays) with the DoAs
        @return: fval, solution, nfev
        """        
        slices = (np.arange(self.xmi, self.xma, self.rstep),np.arange(self.ymi, self.yma, self.rstep))
        slices = slices + (np.arange(-self.span, self.span, self.ostep),)
        x0, fval, nfev  = findit(rms_single_pair, slices, args=(ds,ts,index,self.p0))            
        #slices = (slice(self.xmi,self.xma,self.rstep),slice(self.ymi, self.yma, self.rstep),slice(-self.span, self.span, self.ostep))
        #x0, fval  = scipy.optimize.brute(rms_single_pair, slices, args=(ds,ts,index,self.p0))
        nfev2=0
        if gradient_descent:
            fval, x0, nfev2 = self.find_single_pair_gradient(x0,ds,ts,index)
        return [fval,x0,nfev+nfev2]      
                
    def find_all_pairs(self,x0,ds,ts,num_arrays=None,dx=0.5):
        """
        Find configuration with gradient descent based on x0, cf. [1]
        
        @param ds:    numpy array (positions,num_arrays) with the TDoAs 
        @param ts:    numpy array (positions,num_arrays) with the DoAs
        @return: fval, solution, nfev
        """
        if num_arrays is None:
            num_arrays = np.array(ts).shape[1]
        num = num_arrays-1    
        rs = np.array(x0[:2*num])
        rs = rs.reshape(-1,2)
        os = x0[2*num:2*num + num]
        bnds = ((max(self.xmi,rs[0,0]-dx),min(self.xma,rs[0,0]+dx)),(max(self.ymi,rs[0,1]-dx),min(self.yma,rs[0,1]+dx)))
        for i in range(1,num):
            bnds = bnds + ((max(self.xmi,rs[i,0]-dx),min(self.xma,rs[i,0]+dx)),(max(self.ymi,rs[i,1]-dx),min(self.yma,rs[i,1]+dx)),)
        for i in range(0,num):
            bnds = bnds + ((os[i]-self.span,os[i]+self.span),)
        ars=(ds,ts,num_arrays,self.p0)        
        x0, fval, d = scipy.optimize.fmin_l_bfgs_b(rms_all_pairs, np.array(x0), args=ars, bounds=bnds, approx_grad=True)
        #res = scipy.optimize.minimize(rms_all_pairs, np.array(x0), (ds,ts,num_arrays), method='L-BFGS-B', bounds=bnds, options={'maxiter':100, 'disp':False})
        if d['warnflag']==2:
            print >> sys.stderr, d['task']
        nfev = d['funcalls']
        if nfev<0:
            return None,None,0        
        return [fval, x0, nfev ]
    
    def find_all_pairs_anneal(self,x0,ds,ts,num_arrays=None,heuristic='anneal',dx=0.5,**kwargs):
        """
        Find configuration using simulated annealing.
        
        @param ds:    numpy array (positions,num_arrays) with the TDoAs 
        @param ts:    numpy array (positions,num_arrays) with the DoAs
        @return: fval, solution, nfev
        """
        
        if num_arrays is None:
            num_arrays = np.array(ts).shape[1]
        num = num_arrays-1    
        rs = np.array(x0[:2*num])
        rs = rs.reshape(-1,2)
        os = x0[2*num:2*num + num]
        bnds = ((max(self.xmi,rs[0,0]-dx),min(self.xma,rs[0,0]+dx)),(max(self.ymi,rs[0,1]-dx),min(self.yma,rs[0,1]+dx)))
        for i in range(1,num):
            bnds = bnds + ((max(self.xmi,rs[i,0]-dx),min(self.xma,rs[i,0]+dx)),(max(self.ymi,rs[i,1]-dx),min(self.yma,rs[i,1]+dx)),)
        for i in range(0,num):
            bnds = bnds + ((os[i]-self.span,os[i]+self.span),)
        ars=(ds,ts,num_arrays,self.p0)
     
        res = scipy.optimize.basinhopping(
                                      rms_all_pairs, np.array(x0),
                                      minimizer_kwargs={'method':'L-BFGS-B', 'bounds':bnds, 'args':ars},
                                      **kwargs         
                                      )
        x0 = res.x
        fval = res.fun   
        nfev = res.nfev             
        return [fval, x0, nfev]
    
    
    def find_all_pairs_evo(self,ds,ts,num_arrays=None,x0=None,dx=0.5,**kwargs):
        """
        Find configuration with differential evolution [4].
        
        @param ds:    numpy array (positions,num_arrays) with the TDoAs 
        @param ts:    numpy array (positions,num_arrays) with the DoAs
        @return: fval, solution, nfev
        """            
        if num_arrays is None:
            num_arrays = np.array(ts).shape[1]
        num = num_arrays-1    
        if x0 is None:
            bnds = ((self.xmi,self.xma),(self.ymi,self.yma))
            for i in range(1,num):
                bnds = bnds + ((self.xmi,self.xma),(self.ymi,self.yma),)
            for i in range(0,num):
                bnds = bnds + ((-self.span,self.span),)
        else:
            rs = np.array(x0[:2*num])
            rs = rs.reshape(-1,2)
            os = x0[2*num:2*num + num]
            bnds = ((max(self.xmi,rs[0,0]-dx),min(self.xma,rs[0,0]+dx)),(max(self.ymi,rs[0,1]-dx),min(self.yma,rs[0,1]+dx)))
            for i in range(1,num):
                bnds = bnds + ((max(self.xmi,rs[i,0]-dx),min(self.xma,rs[i,0]+dx)),(max(self.ymi,rs[i,1]-dx),min(self.yma,rs[i,1]+dx)),)
            for i in range(0,num):
                bnds = bnds + ((os[i]-self.span,os[i]+self.span),)
        ars=(ds,ts,num_arrays,self.p0)
        res = scipy.optimize.differential_evolution(rms_all_pairs, bnds, ars, **kwargs)
        x0 = res.x
        fval = res.fun               
        nfev = res.nfev  
        return [fval, x0, nfev]
    
    def split_solution(self,x0,num_arrays):
        num = num_arrays-1    
        rs = self.p0[:2]
        rs.extend( np.array(x0[:2*num]))
        rs = np.array(rs)        
        rs = rs.reshape(-1,2)        
        os = [ self.p0[-1] ]
        os.extend(x0[2*num:2*num + num])
        os = np.array(os)
        return os,rs
#     
#     def pack_solution(self,x0,rms,num_arrays):
#         os,rs = self.split_solution(x0,num_arrays)        
#         #return [np.hstack((rs.flatten(),os)), rms**2]
#         x = [v for v in rs.flatten()]
#         x.extend(os)
#         return [x, rms**2]
    
    def weighted_mean(self,allests,num_arrays):    
        '''
        Compute the weigthed mean of a list of estimates
        @param allests: estimates and their rms as (x1,rms1),(x2,rms2)...
        @param num_arrays: number of nodes
        
        @return: orientations, positions   
        '''
        if len(allests)<1:
            raise RuntimeError('No estimates to calculate mean!')
        sumx=np.zeros(3*num_arrays,)
        sume=0
        for est in allests:
            [x,rms] = est[:2]
            y = x * 1.0 / (rms+1e-9)
            e = 1.0 / (rms+1e-9)
            sume += e
            sumx += y
        if sume<=0.0:
            return None,None
        est = sumx / sume
        rs = est[:2*num_arrays].reshape(num_arrays,2)
        os = est[2*num_arrays:]
        return os,rs    