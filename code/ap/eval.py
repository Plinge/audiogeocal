'''
Created on Aug 5, 2016

@author: aplinge
'''
import numpy as np
from scipy import linalg
import angles

    
def __geo_match(X, Xhat):
    #dim = Xhat.shape[1]
    N = Xhat.shape[0]
    # zero-mean position vectors
    Xm = X - np.mean(X,0)
    Xhatm = Xhat - np.mean(Xhat,0)
    # dispersion matrix
    D = 1/float(N) * np.dot(Xm.T,Xhatm)
    # optimal rotation based on the eigenvectors of the svd of D
    u,w,v = linalg.svd(D)
    rot = np.dot(u,v).T
    tra = np.mean(X - np.dot(Xhat,rot),0)
    return [rot,tra]

def __best_rotation(os,true_geometry):        
    das=[]
    for o,g in zip(os,[v[-1] for v in true_geometry]):
        da = angles.differences(g,o)
        das.append(da)    
    return np.mean(das)    

def eval_geo(rs,os,true_geometry):
    errs=[]; oers=[]
    numarrays = len(true_geometry)
    # minimize e_r by  SVD match
    r_est_ = np.array(rs)
    r_gt   = np.array([ v[:2] for v in  true_geometry ])
    [rot,tra] = __geo_match(r_gt,r_est_)
    rs = np.dot(r_est_,rot) + tra                                 
    for ma in range(numarrays):
        errs.append( np.sqrt((true_geometry[ma][0]-rs[ma][0])**2  + (true_geometry[ma][1]-rs[ma][1])**2) )
        
    # minimize e_o by best rot +  mirror            
    oers1=[]
    do1 = __best_rotation(np.array(os),true_geometry)
    for ma in range(numarrays):        
        oers1.append( angles.difference(true_geometry[ma][-1],os[ma]+do1) )
    oers2=[]
    do2 = __best_rotation(-np.array(os),true_geometry)
    for ma in range(numarrays):
        oers2.append( angles.difference(true_geometry[ma][-1],-os[ma]+do2) )            
    if np.mean(oers1)<np.mean(oers2):
        oers = oers1
        os =  np.array(os) + do1
    else:
        oers = oers2
        os = -np.array(os) + do2
       
    return errs, oers, rs, os 