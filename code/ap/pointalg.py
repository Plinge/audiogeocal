'''
Created on Nov 24, 2015

@author: aplinge
'''
import numpy as np
import math 

def disteuclidean(x,y):   
    """
    Eucledian distance of two points.
    """
    return np.sqrt(np.sum((x-y)**2))

def dist2d(p1,p2):
    """
    Eucledian distance of two points in 2d.
    """
    return np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))

def diff2d(p1,p2):
    return np.array([ p2[0] - p1[0] , p2[1] - p1[1]])

def calc_angle(s,r):
    """
    angle between vectors.
    """
    return 180.0/np.pi * math.atan2(s[1]-r[1], s[0]-r[0])

def perp(a) :
    """
    perpendicular 2d vector.
    """
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] =  a[0]
    return b

def goodness(a1,a2):
    """
    quality measure of angular intesection, cf [2]
    """
    return abs(math.sin((a1-a2)*math.pi/180.0 ))

def avect(a):
    return np.array( [ np.cos(a*np.pi/180.0), np.sin(a*np.pi/180.0) ] )


def line_intersect(a1, da, b1, db):
    """
    compute intersection of infinetly long lines
    """
    dba = np.array(a1) - np.array(b1)
    da_perpendicular = perp(da)
    num = np.dot(da_perpendicular, dba)
    denom = np.dot(da_perpendicular, db)
    dist_b = (num / denom)
    return dist_b*db + b1
    

def seg_intersect(a1, da, b1, db):
    """
    compute intersection of directed lines.
    @return: intersection point, None if they do not intersect. 
    """
    dba = np.array(a1) - np.array(b1)
    da_perpendicular = perp(da)    
    dist_b = (np.dot(da_perpendicular, dba) / np.dot(da_perpendicular, db))
    if dist_b<0:
        return None
    xx = dist_b*db + b1
    
    dab = np.array(b1) - np.array(a1)
    db_perpendicular = perp(db)    
    dist_a = (np.dot(db_perpendicular, dab) / np.dot(db_perpendicular, da))
    if dist_a<0:
        return None
    #xx == dist_a*da + a1
    return xx

def seg_intersect_dist(a1, da, b1, db):
    """
    compute intersection of directed lines.
    @return: intersection point, None if they do not intersect. 
    """
    dba = np.array(a1) - np.array(b1)
    da_perpendicular = perp(da)    
    dist_b = (np.dot(da_perpendicular, dba) / np.dot(da_perpendicular, db))
    if dist_b<0:
        return None
    xx = dist_b*db + b1
    
    dab = np.array(b1) - np.array(a1)
    db_perpendicular = perp(db)    
    dist_a = (np.dot(db_perpendicular, dab) / np.dot(db_perpendicular, da))
    if dist_a<0:
        return None
    #xx == dist_a*da + a1
    return xx,dist_a,dist_b

