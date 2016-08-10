# -*- coding: utf-8 -*-
'''
Created on 21 Feb 2013

@author: aplinge

some math for angles
'''

def make360(x):    
    while x < 0.0:
        x = x+360.0
    while x > 360.0:
        x = x-360.0        
    return x
    
def make180(x):
    while x > 180.0:
        x = x-360.0
    while x < -180.0:
        x = x+360.0
    return x

def difference(a,b):
    return abs(make180(abs(a-b)))

def differences(a,b):
    return make180(a-b)