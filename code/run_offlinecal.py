# -*- coding: utf-8 -*-
import os,sys
import numpy as np
import itertools
import cPickle as pickle
import glob
from argparse import ArgumentParser
from nltk.sem.logic import AllExpression
sys.path.append('.')
from ap.calau import Calibrator
from ap.eval import eval_geo
from ap.pointalg import avect

def plot_axis(ax,room):
    ax.clear()
    ax.grid(True, 'major')    
    ax.set_xlim(room[0])
    ax.set_ylim(room[1])     
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")   
    ax.xaxis.grid(True,'major',linewidth=1,linestyle='-', zorder=4)
    ax.yaxis.grid(True,'major',linewidth=1,linestyle='-', zorder=4)
    

def plot_node(ax,col,ri,oi,wi=5,zz=12):
    r = np.array([ri[0], ri[1]])
    o = avect(oi)*0.1*wi/5.0
    scattered = plt.Circle(r,0.1*wi/5.0,edgecolor=col,facecolor=(1,1,1,0),zorder=zz,clip_on=False,linewidth=wi*0.3)
    ax.add_artist(scattered)
    scattered = plt.Circle(r,0.02*wi/5.0,facecolor=col,zorder=zz,clip_on=False,edgecolor=col,linewidth=0)
    ax.add_artist(scattered)
    scattered = ax.arrow(r[0], r[1], o[0],  o[1], head_width=0.05*wi/5.0, head_length=0.1*wi/5.0, color=col,clip_on=False,zorder=zz)
    ax.add_artist(scattered)
    
def runoffline(alldata,setsize,setlimit,plot=False,name=None,maxrms=0.1,grid=True):    
    gt   = alldata['true_positions']   
    room = alldata['room_dimensions'] 
    cali = Calibrator(gt[0])     
    cali.set_room_bounds(room[0][0],room[0][1],room[1][0],room[1][1])
    doas = alldata['doas']
    tdoas = alldata['tdoas']    
    nodes = len(doas[0])        
    allests = []    
    results = []    
    if plot:
        global plt
        fig = plt.figure(figsize=(16*0.8,9*0.8))   
        fig.canvas.set_window_title('Acoustic Node Geometry Calibration (c) 2016 Axel Plinge, TU Dortmund University')             
        ax = plt.subplot2grid((2,3),(0, 0),rowspan=2, colspan=2, aspect='equal')
        axer = plt.subplot2grid((2,3),(0, 2),)        
        axor = plt.subplot2grid((2,3),(1, 2),)        
        plot_axis(ax,room)    
        rg,og = [a[:2] for a in gt],[a[-1] for a in gt]        
        for r,o in zip(rg,og):
            plot_node(ax,'blue',r,o,zz=20)
        plt.draw()
        plt.pause(0.1)
    
    sets = itertools.combinations(range(len(doas)),setsize)
    sets = [s for s in sets]
    np.random.shuffle(sets)
    sets = sets[:setlimit*4]
     
    for setindex, theset in enumerate(sets):
        ds = np.array([ tdoas[i] for i in theset]) 
        ts = np.array([ doas[i] for i in theset])
        if not grid:
            [rms, x0, _] = cali.find_all_pairs_evo(ds, ts, popsize=25)
            eos, ers = cali.split_solution(x0,nodes)
        else:
            ri=[]
            oi=[]
            for index in range(1,nodes):
                [rms, [x,y,o], _] = cali.find_single_pair(ds, ts, index, gradient_descent=True)
                ri.extend([x,y])
                oi.append(o)
            [rms, x0, _] = cali.find_all_pairs(np.hstack((ri,oi)), ds, ts)    
            eos, ers = cali.split_solution(x0,nodes)
        errs, oers, _, _ = eval_geo(ers,eos,gt)
        print ','.join([str(v) for v in theset]),
        print 'rms %.5f' % rms
        print '=>',
        print '%.2f+-%.2fcm %.2f+-%.2f deg'  % ( np.mean(errs)*100,np.std(errs)*100,np.mean(oers),np.std(oers))

        if rms < maxrms:
            allests.append([np.hstack([ers.flatten(),eos]), rms**2])        
            eos,ers = cali.weighted_mean(allests, nodes)                   
            errs, oers, ra, oa = eval_geo(ers,eos,gt)   
            print 'weighed mean',len(allests),
            print '%.2f+-%.2fcm %.2f+-%.2f deg'  % ( np.mean(errs)*100,np.std(errs)*100,np.mean(oers),np.std(oers))
            results.append( (setindex, np.mean(errs), np.mean(oers)) )
               
            if plot:
                plot_axis(ax,room)    
                for r,o in zip(rg,og):
                    plot_node(ax,'blue',r,o,zz=20)
                for r,o in zip(ra,oa):
                    plot_node(ax,'red',r,o,zz=22)
               
                if name is None:
                    title = ''
                else:
                    title = name+'\n'                    
               
                if len(allests)>1:
                    title+= ' mean over %d sets of size %d (out of %d)' % (len(allests),setsize, len(doas))                
                title += '\n%.1f+-%.1f cm %.1f+-%.1f deg' % (np.mean(errs)*100,np.std(errs)*100,np.mean(oers),np.std(oers))
                ax.set_title(title)
                axer.clear()
                axer.set_title('position error')
                axer.bar([e[0] for e in results],[e[1] for e in results],color='red',width=0.5)                
                axor.clear()
                axor.set_title('orientation error')
                axor.bar([e[0] for e in results],[e[2] for e in results],color='red',width=0.5)
                plt.tight_layout()
                plt.draw()
                plt.pause(0.1)    
                
        if len(allests)>=setlimit:
            break
                   
    if plot:
        plt.show()
    results=np.array(results)
    return np.mean(results[:,1]), np.mean(results[:,2])

print """This program demonstrates the method described in:

[1] Plinge, A., & Fink, G. A. (2014) 
    Geometry Calibration of Multiple Microphone Arrays in Highly Reverberant Environments
    In International Workshop on Acoustic Signal Enhancement, Antibes -- Juan les Pins, France
""" 

parser = ArgumentParser()    
parser.add_argument('--input', type=str, default=glob.glob('../data/input/*.pickle')[0], help='sequence to test')
parser.add_argument('--grid', action='store_const', help='use grid search instead of differential evolution', default=False, const=True)
parser.add_argument('--setsize', type=int , default=6, help='number positions per set')
parser.add_argument('--setlim', type=int , default=32, help='max number of position sets')
parser.add_argument('--nogui', action='store_const', help='do not plot calibration', default=False, const=True)
parsed_args = parser.parse_args(sys.argv[1:])
parsed_args = vars(parsed_args)
plot = not parsed_args['nogui']
if plot:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

with open(parsed_args['input']) as datafile:
    alldata = pickle.load(datafile)
    print 'Using data from',parsed_args['input']
inputname = os.path.basename(parsed_args['input']).split('.')[0]        
 
er, eo = runoffline(alldata,
                   setsize = parsed_args['setsize'],
                   setlimit= parsed_args['setlim'],
                   maxrms  = 0.2,
                   grid    = parsed_args['grid'],
                   plot    = plot,
                   name    = inputname)
