# -*- coding: utf-8 -*-
import os,sys
import numpy as np
import itertools
import cPickle as pickle
import glob
from argparse import ArgumentParser
sys.path.append('.')
from ap.calauonline import OnlineCalibration
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
    
def runonline(alldata,setsize,popsize,cores,corelim=4,plot=False,name=None):    
    gt = alldata['true_positions']   
    room = alldata['room_dimensions'] 
    cali = OnlineCalibration(num_arrays=len(gt),num_cores=cores,num_host=corelim,pop=popsize)
    sum_vcores = cali.get_total_vcores()    
    doas = alldata['doas']
    tdoas = alldata['tdoas']
    times = alldata['times']        
    allests = []
    sumtime = 0.0    
    lastresulttime=None
    usedsets=[]
    results=[]; 
    if plot:
        global plt
        fig = plt.figure(figsize=(16*0.8,9*0.8))   
        fig.canvas.set_window_title('Online Acoustic Node Geometry Calibration (c) 2016 Axel Plinge, TU Dortmund University')             
        ax = plt.subplot2grid((2,3),(0, 0),rowspan=2, colspan=2, aspect='equal')
        axer = plt.subplot2grid((2,3),(0, 2),)        
        axor = plt.subplot2grid((2,3),(1, 2),)        
        plot_axis(ax,room)    
        rg,og = [a[:2] for a in gt],[a[-1] for a in gt]        
        for r,o in zip(rg,og):
            plot_node(ax,'blue',r,o,zz=20)
        plt.draw()
        plt.pause(0.1)
        
    for index, (measurementstart, measurementtime) in enumerate(times):                          
        print 'got new measurement at %4.2f-%4.2fs' % (measurementstart,measurementtime)
        numutterances = index+1
        #if numutterances < 3:
        if numutterances < setsize:
            print 'waiting for more data..'
            continue            
        if index+1<len(times):
            nexttime = times[index+1][0] + 1.0
        else:
            nexttime = measurementtime + 30.0 
       
        setsz = min(numutterances,setsize)
        sets = itertools.combinations(range(numutterances),setsz)
        sets = list(set([s for s in sets]) - set(usedsets))
        np.random.shuffle(sets)        
        sumtime_this_update = 0.0    
        print 'computing up to',len(sets),'sets of size',setsz,'on',sum_vcores,'core(s)'    
        if lastresulttime is None:
            starttime = measurementtime
        else:
            starttime = max(lastresulttime,measurementtime)        
        oslices = range(0,len(sets),sum_vcores)        
        for offset in oslices:
            thesets = sets[offset:offset+sum_vcores]
            datasets=[]        
            for theset in thesets:
                datasets.append( {'set'   : theset,
                                  'deltas': [ tdoas[i] for i in theset], 
                                  'doas'  : [ doas[i] for i in theset],
                                  } )
            maxtime = nexttime - (sumtime_this_update+starttime)
            ests,comptime,loadtime = cali.calibrate_sets(datasets,maxtime)
            
            if len(ests)<1:
                print 'aborting, all was cancelled. elapsed %.2fs , next measurement is at %.2fs' % (sumtime_this_update+comptime+starttime, nexttime)
                break
             
            allests.extend(ests)
            usedsets.extend([v['set'] for v in datasets])            
            ers,eos = cali.computemeanestimate(allests)
            if ers is None: 
                break
            sumtime += comptime        
            sumtime_this_update += comptime    
            errs, oers, ra, oa = eval_geo(ers,eos,gt)          
            print '  =>  ',
            print '%.2f+-%.2fcm %.2f+-%.2f deg'  % ( np.mean(errs)*100,np.std(errs)*100,np.mean(oers),np.std(oers))
            print 'result at %4.2fs started at %.2fs last measurement at %4.2fs, total computation time %.2fs,' % (sumtime_this_update+starttime, starttime, measurementtime, sumtime),        
            print 'this update (%02d): %.2fs' % (numutterances, comptime)
            print 'realtime factor is %.1f%%' % (100.0*sumtime/(sumtime_this_update+starttime))
            results.append((sumtime_this_update+starttime,np.mean(errs),np.mean(oers)) )
            if plot:
                plot_axis(ax,room)    
                for r,o in zip(rg,og):
                    plot_node(ax,'blue',r,o,zz=20)
                for r,o in zip(ra,oa):
                    plot_node(ax,'red',r,o,zz=22)
                t = int(sumtime_this_update+starttime)
                if name is None:
                    title = ''
                else:
                    title = name+'\n'                    
                title += '%d:%02d' % (t/60, t%60)
                if len(allests)>1:
                    title+= ' mean over %d sets of size %d (out of %d)' % (len(allests),setsz,len(doas))                
                title += '\n%.1f+-%.1f cm %.1f+-%.1f deg' % (np.mean(errs)*100,np.std(errs)*100,np.mean(oers),np.std(oers))
                ax.set_title(title)
                axer.clear()
                axer.set_title('position error')
                axer.bar([e[0] for e in results],[e[1] for e in results],color='red',width=3)                
                axor.clear()
                axor.set_title('orientation error')
                axor.bar([e[0] for e in results],[e[2] for e in results],color='red',width=3)
                plt.tight_layout()
                plt.draw()
                plt.pause(0.1)
                
            if sumtime_this_update+starttime > nexttime:
                break
            
        lastresulttime = sumtime_this_update+starttime        
        # do not keep sets smaller than the desired size
        if setsz < setsize:
            allests=[]
           
    results = np.array(results)
    print
    print 'mean error over run',
    print '%.2f+-%.2fcm %.2f+-%.2f deg'  % ( np.mean(results[:,1])*100, np.std(results[:,1])*100, 
                                             np.mean(results[:,2]),np.std(results[:,2]))    
    print
    
    if plot:
        plt.show()
        
    return np.mean(results[:,1]), np.mean(results[:,2])

  
print """This program demonstrates the method described in:

[4] Plinge, A., & Fink, G. A., Gannot, S. (2016). 
    Passive online geometry calibration of acoustic sensor networks
""" 

parser = ArgumentParser()    
parser.add_argument('--input', type=str, default=glob.glob('../data/input/*.pickle')[0], help='sequence to test')
parser.add_argument('--cores', type=int, default=2, help='number cores per node')
parser.add_argument('--setsize', type=int , default=5, help='number positions per set')
parser.add_argument('--popsize', type=int , default=25, help='population for differential evolution')
parser.add_argument('--corelimit', type=int, default=1, help='cores to use on this machine')
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
 
er, eo = runonline(alldata,
                   setsize = parsed_args['setsize'],
                   popsize = parsed_args['popsize'], 
                   cores   = parsed_args['cores'], 
                   corelim = parsed_args['corelimit'], 
                   plot    = plot,
                   name    = inputname+ (' U=%d' % parsed_args['popsize'])
                   )
