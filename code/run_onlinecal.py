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
    
def update_estimates(gt,cali,
                     allests,
                     comptime,resulttime,measurementtime,
                     results,
                     (name,room,ax,axer,axor)):
    
    ests = []
    for t,est in allests:
        if t<=resulttime:
            ests.append(est) 
    ers,eos = cali.computemeanestimate(ests)    
    errs, oers, ra, oa = eval_geo(ers, eos, gt)
    rg,og = [a[:2] for a in gt],[a[-1] for a in gt]
    print '  =>  ',    
    print '%.1f+-%.1f cm %.1f+-%.1f deg' % (np.mean(errs)*100,np.std(errs)*100,np.mean(oers),np.std(oers))
    results.append((resulttime,np.mean(errs),np.mean(oers)))    
    if plot:
        plot_axis(ax,room)    
        for r,o in zip(rg,og):
            plot_node(ax,'blue',r,o,zz=20)
        for r,o in zip(ra,oa):
            plot_node(ax,'red',r,o,zz=22)
        t = int(resulttime)
        if name is None:
            title = ''
        else:
            title = name+'\n'                    
        title += '%d:%02d' % (t/60, t%60)
        if len(allests)>1:
            title+= ' mean over %d sets' % (len(allests))                
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
    
def runonline(alldata,setsize,popsize,cores,corelim=4,plot=False,name=None):    
    gt = alldata['true_positions']   
    room = alldata['room_dimensions'] 
    cali = OnlineCalibration(num_arrays=len(gt),num_cores=cores,num_host=corelim,pop=popsize)
    sum_vcores = cali.get_total_vcores()
    num_nodes = cali.get_node_count()
        
    doas = alldata['doas']
    tdoas = alldata['tdoas']
    times = alldata['times']        
    allests = []
    
    lastresulttime=None
    usedsets=[]
    results=[]
    totaltime1,totaltime0=0,0
    if plot:
        global plt
        fig = plt.figure(figsize=(16*0.8,9*0.8))   
        fig.canvas.set_window_title('Online Acoustic Node Geometry Calibration (c) 2016 Axel Plinge, TU Dortmund University')             
        ax = plt.subplot2grid((3,3),(0, 0),rowspan=2, colspan=2, aspect='equal')
        axer = plt.subplot2grid((3,3),(0, 2),)        
        axor = plt.subplot2grid((3,3),(1, 2),)
        axcm = plt.subplot2grid((3,3),(2, 0),colspan=3)
        axcm.set_xlim(np.floor(np.min(times)), np.ceil(np.max(times)+50) )
        axcm.set_ylim(0.3,cores+0.7)
        axcm.set_title('computation')        
        plot_axis(ax,room)    
        rg,og = [a[:2] for a in gt],[a[-1] for a in gt]        
        for r,o in zip(rg,og):
            plot_node(ax,'blue',r,o,zz=20)
        plt.draw()
        plt.pause(0.1)
        
    for index, (measurementstart, measurementtime) in enumerate(times):
        print                          
        print 'got new measurement at %4.2f..%4.2fs' % (measurementstart,measurementtime)
        
        
        if plot:
            axcm.add_patch(Rectangle((measurementstart, 1 - .4), measurementtime-measurementstart, 0.8, facecolor="green"))
            plt.pause(0.1)
        
        numutterances = index+1
        if numutterances < setsize:
            print 'waiting for more data..'
            continue            
        if index+1<len(times):
            nexttime = times[index+1][0]
        else:
            nexttime = measurementtime + 30.0       
        setsz = min(numutterances,setsize)
        sets = itertools.combinations(range(numutterances),setsz)
        sets = list(set([s for s in sets]) - set(usedsets))
        np.random.shuffle(sets)        
       
        
        sumtime0 = 0.0; core0=True
        starttime1 = measurementtime if lastresulttime is None else max(lastresulttime,measurementtime)
        sumtime1 = 0.0; core1=True
        resulttime1 = None
        offset=0
        while offset < len(sets):        
            """ 
            emulate core 0, will abort on new data 
            """        
            thesets = sets[offset:offset+num_nodes]
            offset += len(thesets)                  
            datasets=[]        
            for theset in thesets:
                datasets.append( {'set'   : theset,
                                  'deltas': [ tdoas[i] for i in theset], 
                                  'doas'  : [ doas[i] for i in theset],
                                  } )                          
            maxtime = nexttime - (sumtime0+measurementtime)     
            if maxtime < 1.0 or len(datasets)<1:
                core0=False
            if core0:
                ests,comptime,_ = cali.calibrate_sets(datasets,maxtime,corelimit=1)            
                if len(ests)<1:
                    print 'aborting on first cores, next measurement is at %.2fs' % nexttime
                    core0=False           
                    totaltime0 += maxtime     
                    if plot:                                                
                        axcm.add_patch(Rectangle((nexttime - maxtime, 1 - .4), maxtime, 0.8, facecolor="gray"))
                else:                    
                    sumtime0 += comptime
                    totaltime0 += comptime
                    resulttime = sumtime0+measurementtime            
                    for est in ests:
                        allests.append((resulttime,est))
                    usedsets.extend([v['set'] for v in datasets])
                    print '%d result(s) from first cores' % len(ests),                              
                    print 'at %4.2fs started at %.2fs for %.2fs' % (resulttime, resulttime-comptime, measurementtime)
                    if plot:                        
                        axcm.add_patch(Rectangle((resulttime -comptime, 1 - .4), comptime, 0.8, facecolor="yellow"))
                    
            """ do not keep sets smaller than the desired size """
            if setsz < setsize:
                update_estimates(gt,cali,allests,
                     comptime,resulttime,measurementtime,
                     results,
                     (name,room,ax,axer,axor))                 
                allests = []
                break            
      
            """
            emulate other cores, will keep computing
            """
            
            if core1 and (lastresulttime is not None) and (lastresulttime > nexttime):
                print 'other cores still busy with last measurement'
                core1=False
          
            if not core1:
                if core0:
                    update_estimates(gt,cali,allests,
                     measurementtime,resulttime,measurementtime,
                     results,
                     (name,room,ax,axer,axor))
                continue
            
            if starttime1 > nexttime:
                core1=False
                break
                
            thesets = sets[offset:offset+sum_vcores-num_nodes]
            offset += len(thesets)
            datasets=[]        
            for theset in thesets:
                datasets.append( {'set'   : theset,
                                  'deltas': [ tdoas[i] for i in theset], 
                                  'doas'  : [ doas[i] for i in theset],
                                  } )            
            if len(datasets)<1:
                break                       
            ests,comptime,_ = cali.calibrate_sets(datasets,None,corelimit=cores-1)  
                   
            if len(ests)<1:
                print 'no result?'
            else:                        
                usedsets.extend([v['set'] for v in datasets])            
                sumtime1 += comptime
                totaltime1 += comptime
                resulttime1 = sumtime1+starttime1
                if plot:                                    
                    for y in range(1,cores):
                        axcm.add_patch(Rectangle((resulttime1 - comptime, 1+y - .4), comptime, 0.8, facecolor="yellow"))
                    
                for est in ests:
                    allests.append((resulttime,est))
                print '%d result(s) from other cores' % len(ests),
                print 'at %4.2fs started at %.2fs for %.2fs' % (resulttime1, resulttime1-comptime, measurementtime)            
                update_estimates(gt,cali,
                     allests,
                     comptime,resulttime1,measurementtime,
                     results,
                     (name,room,ax,axer,axor))                
                if resulttime1 > nexttime:
                    core1=False
                    break
                
        if resulttime1 is not None:
            lastresulttime = resulttime1
        
    results = np.array(results)
    print
    print 'computational load %.1f%%' % ( (totaltime0+totaltime1)*50.0 / nexttime)
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
parser.add_argument('--input', type=str, default=glob.glob('../data/input/*.pickle')[-1], help='sequence to test')
parser.add_argument('--cores', type=int, default=2, help='number cores per node')
parser.add_argument('--setsize', type=int , default=5, help='number positions per set')
parser.add_argument('--popsize', type=int , default=15, help='population for differential evolution')
parser.add_argument('--corelimit', type=int, default=1, help='cores to use on this machine')
parser.add_argument('--nogui', action='store_const', help='do not plot calibration', default=False, const=True)
parsed_args = parser.parse_args(sys.argv[1:])
parsed_args = vars(parsed_args)
plot = not parsed_args['nogui']
if plot:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    from matplotlib.patches import Rectangle
    
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
