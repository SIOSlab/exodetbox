""" Brown2010 Dynamic Completeness Replication
"""
#dynamicCompleteness
import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
import time
from astropy import constants as const
import astropy.units as u
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.planet_star_separation import planet_star_separation
import itertools
import datetime
import re
from matplotlib import colors
import csv
import pickle

#IF USING GPU
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
linalg.init()


#### PLOT BOOL
plotBool = False
if plotBool == True:
    from plotProjectedEllipse import *
folder = './'
PPoutpath = './'

#### Randomly Generate Orbits
folder_load = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
filename = 'HabEx_CBrownKL_PPBrownKL_compSubtype.json'
#filename = 'HabEx_CSAG13_PPSAG13_compSubtypeHighEccen.json'
scriptfile = os.path.join(folder_load,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
PPop = sim.PlanetPopulation
comp = sim.Completeness
TL = sim.TargetList
n = 10**5 #Dean's nice computer can go up to 10**8 what can atuin go up to?
inc, W, w = PPop.gen_angles(n,None)
W = W.to('rad').value
w = w.to('rad').value
#w correction caused in smin smax calcs
wReplacementInds = np.where(np.abs(w-1.5*np.pi)<1e-4)[0]
w[wReplacementInds] = w[wReplacementInds] - 0.001
wReplacementInds = np.where(np.abs(w-0.5*np.pi)<1e-4)[0]
w[wReplacementInds] = w[wReplacementInds] - 0.001
del wReplacementInds
inc = inc.to('rad').value
#inc[np.where(inc>np.pi/2.)[0]] = np.pi - inc[np.where(inc>np.pi/2.)[0]]
sma, e, p, Rp = PPop.gen_plan_params(n)

#### Classify Planets
bini, binj, earthLike = comp.classifyPlanets(Rp, TL, np.arange(len(sma)), sma, e)
sma = sma.to('AU').value
####

#Separations
s_circle = np.ones(len(sma))
dmag = 26. #29.0
dmag_upper = 26. #29.0
IWA_HabEx = 0.075*u.arcsec #taken from a Habex Script in units of mas
IWA2=0.150*u.arcsec #Suggested by dmitry as analahous to WFIRST
OWA_HabEx = 600.*u.arcsec #from the HabEx Standards Team Final Report
s_inner = 10.215*u.pc.to('AU')*IWA_HabEx.to('rad').value
s_outer = 10.215*u.pc.to('AU')*OWA_HabEx.to('rad').value #HIP 29271 is 10.2 pc away

#starMass
starMass = const.M_sun
M_HIP29271 = 1.103 #solar masses from wikipedia

periods = (2.*np.pi*np.sqrt((sma*u.AU)**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value*np.sqrt(1./M_HIP29271)

#Random time past periastron of first observation
tobs1 = np.random.rand(len(periods))*periods*u.year.to('day')

nus, planetIsVisibleBool = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, s_inner, s_outer, dmag_upper, dmag_lower=None) #Calculate planet-star nu edges and visible regions
ts = timeFromTrueAnomaly(nus,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges
dt = ts[:,1:] - ts[:,:-1] #Calculate time region widths
maxIntTime = 0.
gtIntLimit = dt > maxIntTime #Create boolean array for inds
totalVisibleTimePerTarget = np.nansum(np.multiply(np.multiply(dt-maxIntTime,planetIsVisibleBool.astype('int')),gtIntLimit),axis=1) #We subtract the int time from the fraction of observable time
totalCompleteness = np.divide(totalVisibleTimePerTarget,periods*u.year.to('day')) # Fraction of time each planet is visible of its period

ts2 = ts[:,0:8] #cutting out all the nans
planetIsVisibleBool2 = planetIsVisibleBool[:,0:7] #cutting out all the nans

#GPU example
# x = np.asarray(np.random.rand(4, 4), np.float32)
# y = np.asarray(np.random.rand(4, 4), np.float32)
# x_gpu = gpuarray.to_gpu(x)
# y_gpu = gpuarray.to_gpu(y)
# z_gpu = linalg.multiply(x_gpu, y_gpu)
#np.allclose(x*y, z_gpu.get())

def dynamicCompleteness(tssStart,tssEnd,planetIsVisibleBool2,startTimes,tpast_startTimes,periods,planetTypeInds=None, GPU=False):
    """
    TODO ADD SUBTYPE COMPLETENESS CAPABILITY
    Args:
    startTimes (ndarray):
        a nplan by 7 array or nplan by 1 array (less efficient) containing the times of the first observation of the target
    planetTypeInds (ndarray):
        inds of planets we are concerned with, specifically used to narrow down a specific type of planet to calculate dynamic completeness for
    DELETEptypeBool ():
        booleans indicating planet types. None means use all planets
    """
    if not GPU:
        #time00 = time.time()
        # if planetTypeInds is None:
        #     #ptypeBool = np.ones(len(tobs1))
        #     planetTypeInds = np.arange(len(tobs1))
        # if len(periods.shape) == 1:
        #     periods = np.tile(periods,(7,1)).T
        # if len(startTimes.shape) == 1:
        #     startTimes = np.tile(tobs1[planetTypeInds],(7,1)).T
        #DELETEplanetTypeBool = np.tile(ptypeBool,(7,1)).T
        #time01 = time.time()
        #time10 = time.time()
        #planetIsVisibleBool2 = planetIsVisibleBool2[planetTypeInds]#np.multiply(planetIsVisibleBool2,planetTypeBool) #Here we remove the planets that are not the desired type
        #time11 = time.time()

        #Find all planets detectable at startTimes
        #time20 = time.time()
        #DELETEplanetDetectedBools_times = np.multiply(ts2[:,:-1] < startTimes,np.multiply(ts2[:,1:] > startTimes,planetIsVisibleBool2)) #multiply time window bools by planetIsVisibleBool2. For revisit Completeness
        #planetDetectedBools_times = np.logical_and(ts2[planetTypeInds,:-1] < startTimes,np.logical_and(ts2[planetTypeInds,1:] > startTimes,planetIsVisibleBool2)) #multiply time window bools by planetIsVisibleBool2. For revisit Completeness
        planetDetectedBools_times = np.zeros(tssStart.shape,dtype=bool)
        for i in np.arange(tssStart.shape[1]):
            planetDetectedBools_times[:,i] = np.logical_and(tssStart[:,i] < startTimes[:,i],tssEnd[:,i] > startTimes[:,i])#,planetIsVisibleBool2))
        #time21 = time.time()
        #time30 = time.time()
        planetDetectedBools = np.any(planetDetectedBools_times,axis=1)
        #time31 = time.time()
        #time40 = time.time()
        planetNotDetectedBools = np.logical_not(planetDetectedBools) #for dynamic completeness
        #time41 = time.time()
        

        #tpast_startTimes = 50. #in days
        #time50 = time.time()
        tobs2 = np.fmod(startTimes+tpast_startTimes,periods*u.year.to('day'))
        #time51 = time.time()
        #time60 = time.time()
        #DELETEplanetDetectedBools2_times = np.multiply(ts2[:,:-1] < tobs2,np.multiply(ts2[:,1:] > tobs2,planetIsVisibleBool2)) #is the planet visible at this time segment in time 2?
        #planetDetectedBools2_times = np.logical_and(ts2[planetTypeInds,:-1] < tobs2,np.logical_and(ts2[planetTypeInds,1:] > tobs2,planetIsVisibleBool2)) #is the planet visible at this time segment in time 2?
        planetDetectedBools2_times = np.logical_and(tssStart < tobs2,tssEnd > tobs2)#,planetIsVisibleBool2)) #is the planet visible at this time segment in time 2?
        #time61 = time.time()
        #time70 = time.time()
        planetDetectedBools2 = np.any(planetDetectedBools2_times,axis=1)#.astype(bool)
        #time71 = time.time()
        #time80 = time.time()
        #UnusedplanetNotDetectedBools2 = np.logical_not(planetDetectedBools2) #for dynamic completeness, the planet is not visible in this time segment at time 2
        #time81 = time.time()

        #Undected Twice

        #Revisit Comp.
        #time90 = time.time()
        planetDetectedthenDetected = np.multiply(planetDetectedBools,planetDetectedBools2) #each planet detected at time 1 and time 2 #planets detected and still in visible region    
        #time91 = time.time()

        #Dynamic Comp.
        #time100 = time.time()
        planetNotDetectedThenDetected = np.multiply(planetNotDetectedBools,planetDetectedBools2) #each planet NOT detected at time 1 and detected at time 2 #planet not detected and now in visible region
        #time101 = time.time()

        #time110 = time.time()
        dynComp = np.sum(planetNotDetectedThenDetected)/len(planetNotDetectedThenDetected) #divide by all planets
        #time111 = time.time()
        #dynComp = np.sum(planetNotDetectedBools)/np.sum(planetTypeBool) #divide by all planets of type
        #time120 = time.time()
        revisitComp = np.sum(planetDetectedthenDetected)/np.sum(planetDetectedBools) #divide by all planetes detected at startTimes
        #time121 = time.time()

    else: #GPU is true
        #ts2,tobs1,tpast_startTimes,periods
        planetIsVisibleBool2_gpu = gpuarray.to_gpu(planetIsVisibleBool2)
        # x_gpu = gpuarray.to_gpu(x)
        # y_gpu = gpuarray.to_gpu(y)
        # z_gpu = linalg.multiply(x_gpu, y_gpu)
        time00 = time.time()
        if ptypeBool is None:
            ptypeBool = np.ones(len(tobs1))
        planetTypeBool = np.tile(ptypeBool,(7,1)).T
        ptypeBool_gpu = gpuarray.to_gpu(planetTypeBool)
        time01 = time.time()
        time10 = time.time()
        planetIsVisibleBool2_gpu = linalg.multiply(planetIsVisibleBool2_gpu,planetTypeBool_gpu) #Here we remove the planets that are not the desired type
        time11 = time.time()

        #Find all planets detectable at startTimes
        # startTime = 100. #startTime approach to planet visibility
        # startTimes = np.tile(np.mod(np.tile(startTime,(len(periods),1)),periods),(8,1)).T #startTime into properly sized array
        time20 = time.time()
        startTimes_gpu = gpuarray.to_gpu(np.tile(tobs1,(7,1)).T) #startTime into properly sized array
        time21 = time.time()
        time30 = time.time()
        #DELETEplanetDetectedBools_times = np.multiply(ts2[:,:-1] < startTimes,np.multiply(ts2[:,1:] > startTimes,planetIsVisibleBool2)) #multiply time window bools by planetIsVisibleBool2. For revisit Completeness
        #planetDetectedBools_times = np.multiply(ts2[:,:-1] < startTimes,np.multiply(ts2[:,1:] > startTimes,planetIsVisibleBool2)) #multiply time window bools by planetIsVisibleBool2. For revisit Completeness
        ts2_gpu = gpuarray.to_gpu(ts2)
        planetDetectedBools_times_gpu = linalg.multiply(ts2_gpu[:,:-1] < startTimes_gpu,linalg.multiply(ts2_gpu[:,1:] > startTimes_gpu,planetIsVisibleBool2_gpu)) #multiply time window bools by planetIsVisibleBool2. For revisit Completeness
        time31 = time.time()
        time40 = time.time()
        planetDetectedBools = np.nansum(planetDetectedBools_times,axis=1)
        time41 = time.time()
        time50 = time.time()
        planetNotDetectedBools = np.logical_not(planetDetectedBools) #for dynamic completeness
        time51 = time.time()
        

        #tpast_startTimes = 50. #in days
        time60 = time.time()
        tobs2 = np.tile(np.mod(tobs1+np.tile(tpast_startTimes,(len(tobs1),1)).T,periods*u.year.to('day')),(7,1)).T
        time61 = time.time()
        time70 = time.time()
        #DELETEplanetDetectedBools2_times = np.multiply(ts2[:,:-1] < tobs2,np.multiply(ts2[:,1:] > tobs2,planetIsVisibleBool2)) #is the planet visible at this time segment in time 2?
        planetDetectedBools2_times_gpu = linalg.multiply(ts2_gpu[:,:-1] < tobs2,linalg.multiply(ts2_gpu[:,1:] > tobs2,planetIsVisibleBool2_gpu)) #is the planet visible at this time segment in time 2?
        time71 = time.time()
        time80 = time.time()
        planetDetectedBools2 = np.nansum(planetDetectedBools2_times,axis=1)
        time81 = time.time()
        time90 = time.time()
        planetNotDetectedBools2 = np.logical_not(planetDetectedBools2) #for dynamic completeness, the planet is not visible in this time segment at time 2
        time91 = time.time()

        #Revisit Comp.
        time100 = time.time()
        planetDetectedthenDetected = linalg.multiply(planetDetectedBools_gpu,planetDetectedBools2_gpu) #each planet detected at time 1 and time 2 #planets detected and still in visible region    
        time101 = time.time()
        #Dynamic Comp.
        time110 = time.time()
        planetNotDetectedThenDetected = linalg.multiply(planetNotDetectedBools_gpu,planetDetectedBools2_gpu) #each planet NOT detected at time 1 and detected at time 2 #planet not detected and now in visible region
        time111 = time.time()

        time120 = time.time()
        dynComp = np.sum(planetNotDetectedThenDetected)/len(planetNotDetectedThenDetected) #divide by all planets
        time121 = time.time()
        #dynComp = np.sum(planetNotDetectedBools)/np.sum(planetTypeBool) #divide by all planets of type
        time130 = time.time()
        revisitComp = np.sum(planetDetectedthenDetected)/np.sum(planetDetectedBools) #divide by all planetes detected at startTimes
        time131 = time.time()

        print(time01-time00)
        print(time11-time10)
        print(time21-time20)
        print(time31-time30)
        print(time41-time40)
        print(time51-time50)
        print(time61-time60)
        print(time71-time70)
        print(time81-time80)
        print(time91-time90)
        print(time101-time100)
        print(time111-time110)
        print(time121-time120)
        print(time131-time130)


        print(saltyburrito)

    return dynComp, revisitComp

timingStart = time.time()
trange = np.linspace(start=0.,stop=365.*13.,num=1000)
dynComps = list()
revisitComps = list()
startTimes = np.asarray([tobs1]*7).T
planetTypeInds = np.arange(len(tobs1)) #using all planets for now
periods2 = np.asarray([periods[planetTypeInds]]*7).T
#We only need to evaluate time windows where the planet is visible
numVisRegions = np.sum(planetIsVisibleBool2[planetTypeInds],axis=1) #Finds the number of visible regions for each planet that is the type we care about
numVisRegionsInds = [np.where(numVisRegions==0)[0],np.where(numVisRegions==1)[0],np.where(numVisRegions==2)[0],np.where(numVisRegions==3)[0],np.where(numVisRegions==4)[0]] # finds inds of those planets we care about
assert np.max(numVisRegions) <= 4, 'need to add more max regions'
numVr = np.max(numVisRegions)+1 #number of vr to iterate over
####Reduce ts2 to only visible regions
tssStart = list()
tssEnd = list()
for i in np.arange(len(numVisRegionsInds)): #iterate over number of vis regions
    if i == 0:
        tssStart.append(np.nan)
        tssEnd.append(np.nan)
    else:
        tsiStart = list()
        tsiEnd = list()
        for j in np.arange(len(numVisRegionsInds[i])): #iterate over number of each planet with number of visible regions
            inds = np.where(planetIsVisibleBool2[numVisRegionsInds[i][j]])[0] #find inds of windows where visible
            tsikStart = list()
            tsikEnd = list()
            for k in np.arange(len(inds)):
                tsikStart.append(ts2[numVisRegionsInds[i][j],inds[k]]) #front edge of visible region
                tsikEnd.append(ts2[numVisRegionsInds[i][j],inds[k]+1]) #end edge of visible region
            tsiStart.append(np.asarray(tsikStart))
            tsiEnd.append(np.asarray(tsikEnd))
        tssStart.append(np.asarray(tsiStart))
        tssEnd.append(np.asarray(tsiEnd))
for k in np.arange(len(trange)):
    dynComp = 0.
    revisitComp = 0.0
    for i in np.arange(numVr-1)+1:#we skip numVr=0 bc there are no visible regions
        dynComp_vr, revisitComp_vr = dynamicCompleteness(tssStart[i],tssEnd[i],planetIsVisibleBool2,startTimes[numVisRegionsInds[i],:i],trange[k],periods2[numVisRegionsInds[i],:i],numVisRegionsInds[i],False)
        dynComp = dynComp + dynComp_vr*len(numVisRegionsInds[i])/len(planetTypeInds)
        revisitComp = revisitComp + revisitComp_vr*len(numVisRegionsInds[i])/len(planetTypeInds)
    dynComps.append(dynComp)
    revisitComps.append(revisitComp)
timingStop = time.time()
print('time: ' + str(timingStop-timingStart))

# #### Running time trials GPU
# gputimeList = list()
# for timei in np.arange(1000):
#     timingStart = time.time()
#     trange = np.linspace(start=0.,stop=365.*13.,num=1000)
#     dynComps = list()
#     revisitComps = list()
#     for k in np.arange(len(trange)):
#         dynComp, revisitComp = dynamicCompleteness(ts2,planetIsVisibleBool2,startTimes,trange[k],periods2,planetTypeInds,True)
#         dynComps.append(dynComp)
#         revisitComps.append(revisitComp)
#     timingStop = time.time()
#     print('time: ' + str(timingStop-timingStart))
#     gputimeList.append(timingStop-timingStart)
# ####



# #### Running time trials NO GPU
# timeList = list()
# for timei in np.arange(1000):
#     timingStart = time.time()
#     trange = np.linspace(start=0.,stop=365.*13.,num=1000)
#     dynComps = list()
#     revisitComps = list()
#     for k in np.arange(len(trange)):
#         dynComp, revisitComp = dynamicCompleteness(ts2,planetIsVisibleBool2,startTimes,trange[k],periods2,planetTypeInds,False)
#         dynComps.append(dynComp)
#         revisitComps.append(revisitComp)
#     timingStop = time.time()
#     print('time: ' + str(timingStop-timingStart))
#     timeList.append(timingStop-timingStart)
# ####

#### Running time trials NO GPU
timeList = list()
for timei in np.arange(1000):
    timingStart = time.time()
    trange = np.linspace(start=0.,stop=365.*13.,num=1000)
    dynComps = list()
    revisitComps = list()
    startTimes = np.asarray([tobs1]*7).T
    planetTypeInds = np.arange(len(tobs1)) #using all planets for now
    periods2 = np.asarray([periods[planetTypeInds]]*7).T
    #We only need to evaluate time windows where the planet is visible
    numVisRegions = np.sum(planetIsVisibleBool2[planetTypeInds],axis=1) #Finds the number of visible regions for each planet that is the type we care about
    numVisRegionsInds = [np.where(numVisRegions==0)[0],np.where(numVisRegions==1)[0],np.where(numVisRegions==2)[0],np.where(numVisRegions==3)[0],np.where(numVisRegions==4)[0]] # finds inds of those planets we care about
    assert np.max(numVisRegions) <= 4, 'need to add more max regions'
    numVr = np.max(numVisRegions)+1 #number of vr to iterate over
    ####Reduce ts2 to only visible regions
    tssStart = list()
    tssEnd = list()
    for i in np.arange(len(numVisRegionsInds)): #iterate over number of vis regions
        if i == 0:
            tssStart.append(np.nan)
            tssEnd.append(np.nan)
        else:
            tsiStart = list()
            tsiEnd = list()
            for j in np.arange(len(numVisRegionsInds[i])): #iterate over number of each planet with number of visible regions
                inds = np.where(planetIsVisibleBool2[numVisRegionsInds[i][j]])[0] #find inds of windows where visible
                tsikStart = list()
                tsikEnd = list()
                for k in np.arange(len(inds)):
                    tsikStart.append(ts2[numVisRegionsInds[i][j],inds[k]]) #front edge of visible region
                    tsikEnd.append(ts2[numVisRegionsInds[i][j],inds[k]+1]) #end edge of visible region
                tsiStart.append(np.asarray(tsikStart))
                tsiEnd.append(np.asarray(tsikEnd))
            tssStart.append(np.asarray(tsiStart))
            tssEnd.append(np.asarray(tsiEnd))
    for k in np.arange(len(trange)):
        dynComp = 0.
        revisitComp = 0.0
        for i in np.arange(numVr-1)+1:#we skip numVr=0 bc there are no visible regions
            dynComp_vr, revisitComp_vr = dynamicCompleteness(tssStart[i],tssEnd[i],planetIsVisibleBool2,startTimes[numVisRegionsInds[i],:i],trange[k],periods2[numVisRegionsInds[i],:i],numVisRegionsInds[i],False)
            dynComp = dynComp + dynComp_vr*len(numVisRegionsInds[i])/len(planetTypeInds)
            revisitComp = revisitComp + revisitComp_vr*len(numVisRegionsInds[i])/len(planetTypeInds)
        dynComps.append(dynComp)
        revisitComps.append(revisitComp)
    timingStop = time.time()
    print('time: ' + str(timingStop-timingStart))
    timeList.append(timingStop-timingStart)
####



#### Load CSV Data Of Brown2010 Paper
BrownData = list()
with open('Brown2010DynamicCompDataPickedFromFigure1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        BrownData.append(row)
BrownData = np.asarray(BrownData).astype('float')

#Dynamic Completeness With Corey's Method
with open('./Brown2010Lambert.pkl', 'rb') as f:
    Brown2010Lambert = pickle.load(f)
with open('./Brown2010QuasiLambert.pkl', 'rb') as f:
    Brown2010QuasiLambert = pickle.load(f)

#### Plot Revisit and Dynamic Completeness of All Planets and Earth-Like Planets
num=8008
plt.close(num)
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.plot(trange*24*60*60,dynComps,color='blue',label='This Work')
#plt.plot(trange*24*60*60,revisitComps,color='red',label='Redetection')
#KEEP plt.scatter(10.**BrownData[:,0],BrownData[:,1],color='black',s=2,label='Figure 1 Brown 2010 Data') #KEEP, not using because dmitry said not to. might confuse the reviewer
plt.plot(Brown2010Lambert[0],Brown2010Lambert[1],color='orange',label='Brown Lambert')
plt.plot(Brown2010QuasiLambert[0],Brown2010QuasiLambert[1],color='red',label='Brown Quasi-Lambert')
plt.xlabel('Time Past Observation (sec)',weight='bold')
plt.ylabel('Dynamic Completeness',weight='bold')
plt.legend(loc=4, prop={'size': 10})
plt.xlim([10**5,np.max(trange*24*60*60)])
plt.ylim([0.,0.3])
plt.xscale('log')
plt.show(block=False)
plt.gcf().canvas.draw()
# Save to a File
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'Brown2010DynamicCompleteness' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)

#### Plot Revisit and Dynamic Completeness of All Planets and Earth-Like Planets
num=8009
plt.close(num)
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
plt.plot(trange*24*60*60,dynComps,color='blue',label='New Detection')
plt.plot(trange*24*60*60,revisitComps,color='red',label='Redetection')
#plt.scatter(10.**BrownData[:,0],BrownData[:,1],color='black',s=2,label='Figure 1 Brown 2010 Data')
#plt.plot(Brown2010Lambert[0],Brown2010Lambert[1],color='orange',label='Brown Lambert')
#plt.plot(Brown2010QuasiLambert[0],Brown2010QuasiLambert[1],color='red',label='Brown Quasi-Lambert')
plt.xlabel('Time Past Observation (sec)',weight='bold')
plt.ylabel('Probability',weight='bold')
plt.legend(loc=4, prop={'size': 10})
plt.xlim([10**5,np.max(trange*24*60*60)])
plt.ylim([0.,1.0])
plt.xscale('log')
plt.show(block=False)
plt.gcf().canvas.draw()
# Save to a File
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'RevisitCompleteness' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)

