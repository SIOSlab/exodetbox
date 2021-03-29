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

#### PLOT BOOL
plotBool = False
if plotBool == True:
    from plotProjectedEllipse import *
folder = './'
PPoutpath = './'

#### Randomly Generate Orbits
folder_load = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/exo-det-box/exo-det-box/convergence_data'))
filename = 'HabEx_CKL2_PPKL2.json'
filename = 'WFIRSTcycle6core.json'
filename = 'HabEx_CSAG13_PPSAG13_compSubtype.json'
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
inc[np.where(inc>np.pi/2)[0]] = np.pi - inc[np.where(inc>np.pi/2)[0]]
sma, e, p, Rp = PPop.gen_plan_params(n)

#### Adjustment for Planets Causing Errors
#Planet to be removed
ar = 0.6840751713914676*u.AU #sma
er = 0.12443160036480415 #e
Wr = 6.1198652952593 #W
wr = 2.661645323283813 #w
incr = 0.8803680245150818 #inc
sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)
ar = 1.1300859542315127*u.AU
er = 0.23306811746716588
Wr = 5.480292250277455
wr = 2.4440871464730183
incr = 1.197618937201339
sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr)



#### Classify Planets
bini, binj, earthLike = comp.classifyPlanets(Rp, TL, np.arange(len(sma)), sma, e)
sma = sma.to('AU').value
####

#Separations
s_circle = np.ones(len(sma))
dmag = 25. #29.0
dmag_upper = 25. #29.0
IWA_HabEx = 0.045*u.arcsec #taken from a Habex Script in units of mas
IWA2=0.150*u.arcsec #Suggested by dmitry as analahous to WFIRST
OWA_HabEx = 6.*u.arcsec #from the HabEx Standards Team Final Report
s_inner = 10.*u.pc.to('AU')*IWA_HabEx.to('rad').value
s_outer = 10.*u.pc.to('AU')*OWA_HabEx.to('rad').value

#starMass
starMass = const.M_sun

#### SAVED PLANET FOR Plot 3D Ellipse to 2D Ellipse Projection Diagram
ellipseProjection3Dto2DInd = 23 #22
sma[ellipseProjection3Dto2DInd] = 1.2164387563540457
e[ellipseProjection3Dto2DInd] = 0.531071885292766
w[ellipseProjection3Dto2DInd] = 3.477496280463054
W[ellipseProjection3Dto2DInd] = 5.333215834002414
inc[ellipseProjection3Dto2DInd] = 1.025093642138022
####
#### SAVED PLANET FOR Plot Projected, derotated, centered ellipse 
derotatedInd = 33
sma[derotatedInd] = 5.738800898338014
e[derotatedInd] = 0.29306873405223816
w[derotatedInd] = 4.436383063578559
W[derotatedInd] = 4.240810639711751
inc[derotatedInd] = 1.072680736014668
####
#### SAVED PLANET FOR Plot Sep vs nu
sepvsnuInd = 24
sma[sepvsnuInd] = 1.817006521549392
e[sepvsnuInd] = 0.08651509983996385
W[sepvsnuInd] = 3.3708439025758006
w[sepvsnuInd] = 4.862116908343989
inc[sepvsnuInd] = 1.2491324942585256
####
#### SAVED PLANET FOR Plot Sep vs t
sepvstInd = 25
sma[sepvstInd] = 2.204556035394906
e[sepvstInd] = 0.2898368164549611
W[sepvstInd] = 4.787284415551434
w[sepvstInd] = 2.71176523941224
inc[sepvstInd] = 1.447634036719772
####
#### A NICE 4 INTERSECTION EXAMPLE
fourIntersectionInd = 2173 #33
sma[fourIntersectionInd] = 5.363760022304063
e[fourIntersectionInd] = 0.557679118292977
w[fourIntersectionInd] = 5.058312201296985
W[fourIntersectionInd] = 0.6867396268911974
inc[fourIntersectionInd] = 0.8122666711110185
p[fourIntersectionInd] = 0.30
Rp[fourIntersectionInd] = 4.*u.earthRad
####



dmajorp,dminorp,theta_OpQ_X,theta_OpQp_X,Op,x,y,Phi,xreal,only2RealInds,yrealAllRealInds,\
    fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt,\
    nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
    t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,t_twoIntSameY0,\
    t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
    minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
    errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,\
    errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,type0_0Inds,\
    type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
    type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
    twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods = calcMasterIntersections(sma,e,W,w,inc,s_circle,starMass,plotBool)



#### START ANALYSIS AND PLOTTING ######################################
#######################################################################
if plotBool == True:
    #### Plotting Projected Ellipse
    start2 = time.time()
    ind = random.randint(low=0,high=n)
    ind = 24 #for testing purposes
    plotProjectedEllipse(ind, sma, e, W, w, inc, Phi, dmajorp, dminorp, Op, num=877)
    plotSeparationvsnu(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep, \
    nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints,\
    nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, num=878)
    plotSeparationVsTime(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep,\
    t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,\
    t_twoIntSameY0,t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
    nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, periods, num=879)
    plotSeparationVsTime
    stop2 = time.time()
    print('stop2: ' + str(stop2-start2))
    del start2, stop2
    #plt.close(877)
    ####

    #### Plot 3D Ellipse to 2D Ellipse Projection Diagram
    start3 = time.time()
    num = 666999888777
    plot3DEllipseto2DEllipseProjectionDiagram(ind, sma, e, W, w, inc, Op, Phi,\
        dmajorp, dminorp, num=num)
    stop3 = time.time()
    print('stop3: ' + str(stop3-start3))
    del start3, stop3
    #plt.close(num)
    ####

    #### Create Projected Ellipse Conjugate Diameters and QQ' construction diagram
    start4 = time.time()
    num = 3335555888
    plotEllipseMajorAxisFromConjugate(ind, sma, e, W, w, inc, Op, Phi,\
        dmajorp, dminorp, num)
    stop4 = time.time()
    print('stop4: ' + str(stop4-start4))
    del start4, stop4
    #plt.close(num)
    ####

    #### Plot Derotated Ellipse
    start6 = time.time()
    num=880
    plotDerotatedEllipse(ind, sma, e, W, w, inc, Phi, dmajorp, dminorp, Op, x, y, num)
    stop6 = time.time()
    print('stop6: ' + str(stop6-start6))
    del start6, stop6
    #plt.close(num)
    ####

    ##### Plot Proving Rerotation method works
    start10 = time.time()
    num=883
    plotReorientationMethod(ind, sma, e, W, w, inc, x, y, Phi, Op, dmajorp, dminorp,\
        minSepPoints_x, minSepPoints_y, num)
    stop10 = time.time()
    print('stop10: ' + str(stop10-start10))
    del start10, stop10
    #plt.close(num)
    ####

    #### Plot Derotated Intersections, Min/Max, and Star Location Type Bounds
    start12 = time.time()
    num = 960
    plotDerotatedIntersectionsMinMaxStarLocBounds(ind, sma, e, W, w, inc, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
        minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        lmaxSepPoints_x, lmaxSepPoints_y, twoIntSameYInds,\
        maxSepPoints_x, maxSepPoints_y, twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
        type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
        type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num)
    stop12 = time.time()
    print('stop12: ' + str(stop12-start12))
    del start12, stop12
    #plt.close(num)
    ####

    #### Plot Derotated Ellipse Separation Extrema
    start12_1 = time.time()
    num = 961
    plotDerotatedExtrema(derotatedInd, sma, e, W, w, inc, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
        maxSepPoints_x, maxSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
        minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, num)
    stop12_1 = time.time()
    print('stop12_1: ' + str(stop12_1-start12_1))
    del start12_1, stop12_1
    ####

    #### Plot Rerotated Points 
    #### Error Plot ####
    num=822
    errorLinePlot(fourIntInds,errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,\
        twoIntSameYInds,errors_twoIntSameY0,errors_twoIntSameY1,twoIntOppositeXInds,errors_twoIntOppositeX0,errors_twoIntOppositeX1,\
        only2RealInds,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,num)
    #plt.close(num)
    ######################

    # ind = yrealAllRealInds[fourIntInds[np.argsort(-errors_fourInt1)[0]]]
    # plotRerotatedFromNus(ind, sma[ind], e[ind], W[ind], w[ind], inc[ind], Op[:,ind], yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
    #     nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    #     twoIntSameY_x, twoIntSameY_y, num=8001)

    tmpind = yrealAllRealInds[twoIntSameYInds[np.argsort(-errors_twoIntSameY1)[0]]]
    plotRerotatedFromNus(tmpind, sma[tmpind], e[tmpind], W[tmpind], w[tmpind], inc[tmpind], Op[:,tmpind], yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
        nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
        twoIntSameY_x, twoIntSameY_y, num=8001)

    #Check If Nus of Intersection are too close
    errornu_twoIntSameY = np.abs(nu_twoIntSameY[np.arange(nu_twoIntSameY.shape[0]),0] - nu_twoIntSameY[np.arange(nu_twoIntSameY.shape[0]),1])
    assert np.all(errornu_twoIntSameY>1e-5), 'nu differences are too small'
    errornu_twoIntOppositeX = np.abs(nu_twoIntOppositeX[np.arange(nu_twoIntOppositeX.shape[0]),0] - nu_twoIntOppositeX[np.arange(nu_twoIntOppositeX.shape[0]),1])
    assert np.all(errornu_twoIntOppositeX>1e-5), 'nu differences are too small'
    errornu_IntersectionsOnly2 = np.abs(nu_IntersectionsOnly2[np.arange(nu_IntersectionsOnly2.shape[0]),0] - nu_IntersectionsOnly2[np.arange(nu_IntersectionsOnly2.shape[0]),1])
    assert np.all(errornu_IntersectionsOnly2>1e-5), 'nu differences are too small'
    #nu_fourInt
    errornu_fourInt0 = np.abs(nu_fourInt[np.arange(nu_fourInt.shape[0]),0] - nu_fourInt[np.arange(nu_fourInt.shape[0]),1])
    errornu_fourInt1 = np.abs(nu_fourInt[np.arange(nu_fourInt.shape[0]),0] - nu_fourInt[np.arange(nu_fourInt.shape[0]),2])
    errornu_fourInt2 = np.abs(nu_fourInt[np.arange(nu_fourInt.shape[0]),0] - nu_fourInt[np.arange(nu_fourInt.shape[0]),3])
    assert np.all(errornu_fourInt0>1e-5), 'nu differences are too small'
    assert np.all(errornu_fourInt1>1e-5), 'nu differences are too small'
    assert np.all(errornu_fourInt2>1e-5), 'nu differences are too small'

    # ind = only2RealInds[np.argsort(-errors_IntersectionsOnly2X0)[0]]
    # plotRerotatedFromNus(ind, sma[ind], e[ind], W[ind], w[ind], inc[ind], Op[:,ind], yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
    #     nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    #     twoIntSameY_x, twoIntSameY_y, num=8001)

    ###### DONE FIXING NU

    #### Plot Histogram of Error
    num= 823
    plotErrorHistogramAlpha(errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,errors_twoIntSameY1,\
        errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,num)
    #plt.close(num) #thinking the above plot is relativly useless
    ####

    #### Plot Histogram of Error
    num=824
    plotErrorHistogram(errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,\
        errors_twoIntSameY0,errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,\
        errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,num)
    #plt.close(num)
    ####

    #### Redo Significant Point plot Using these Nu
    num=3690
    plotProjectedEllipseWithNu(ind,sma,e,W,w,inc,nu_minSepPoints,nu_maxSepPoints, yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds,\
        only2RealInds, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2, num)
    ####

    #### Plot separation vs nu
    num=962
    plotSeparationvsnu(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep, \
        nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints,\
        nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
        yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, num)
    ####

    #### Plot separation vs time
    num=963
    plotSeparationVsTime(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep,\
        t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,\
        t_twoIntSameY0,t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
        nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
        yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, periods, num)
    ####

    ####  Plot Derotate Ellipse
    tinds = np.argsort(-np.abs(errors_fourInt1))
    tind2 = yrealAllRealInds[fourIntInds[tinds[1]]]
    num=55670
    plotDerotatedEllipseStarLocDividers(tind2, sma, e, W, w, inc, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
        minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        lmaxSepPoints_x, lmaxSepPoints_y, twoIntSameYInds,\
        maxSepPoints_x, maxSepPoints_y, twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
        type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
        type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num)
    #plt.close(num)
    ####

    #### Min Seps Histogram
    num=9701
    plotSepsHistogram(minSep,maxSep,lminSep,lmaxSep,sma,yrealAllRealInds,num)





#### nu From dMag #####################################################
#### Solving for dmag_min and dmag_max for each planet ################
mindmag, maxdmag, dmaglminAll, dmaglmaxAll, indsWith2, indsWith4, nuMinDmag, nuMaxDmag, nulminAll, nulmaxAll = calc_planet_dmagmin_dmagmax(e,inc,w,sma*u.AU,p,Rp)
print('Num Planets with At Least 2 Int given dmag: ' + str(np.sum((mindmag < dmag)*(maxdmag > dmag))))
print('Num Planets with dmag local extrema: ' + str(len(indsWith4)))
print('Num Planets with given 4 Int given dmag: ' + str(np.sum((dmaglminAll < dmag)*(dmaglmaxAll > dmag))))
indsWith4Int = indsWith4[np.where((dmaglminAll < dmag)*(dmaglmaxAll > dmag))[0]]
indsWith2Int = np.asarray(list(set(np.where((mindmag < dmag)*(maxdmag > dmag))[0]) - set(indsWith4Int)))
######################################################################

#### Dmag Extrema Times ##############################################
time_dmagmin = timeFromTrueAnomaly(nuMinDmag,periods,e)
time_dmagmax = timeFromTrueAnomaly(nuMaxDmag,periods,e)
time_dmaglmin = timeFromTrueAnomaly(nulminAll,periods[indsWith4],e[indsWith4])
time_dmaglmax = timeFromTrueAnomaly(nulmaxAll,periods[indsWith4],e[indsWith4])
######################################################################

######################################################################
#### Solving for nu, dmag intersections ##############################
nus2Int, nus4Int, dmag2Int, dmag4Int = calc_planetnu_from_dmag(dmag,e,inc,w,sma*u.AU,p,Rp,mindmag, maxdmag, indsWith2Int, indsWith4Int)
time_dmagInts = np.zeros((len(e),4))*np.nan
time_dmagInts[indsWith2Int,0] = timeFromTrueAnomaly(nus2Int[:,0],periods[indsWith2Int],e[indsWith2Int])
time_dmagInts[indsWith2Int,1] = timeFromTrueAnomaly(nus2Int[:,1],periods[indsWith2Int],e[indsWith2Int])
if not indsWith4Int is None and not nus4Int is None:
    time_dmagInts[indsWith4Int,0] = timeFromTrueAnomaly(nus4Int[:,0],periods[indsWith4Int],e[indsWith4Int])
    time_dmagInts[indsWith4Int,1] = timeFromTrueAnomaly(nus4Int[:,1],periods[indsWith4Int],e[indsWith4Int])
    time_dmagInts[indsWith4Int,2] = timeFromTrueAnomaly(nus4Int[:,2],periods[indsWith4Int],e[indsWith4Int])
    time_dmagInts[indsWith4Int,3] = timeFromTrueAnomaly(nus4Int[:,3],periods[indsWith4Int],e[indsWith4Int])
# t2Int = np.zeros((len(indsWith2Int),4))
# t2Int[:,0] = timeFromTrueAnomaly(nus2Int[:,0],periods[indsWith2Int],e[indsWith2Int])
# t2Int[:,1] = timeFromTrueAnomaly(nus2Int[:,1],periods[indsWith2Int],e[indsWith2Int])
# t4Int = np.zeros((len(indsWith4Int),4))
# t4Int[:,0] = timeFromTrueAnomaly(nus4Int[:,0],periods[indsWith4Int],e[indsWith4Int])
# t4Int[:,1] = timeFromTrueAnomaly(nus4Int[:,1],periods[indsWith4Int],e[indsWith4Int])
# t4Int[:,2] = timeFromTrueAnomaly(nus4Int[:,2],periods[indsWith4Int],e[indsWith4Int])
# t4Int[:,3] = timeFromTrueAnomaly(nus4Int[:,3],periods[indsWith4Int],e[indsWith4Int])
######################################################################

#### Bulking all Times Together ######################################
times_s = calc_t_sInnersOuter(sma,e,W,w,inc,s_inner*np.ones(len(sma)),s_outer*np.ones(len(sma)),starMass,plotBool)
times = np.concatenate((np.zeros((len(e),1)),times_s,time_dmagInts,np.reshape(periods,(len(periods),1))),axis=1)
timesSortInds = np.argsort(times,axis=1)
times2 = np.sort(times,axis=1) #sorted from smallest to largest
indsWithAnyInt = np.where(np.sum(~np.isnan(times2),axis=1))[0] #Finds the planets which have any intersections
#####################################################################




#Check visibility in all given bounds (For Completeness)
#NEED TO BE ABLE TO PUT BOUNDS INTO BOX WITH 4 SIDES
#AND BOX WITH 3 SIDES

#### dmag vs nu extrema and intersection Verification plot
num=88833543453218
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
ind = fourIntersectionInd#indsWith4Int[0]
nus = np.linspace(start=0,stop=2.*np.pi,num=100)
phis = (1.+np.sin(inc[ind])*np.sin(nus+w[ind]))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
ds = sma[ind]*(1.-e[ind]**2.)/(e[ind]*np.cos(nus)+1.)
dmags = deltaMag(p[ind],Rp[ind].to('AU'),ds*u.AU,phis) #calculate dmag of the specified x-value

plt.plot(nus,dmags,color='black',zorder=10)
#plt.plot([0.,2.*np.pi],[dmag,dmag],color='blue')
plt.scatter(nuMinDmag[ind],mindmag[ind],color='teal',marker='D',zorder=20)
plt.plot([0.,2.*np.pi],[mindmag[ind],mindmag[ind]],color='teal',zorder=20)
plt.scatter(nuMaxDmag[ind],maxdmag[ind],color='red',marker='D',zorder=20)
plt.plot([0.,2.*np.pi],[maxdmag[ind],maxdmag[ind]],color='red',zorder=20)
lind = np.where(ind == indsWith4)[0]
if  ind in indsWith2Int:
    mind = np.where(ind == indsWith2Int)[0][0]
    plt.scatter(nus2Int[mind],dmag2Int[mind],color='green',marker='o',zorder=20)
    plt.plot([0.,2.*np.pi],[dmag,dmag],color='green',zorder=10)
elif ind in indsWith4Int:
    nind = np.where(ind == indsWith4Int)[0]
    plt.scatter(nus4Int[nind],dmag4Int[nind],color='green',marker='o',zorder=20)
    plt.plot([0.,2.*np.pi],[dmag,dmag],color='green',zorder=10)
plt.scatter(nulminAll[lind],dmaglminAll[lind],color='magenta',marker='D',zorder=20)
#plt.plot([0.,2.*np.pi],[dmaglminAll[lind],dmaglminAll[lind]],color='magenta',zorder=20)
plt.scatter(nulmaxAll[lind],dmaglmaxAll[lind],color='gold',marker='D',zorder=20)
#plt.plot([0.,2.*np.pi],[dmaglmaxAll[lind],dmaglmaxAll[lind]],color='gold',zorder=20)
plt.xlim([0.,2.*np.pi])
plt.ylim([-0.05*(maxdmag[ind]-mindmag[ind])+mindmag[ind],0.05*(maxdmag[ind]-mindmag[ind])+maxdmag[ind]])
plt.ylabel(r'$\Delta \mathrm{mag}$',weight='bold')
plt.xlabel('True Anomaly, ' + r'$\nu$' + ', in (rad)', weight='bold')
plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + ' w: ' + str(np.round(w[ind],4)) + '\ninc: ' + str(np.round(inc[ind],4)) + ' p: ' + str(np.round(p[ind],4)) + ' Rp: ' + ' inc: ' + str(np.round(Rp[ind],4)))
plt.gcf().canvas.draw()
# Save to a File
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'KeithlyDmagvsnuInts' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
print('Done plotting KeithlyDmagvsnuInts')
plt.show(block=False)

#### Verification With Time
num=8883354345329
plt.figure(num=num)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
ind = fourIntersectionInd#indsWith4Int[0]
nus = np.linspace(start=0,stop=2.*np.pi,num=100)
tmp_times = timeFromTrueAnomaly(nus,periods[ind],e[ind])
phis = (1.+np.sin(inc[ind])*np.sin(nus+w[ind]))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
ds = sma[ind]*(1.-e[ind]**2.)/(e[ind]*np.cos(nus)+1.)
dmags = deltaMag(p[ind],Rp[ind].to('AU'),ds*u.AU,phis) #calculate dmag of the specified x-value

plt.plot(tmp_times,dmags,color='black',zorder=10)
#plt.plot([0.,2.*np.pi],[dmag,dmag],color='blue')
plt.scatter(time_dmagmin[ind],mindmag[ind],color='teal',marker='D',zorder=20)
plt.plot([0.,periods[ind]],[mindmag[ind],mindmag[ind]],color='teal',zorder=20)
plt.scatter(time_dmagmax[ind],maxdmag[ind],color='red',marker='D',zorder=20)
plt.plot([0.,periods[ind]],[maxdmag[ind],maxdmag[ind]],color='red',zorder=20)
lind = np.where(ind == indsWith4)[0]
if  ind in indsWith2Int:
    mind = np.where(ind == indsWith2Int)[0][0]
    plt.scatter(time_dmagInts[indsWith2Int[mind],:2],dmag2Int[mind],color='green',marker='o',zorder=20)
    plt.plot([0.,periods[ind]],[dmag,dmag],color='green',zorder=10)
elif ind in indsWith4Int:
    nind = np.where(ind == indsWith4Int)[0][0]
    plt.scatter(time_dmagInts[indsWith4Int[nind]],dmag4Int[nind],color='green',marker='o',zorder=20)
    plt.plot([0.,periods[ind]],[dmag,dmag],color='green',zorder=10)
plt.scatter(time_dmaglmin[lind],dmaglminAll[lind],color='magenta',marker='D',zorder=20)
#plt.plot([0.,periods[ind]],[dmaglminAll[lind],dmaglminAll[lind]],color='magenta',zorder=20)
plt.scatter(time_dmaglmax[lind],dmaglmaxAll[lind],color='gold',marker='D',zorder=20)
#plt.plot([0.,periods[ind]],[dmaglmaxAll[lind],dmaglmaxAll[lind]],color='gold',zorder=20)
plt.xlim([0.,periods[ind]])
plt.ylim([-0.05*(maxdmag[ind]-mindmag[ind])+mindmag[ind],0.05*(maxdmag[ind]-mindmag[ind])+maxdmag[ind]])
plt.ylabel(r'$\Delta \mathrm{mag}$',weight='bold')
plt.xlabel('Time Past Periastron, t, (years)',weight='bold')
plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + ' w: ' + str(np.round(w[ind],4)) + '\ninc: ' + str(np.round(inc[ind],4)) + ' p: ' + str(np.round(p[ind],4)) + ' Rp: ' + ' inc: ' + str(np.round(Rp[ind],4)))
plt.gcf().canvas.draw()
# Save to a File
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'KeithlyDmagvstInts' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
print('Done plotting KeithlyDmagvstInts')
plt.show(block=False)
#######################################################################


#### Verifying Planet Visibility Windows #################################
##########################################################################


nus, planetIsVisibleBool = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, s_inner, s_outer, dmag_upper, dmag_lower=None) #Calculate planet-star nu edges and visible regions
ts = timeFromTrueAnomaly(nus,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges
dt = ts[:,1:] - ts[:,:-1] #Calculate time region widths
maxIntTime = 30.
gtIntLimit = dt > maxIntTime #Create boolean array for inds
totalVisibleTimePerTarget = np.nansum(np.multiply(np.multiply(dt-maxIntTime,planetIsVisibleBool.astype('int')),gtIntLimit),axis=1) #We subtract the int time from the fraction of observable time
totalCompleteness = np.divide(totalVisibleTimePerTarget,periods*u.year.to('day')) # Fraction of time each planet is visible of its period


#### Data Struct of Completeness
compDict = dict()
maxIntTimes = [0.,0.25,0.5,0.75,1.,1.5,2.,2.5,3.,4.,5.,8.,10.,15.,20.,25.,30.,45.,60.,75.,90.] #in days
starDistances = [5.,10.,15.,20.,25.] #in pc
for i in np.arange(len(starDistances)):
    starDistance = starDistances[i]
    s_inner = starDistance*u.pc.to('AU')*IWA_HabEx.to('rad').value
    s_outer = starDistance*u.pc.to('AU')*OWA_HabEx.to('rad').value #RANDOMLY MULTIPLY BY 3 HERE
    #will need to recalculate separations
    for j in np.arange(len(maxIntTimes)):
        maxIntTime = maxIntTimes[j]
        compDict[(i,j)] = dict()
        compDict[(i,j)]['maxIntTime'] = maxIntTime
        compDict[(i,j)]['stardistance'] = starDistance
        compDict[(i,j)]['s_inner'] = s_inner
        compDict[(i,j)]['s_outer'] = s_outer
        nus, planetIsVisibleBool = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, s_inner, s_outer, dmag_upper, dmag_lower=None) #Calculate planet-star nu edges and visible regions
        ts = timeFromTrueAnomaly(nus,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges
        dt = ts[:,1:] - ts[:,:-1] #Calculate time region widths
        # Completeness Calculated Based On Planets In the instrument's visibility limits
        compDict[(i,j)]['totalVisibleTimePerTarget'] = np.nansum(np.multiply(dt,planetIsVisibleBool.astype('int')),axis=1) #The traditional calculation, accounting for how long the planet is in the visible region
        compDict[(i,j)]['totalCompletenessPerTarget'] = np.divide(compDict[(i,j)]['totalVisibleTimePerTarget'],periods*u.year.to('day')) # Fraction of time each planet is visible of its period
        compDict[(i,j)]['totalCompleteness'] = np.sum(compDict[(i,j)]['totalCompletenessPerTarget'])/len(compDict[(i,j)]['totalCompletenessPerTarget']) #Calculates the total completenss by summing all the fractions and normalize by number of targets
        assert np.all(compDict[(i,j)]['totalCompletenessPerTarget'] >= 0), 'Not all positive comp'
        assert compDict[(i,j)]['totalCompleteness'] >= 0, 'Not positive comp'
        # Completeness 
        gtIntLimit = dt > maxIntTime #Create boolean array for inds
        compDict[(i,j)]['totalVisibleTimePerTarget_maxIntTimeCorrected'] = np.nansum(np.multiply(np.multiply(dt-maxIntTime,planetIsVisibleBool.astype('int')),gtIntLimit.astype('int')),axis=1) #We subtract the int time from the fraction of observable time
        compDict[(i,j)]['totalCompletenessPerTarget_maxIntTimeCorrected'] = np.divide(compDict[(i,j)]['totalVisibleTimePerTarget_maxIntTimeCorrected'],periods*u.year.to('day')) # Fraction of time each planet is visible of its period
        compDict[(i,j)]['totalCompleteness_maxIntTimeCorrected'] = np.sum(compDict[(i,j)]['totalCompletenessPerTarget_maxIntTimeCorrected'])/len(compDict[(i,j)]['totalCompletenessPerTarget_maxIntTimeCorrected']) #Calculates the total completenss by summing all the fractions and normalize by number of targets
        compDict[(i,j)]['SubTypeCompPerTarget'] = dict()
        compDict[(i,j)]['SubTypeCompPerTarget_maxIntTimeCorrected'] = dict()
        compDict[(i,j)]['SubTypeComp'] = dict()
        compDict[(i,j)]['SubTypeComp_maxIntTimeCorrected'] = dict()
        for overi, overj in itertools.product(np.arange(len(comp.Rp_hi)),np.arange(len(comp.L_lo[0,:]))):
            compDict[(i,j)]['SubTypeCompPerTarget'][(overi,overj)] = np.multiply(compDict[(i,j)]['totalCompletenessPerTarget'],((bini==overi)*(binj==overj)).astype('int'))/np.sum(np.multiply(periods*u.year.to('day'),((bini==overi)*(binj==overj)).astype('int'))) #Calculate completeness for this specific planet subtype
            compDict[(i,j)]['SubTypeCompPerTarget_maxIntTimeCorrected'][(overi,overj)] = np.multiply(compDict[(i,j)]['totalCompletenessPerTarget_maxIntTimeCorrected'],((bini==overi)*(binj==overj)).astype('int'))/np.sum(np.multiply(periods*u.year.to('day'),((bini==overi)*(binj==overj)).astype('int'))) #Calculate completeness for this specific planet subtype
            compDict[(i,j)]['SubTypeComp'][(overi,overj)] = np.sum(compDict[(i,j)]['SubTypeCompPerTarget'][(overi,overj)])
        compDict[(i,j)]['SubTypeComp_maxIntTimeCorrected'][(overi,overj)] = np.sum(compDict[(i,j)]['SubTypeCompPerTarget_maxIntTimeCorrected'][(overi,overj)])
        
        #Earth-Like Completeness, The probability of the detected planet being Earth-Like
        compDict[(i,j)]['EarthlikeCompPerTarget'] = np.multiply(compDict[(i,j)]['totalCompletenessPerTarget'],(earthLike).astype('int')) #/np.sum(np.multiply(periods*u.year.to('day'),(earthLike).astype('int'))) #Calculates the completeness for Earth-Like Planets
        compDict[(i,j)]['EarthlikeCompPerTarget_maxIntTimeCorrected'] = np.multiply(compDict[(i,j)]['totalCompletenessPerTarget_maxIntTimeCorrected'],(earthLike).astype('int')) #/np.sum(np.multiply(periods*u.year.to('day'),(earthLike).astype('int'))) #Calculates the completeness for Earth-Like Planets
        compDict[(i,j)]['EarthlikeComp'] = np.sum(compDict[(i,j)]['EarthlikeCompPerTarget'])/len(earthLike) #np.sum(earthLike.astype('int'))
        compDict[(i,j)]['EarthlikeComp_maxIntTimeCorrected'] = np.sum(compDict[(i,j)]['EarthlikeCompPerTarget_maxIntTimeCorrected'])/len(earthLike) #np.sum(earthLike.astype('int'))

        #Earth-Like Completeness
        compDict[(i,j)]['EarthlikeComp2'] = np.sum(compDict[(i,j)]['EarthlikeCompPerTarget'])/np.sum(earthLike.astype('int'))
        compDict[(i,j)]['EarthlikeComp2_maxIntTimeCorrected'] = np.sum(compDict[(i,j)]['EarthlikeCompPerTarget_maxIntTimeCorrected'])/np.sum(earthLike.astype('int'))

maxIntTime = 30. #days
gtIntLimit = dt > maxIntTime #Create boolean array for inds
totalCompletenessIntLimit = np.nansum(np.multiply(np.multiply(dt-maxIntTime,planetIsVisibleBool.astype('int')),gtIntLimit),axis=1) #We subtract the int time from the fraction of observable time
totalCompletenessIntLimit = np.divide(totalVisibleTimePerTarget,periods*u.year.to('day')) # Fraction of time each planet is visible of its period

num=979087987234
plt.figure(num=num)
#plt.hist(totalVisibleTimePerTarget)
plt.hist(totalCompleteness[np.where(np.logical_not(totalCompleteness == 0))[0]])
plt.yscale('log')
plt.show(block=False)


print('TotalCompleteness:')
for i in np.arange(len(starDistances)):
    for j in np.arange(len(maxIntTimes)):
        print('(' + str(i) + ',' + str(j) + '): ' + str(compDict[(i,j)]['totalCompleteness']))

print('TotalCompleteness Max Int Time Corrected:')
for i in np.arange(len(starDistances)):
    for j in np.arange(len(maxIntTimes)):
        print('(' + str(i) + ',' + str(j) + '): ' + str(compDict[(i,j)]['totalCompleteness_maxIntTimeCorrected']))

print('Earth-Like Completeness:')
for i in np.arange(len(starDistances)):
    for j in np.arange(len(maxIntTimes)):
        print('(' + str(i) + ',' + str(j) + '): ' + str(compDict[(i,j)]['EarthlikeComp']))

print('Earth-Like Completeness Max Int Time Corrected:')
for i in np.arange(len(starDistances)):
    for j in np.arange(len(maxIntTimes)):
        print('(' + str(i) + ',' + str(j) + '): ' + str(compDict[(i,j)]['EarthlikeComp_maxIntTimeCorrected']))

print('Earth-Like Completeness2:')
for i in np.arange(len(starDistances)):
    for j in np.arange(len(maxIntTimes)):
        print('(' + str(i) + ',' + str(j) + '): ' + str(compDict[(i,j)]['EarthlikeComp2']))

print('Earth-Like Completeness 2 Max Int Time Corrected:')
for i in np.arange(len(starDistances)):
    for j in np.arange(len(maxIntTimes)):
        print('(' + str(i) + ',' + str(j) + '): ' + str(compDict[(i,j)]['EarthlikeComp2_maxIntTimeCorrected']))


#Printing Table For Paper###########################
print('Table with Completeness and Earth-Like Completeness')
for i in np.arange(len(starDistances)):
    for j in np.arange(len(maxIntTimes)):
        print(str(int(starDistances[i])) + ' & ' + str(maxIntTimes[j]) + ' & ' + str(np.round(compDict[(i,j)]['totalCompleteness_maxIntTimeCorrected'],4)) + ' & ' + str(np.round(compDict[(i,j)]['EarthlikeComp2_maxIntTimeCorrected'],4)))

#getting the colors for the plot
norm = matplotlib.colors.Normalize(vmin=starDistances[0], vmax=starDistances[-1])
cmap = matplotlib.cm.get_cmap('viridis')#('Spectral')
colors = list()
for i in np.arange(len(starDistances)):
    colors.append(cmap(norm(starDistances[i])))
markers = ['s','v','o','+','p']
plt.figure(num=1010101)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
maxpt = 0
for i in np.arange(len(starDistances)):
    ypts = list()
    for j in np.arange(len(maxIntTimes)):
        ypts.append(compDict[(i,j)]['totalCompleteness_maxIntTimeCorrected'])
        if compDict[(i,j)]['totalCompleteness_maxIntTimeCorrected'] > maxpt:
            maxpt = compDict[(i,j)]['totalCompleteness_maxIntTimeCorrected']
    plt.plot(maxIntTimes,ypts,color=colors[i],marker=markers[i],label= str(int(starDistances[i])) + ' pc')
plt.xlabel('Maximum Integration Time (d)',weight='bold')
plt.ylabel('Integration Time Adjusted Completeness',weight='bold')
plt.xlim([0.,100.])
plt.ylim([0.,1.05*np.max(maxpt)])
plt.legend()
plt.show(block=False)
plt.gcf().canvas.draw()
# Save to a File
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'KeithlyCompvsIntTime' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
print('Done plotting KeithlyCompvsIntTime')


#### Integration Time Adjusted Completenes For Earth-Like Planets
plt.figure(num=1010102)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
maxpt = 0
for i in np.arange(len(starDistances)):
    ypts = list()
    for j in np.arange(len(maxIntTimes)):
        ypts.append(compDict[(i,j)]['EarthlikeComp2_maxIntTimeCorrected'])
        if compDict[(i,j)]['EarthlikeComp2_maxIntTimeCorrected'] > maxpt:
            maxpt = compDict[(i,j)]['EarthlikeComp2_maxIntTimeCorrected']
    plt.plot(maxIntTimes,ypts,color=colors[i],marker=markers[i],label= str(int(starDistances[i])) + ' pc')
plt.xlabel('Maximum Integration Time (d)',weight='bold')
plt.ylabel('Integration Time Adjusted Completeness',weight='bold')
plt.xlim([0.,100.])
plt.ylim([0.,1.05*np.max(maxpt)])
plt.legend()
plt.show(block=False)
plt.gcf().canvas.draw()
# Save to a File
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'KeithlyCompvsIntTimeEarthLike' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
print('Done plotting KeithlyCompvsIntTimeEarthLike2')
############################################



#### Log-scale Integration Time Adjusted Completeness
plt.figure(num=101010133)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
maxpt = 0
for i in np.arange(len(starDistances)):
    ypts = list()
    for j in np.arange(len(maxIntTimes)):
        ypts.append(compDict[(i,j)]['totalCompleteness_maxIntTimeCorrected'])
        if compDict[(i,j)]['totalCompleteness_maxIntTimeCorrected'] > maxpt:
            maxpt = compDict[(i,j)]['totalCompleteness_maxIntTimeCorrected']
    plt.plot(maxIntTimes,ypts,color=colors[i],marker=markers[i],label= str(int(starDistances[i])) + ' pc')
plt.xlabel('Maximum Integration Time (d)',weight='bold')
plt.ylabel('Integration Time Adjusted Completeness',weight='bold')
plt.xlim([0.1,100.])
plt.ylim([0.,1.05*np.max(maxpt)])
plt.legend()
plt.xscale('log')
plt.show(block=False)
plt.gcf().canvas.draw()
# Save to a File
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'KeithlyCompvsIntTimeLOG' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
print('Done plotting KeithlyCompvsIntTimeLOG')


#### Integration Time Adjusted Completenes For Earth-Like Planets
plt.figure(num=101010233)
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rcParams['axes.linewidth']=2
plt.rc('font',weight='bold')
maxpt = 0
for i in np.arange(len(starDistances)):
    ypts = list()
    for j in np.arange(len(maxIntTimes)):
        ypts.append(compDict[(i,j)]['EarthlikeComp2_maxIntTimeCorrected'])
        if compDict[(i,j)]['EarthlikeComp2_maxIntTimeCorrected'] > maxpt:
            maxpt = compDict[(i,j)]['EarthlikeComp2_maxIntTimeCorrected']
    plt.plot(maxIntTimes,ypts,color=colors[i],marker=markers[i],label= str(int(starDistances[i])) + ' pc')
plt.xlabel('Maximum Integration Time (d)',weight='bold')
plt.ylabel('Integration Time Adjusted Completeness',weight='bold')
plt.xlim([0.1,100.])
plt.ylim([0.,1.05*np.max(maxpt)])
plt.legend()
plt.xscale('log')
plt.show(block=False)
plt.gcf().canvas.draw()
# Save to a File
date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'KeithlyCompvsIntTimeEarthLikeLOG' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
print('Done plotting KeithlyCompvsIntTimeEarthLike2LOG')
####

# for i in [0,1,2,3,4,5,6,7,8,9,10,11]:
#     print(maxIntTimes[i])
#     print(compDict[(4,i)]['EarthlikeComp2_maxIntTimeCorrected'])

