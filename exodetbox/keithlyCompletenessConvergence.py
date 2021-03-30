#### Completeness Convergence
import os
from projectedEllipse import *
import EXOSIMS.MissionSim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.random as random
import time
from astropy import constants as const
import astropy.units as u
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.planet_star_separation import planet_star_separation
import itertools
import csv
import corner
import datetime
import re

calcCompBool = False#True
calcBrownComp = True#True
plotBool = True#True
folder='./'
PPoutpath='./'


# "ppFact": 0.09,
# "erange": [0,0.35],
# "arange":[0.09, 12.4],
# "Rprange":[0.5, 11.6],
# "eta": 0.243731,
# "scaleOrbits": false,
# "constrainOrbits": true,
# "keepStarCatalog": false,
# "fillPhotometry": true,
# "explainFiltering": true,
# "whichPlanetPhaseFunction": "quasiLambertPhaseFunction",
# "PlanetPopulation": "SAG13",
# "StarCatalog": "EXOCAT1",
# "OpticalSystem": "Nemati",
# "ZodiacalLight": "Stark",
# "BackgroundSources": "GalaxiesFaintStars",
# "PlanetPhysicalModel": "Forecaster",
# "Observatory": "WFIRSTObservatoryL2",
# "TimeKeeping": " ",
# "PostProcessing": " ",
# "Completeness": "SubtypeCompleteness",
# "TargetList": " ",
# "SimulatedUniverse": "SAG13Universe",
# "SurveySimulation": "SLSQPScheduler",
# "SurveyEnsemble": "IPClusterEnsemble"


#### Randomly Generate Orbits #####################################################
dmag = 25. #29.0
dmag_upper = 25. #29.0
IWA_HabEx = 0.045*u.arcsec #taken from a Habex Script in units of mas
IWA2=0.150*u.arcsec #Suggested by dmitry as analahous to WFIRST
OWA_HabEx = 6.*u.arcsec #from the HabEx Standards Team Final Report
s_inner = 10.*u.pc.to('AU')*IWA_HabEx.to('rad').value
s_outer = 10.*u.pc.to('AU')*OWA_HabEx.to('rad').value
if calcCompBool == True:
    folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/exodetbox/exodetbox/scripts'))
    filename = 'HabEx_CKL2_PPKL2.json'
    filename = 'WFIRSTcycle6core.json'
    filename = 'HabEx_CSAG13_PPSAG13_compSubtype.json'
    #filename = 'HabEx_CSAG13_PPSAG13_compSubtypeHighEccen.json'
    scriptfile = os.path.join(folder,filename)
    sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
    PPop = sim.PlanetPopulation
    comp = sim.Completeness
    TL = sim.TargetList

    #starMass
    starMass = const.M_sun

    tmp = np.linspace(start=2,stop=7,num=50)
    ns = np.floor(10**tmp).astype('int')
    ns = [10**5]*1000
    #ns = np.append(ns,np.asarray([10**7,5*10**7,10**8,2*10**8]))
    executionTimeInHours = np.sum(ns)*352./7906043/60./60. #Uses a rate calc from the nu from dmag function so this is an underestimate
    # = [10**2,10**3,10**4,10**5,10**6,10**7,10**8,10**9]

    badsma = list()
    bader = list()
    badWr = list()
    badwr = list()
    badinc = list()
    #Planet to be removed
    ar = 0.6840751713914676*u.AU #sma
    er = 0.12443160036480415 #e
    Wr = 6.1198652952593 #W
    wr = 2.661645323283813 #w
    incr = 0.8803680245150818 #inc
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.1300859542315127*u.AU
    er = 0.23306811746716588
    Wr = 5.480292250277455
    wr = 2.4440871464730183
    incr = 1.197618937201339
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.294036031163908*u.AU
    er = 0.21199750171689644
    Wr = 4.2133670225641655
    wr = 5.065937897601312
    incr = 1.1190687587235382
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.066634748569753*u.AU
    er = 0.23799650191541077
    Wr = 3.448179965706161
    wr = 1.6569964041754257
    incr = 1.2153358668430527
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    #Removed because large imag components
    ar = 12.006090070519544*u.AU
    er = 0.011103778198982318
    Wr = 2.1899766657830924
    wr = 1.5709200609443608
    incr = 0.027074317262287463
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.551574519580606*u.AU
    er = 0.170910065686594
    Wr = 3.771587648556721
    wr = 2.6878240121964816
    incr = 1.2879427951683768
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.861296088681235*u.AU
    er = 0.3114402152030745
    Wr = 2.3598509674140193
    wr = 4.2429329404990686
    incr = 1.1280958647535413
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 12.148033510787947*u.AU
    er = 0.010235555197470866
    Wr = 4.722538539300574
    wr = 1.5705497614401225
    incr = 0.021803790927026074
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 6.972239708414176*u.AU
    er = 0.3314991502307672
    Wr = 1.0597252167081723
    wr = 4.95574654788906
    incr = 1.2972537064559475e-05
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 2.495183375446141*u.AU
    er = 0.31004037695684444
    Wr = 1.7283060888056982
    wr = 4.299935904170587
    incr = 1.4239980349024444
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.8228224720144062*u.AU
    er = 0.13559049816785979
    Wr = 5.166595438971132
    wr = 1.5443323432321039
    incr = 1.3484780997109205
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 3.450592571751904*u.AU
    er = 0.1558952135175637
    Wr = 4.784618309247885
    wr = 2.325605078468764
    incr = 1.4505349452903014
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.1467985619689993*u.AU
    er = 0.34948548176568894
    Wr = 2.7970886784168933
    wr = 2.3359556967427233
    incr = 1.2199924561080056
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.6352179069072269*u.AU
    er = 0.09586974068780153
    Wr = 3.552976712375306
    wr = 1.701536231563728
    incr = 0.8590721139355728
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.3564706741460606*u.AU
    er = 0.24583057956427973
    Wr = 1.0116250493190593
    wr = 4.811278309185773
    incr = 1.294999763050334
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 2.951153525962795*u.AU
    er = 0.26561769638150173
    Wr = 3.4042146343498336
    wr = 6.044912637660641
    incr = 1.5694821942530695
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.7055777115877997*u.AU
    er = 0.3240666909426986
    Wr = 2.797879927765729
    wr = 1.7129312058036235
    incr = 1.054011721061132
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 5.614277028833573*u.AU
    er = 0.2230520461215672
    Wr = 5.438854196656641
    wr = 3.1415912469920078
    incr = 0.02661160260601303
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 2.217718662163621*u.AU
    er = 0.1480757498429565
    Wr = 5.015130668169767
    wr = 2.957230430341968
    incr = 1.3667607352484137
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 4.035453468065543*u.AU
    er = 0.1581954368371899
    Wr = 1.4140158925707271
    wr = 1.9103432300736989
    incr = 1.4718168255798403
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.15041057353171625*u.AU
    er = 0.3203842722098759
    Wr = 6.243988121431742
    wr = 4.63108228811006
    incr = 0.0001091935232957475
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 3.0763794954240375*u.AU
    er = 0.16505427386305543
    Wr = 5.955818374387588
    wr = 4.592314142661588
    incr = 1.4426361628310929
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 2.6757045356377125*u.AU
    er = 0.29114050261607927
    Wr = 1.2176396419353828
    wr = 1.5702644420221834
    incr = 0.0008905192212513846
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.0925194858747385*u.AU
    er = 0.21163968677613113
    Wr = 4.923691080060667
    wr = 0.7333568494251783
    incr = 1.1848680979936506
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 6.746636274229815*u.AU
    er = 0.33851635044014633
    Wr = 2.1448606168163713
    wr = 5.502267850904032
    incr = 1.5121402765141776
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 2.517111527657932*u.AU
    er = 0.08382598849952719
    Wr = 4.673984139826626
    wr = 1.570681074314101
    incr = 0.12169021537376803
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 4.506837854851513*u.AU
    er = 0.271685417589626
    Wr = 3.996867374120977
    wr = 1.796968853389196
    incr = 1.4896312648876093
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.6786137208368946*u.AU
    er = 0.2032221187024536
    Wr = 5.06232026422141
    wr = 3.56470028130332
    incr = 0.8642182599195278
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 9.869683888785596*u.AU
    er = 0.2548065900572034
    Wr = 2.833401261309454
    wr = 5.3230258741710115
    incr = 1.5314443354564586
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 8.34515242427957*u.AU
    er = 0.14714878252795385
    Wr = 5.517934108063367
    wr = 4.216084030371432
    incr = 1.5221262810974039
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 9.04894440086429*u.AU
    er = 0.31459263385636643
    Wr = 3.736327581067437
    wr = 6.010778700507748
    incr = 1.5685725275431126
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.3988252117358604*u.AU
    er = 0.28186787841078925
    Wr = 5.615792260142231
    wr = 3.7089619126513256
    incr = 1.2637870267885019
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 5.3848786686132595*u.AU
    er = 0.29133954207853857
    Wr = 4.289188061612012
    wr = 3.1505639277706723
    incr = 0.0006492259688863911
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 11.895980404558737*u.AU
    er = 0.018840050513904052
    Wr = 1.0283860706865589
    wr = 1.5709078991163201
    incr = 0.04800981503666124
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.3094476271152629*u.AU
    er = 0.2711250243071124
    Wr = 5.115631061882926
    wr = 1.3259297237634526
    incr = 1.2876178024406189
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 5.166299645408495*u.AU
    er = 0.1903753362397787
    Wr = 3.2072506093199005
    wr = 3.7478866099072365
    incr = 1.4891542789719283
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 10.008552540742436*u.AU
    er = 0.05606373653528311
    Wr = 1.4007391289593978
    wr = 1.57069447107186
    incr = 0.10698051645814098
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.5409798886463959*u.AU
    er = 0.30676492030052993
    Wr = 5.857921930645498
    wr = 5.307240168892069
    incr = 0.769878373961876
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.605500943591448*u.AU
    er = 0.16706415516826403
    Wr = 1.8481192179296257
    wr = 2.519094056275588
    incr = 0.7889380976036461
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.7503833750938157*u.AU
    er = 0.2734245682373098
    Wr = 0.12183860376119011
    wr = 5.154228333847816
    incr = 1.0483193440388248
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.7245012861447346*u.AU
    er = 0.17399843722327732
    Wr = 0.14426559097376948
    wr = 3.7659883507843293
    incr = 0.9457737491072513
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.18653997507157397*u.AU
    er = 0.3465802451709428
    Wr = 6.181785763522057
    wr = 0.0005651180313909507
    incr = 0.0010349480533182742
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 2.7615839776057705*u.AU
    er = 0.2595007291593696
    Wr = 1.4885091984252292
    wr = 5.502473968680285
    incr = 1.424848171794064
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.9583923063078406*u.AU
    er = 0.3484634924577057
    Wr = 5.268822885025783
    wr = 5.989035619995281
    incr = 1.556040922232683
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 3.9581943562524957*u.AU
    er = 0.0693436154248298
    Wr = 0.09574653635835768
    wr = 1.5709046013745063
    incr = 0.10671483720358887
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 11.688245396237992*u.AU
    er = 0.058141088100520245
    Wr = 2.896141295376132
    wr = 6.22539866873384
    incr = 1.570016566156455
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.9235258615781312*u.AU
    er = 0.2090851902221292
    Wr = 2.5924123599606372
    wr = 4.293637186621654
    incr = 1.36783930865649
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 5.264348167698599*u.AU
    er = 0.020150468053313112
    Wr = 5.365394570812489
    wr = 4.712211103020281
    incr = 0.03332725074398546
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.6657140011779616*u.AU
    er = 0.17580409279346237
    Wr = 4.117260961566834
    wr = 3.9971421024222455
    incr = 1.3229936697699747
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 9.912955735484166*u.AU
    er = 0.1425773011226616
    Wr = 1.1730247507829423
    wr = 5.32299264480415
    incr = 1.5293367102154605
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.2898099529660976*u.AU
    er = 0.11308046946712258
    Wr = 6.171859980453564
    wr = 4.842391848326435
    incr = 1.2476256744951797
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.2803929648821695*u.AU
    er = 0.16005633452749468
    Wr = 3.4678095867770007
    wr = 5.387663943412752
    incr = 1.24505365352379
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.669335934876399*u.AU
    er = 0.3182819750374341
    Wr = 0.7909808228812486
    wr = 4.00225965659065
    incr = 1.3357949962985822
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.0071256685616026*u.AU
    er = 0.3418495770048641
    Wr = 5.432474669556755
    wr = 5.964435848286926
    incr = 1.0869859147290208
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.7697361406040367*u.AU
    er = 0.07429777530569893
    Wr = 3.5004214828964724
    wr = 4.712495202548991
    incr = 0.13922973023877017
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 6.313340025767313*u.AU
    er = 0.2576454818198619
    Wr = 3.4158756990310706
    wr = 3.373996536554064
    incr = 1.570512989385663
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.3851834621387076*u.AU
    er = 0.13498851689870783
    Wr = 3.3669396315432123
    wr = 4.208027551190937
    incr = 1.2707627263812369
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.4437771512506677*u.AU
    er = 0.2874578171280883
    Wr = 5.020186319639935
    wr = 4.769029405128971
    incr = 0.6571440788953004
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 6.26875781991671*u.AU
    er = 0.24973721429430076
    Wr = 0.45478727856267204
    wr = 3.105503669541162
    incr = 0.00020267176030142053
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 4.041375916710558*u.AU
    er = 0.05606066495843931
    Wr = 4.1463925733048885
    wr = 2.52687949245943
    incr = 1.4620511251536181
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 4.59886124193852*u.AU
    er = 0.28243623995611367
    Wr = 5.064557253859115
    wr = 1.6467402008634016
    incr = 0.6148407272753281
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 0.18014768496753042*u.AU
    er = 0.15219128249483882
    Wr = 1.1891414090936223
    wr = 6.136952566833045
    incr = 1.5706822377770073
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 1.254736294764488*u.AU
    er = 0.0449146520031593
    Wr = 5.9600430415183565
    wr = 4.71227719684002
    incr = 0.0984649979565771
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)
    ar = 3.022988960633602*u.AU
    er = 0.16209641024394342
    Wr = 2.1110264236350407
    wr = 0.855702648666915
    incr = 1.434416369881624
    badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)


    # num=69
    # plt.figure(num=num)
    # plt.
    # plt.show(block=False)
    # samples = np.asarray([badsma,bader,badWr,badwr,badinc]).T
    # figure = corner.corner(samples)
    # figure.savefig("corner.png")
    # plt.show(block=False)
    # plt.gcf().canvas.draw()


    comps = list()
    #n = 10**5 #Dean's nice computer can go up to 10**8 what can atuin go up to?
    for i in np.arange(len(ns)):
        print('Working on completeness for ' + str(ns[i]) + ' planets')
        start = time.time()
        inc, W, w = PPop.gen_angles(ns[i],None)
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
        sma, e, p, Rp = PPop.gen_plan_params(ns[i])

        #remove planet KOE with poor 
        for i in np.arange(len(badsma)):
            sma,e,W,w,inc = nukeKOE(sma,e,W,w,inc,badsma[i]*u.AU,bader[i],badWr[i],badwr[i],badinc[i])

        #### Classify Planets
        bini, binj, earthLike = comp.classifyPlanets(Rp, TL, np.arange(len(sma)), sma, e)
        sma = sma.to('AU').value
        ####

        #Separations
        s_circle = np.ones(len(sma))
        #Planet Periods
        periods = (2.*np.pi*np.sqrt((sma*u.AU)**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value

        nus, planetIsVisibleBool = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,False, s_inner, s_outer, dmag_upper, dmag_lower=None) #Calculate planet-star nu edges and visible regions
        ts = timeFromTrueAnomaly(nus,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges
        dt = ts[:,1:] - ts[:,:-1] #Calculate time region widths
        maxIntTime = 0.
        gtIntLimit = dt > maxIntTime #Create boolean array for inds
        totalVisibleTimePerTarget = np.nansum(np.multiply(np.multiply(dt-maxIntTime,planetIsVisibleBool.astype('int')),gtIntLimit),axis=1) #We subtract the int time from the fraction of observable time
        totalCompleteness = np.divide(totalVisibleTimePerTarget,periods*u.year.to('day')) # Fraction of time each planet is visible of its period
        #comps.append(list())
        stop = time.time()
        with open("./keithlyCompConvergence.csv", "a") as myfile:
            myfile.write(str(ns[i]) + "," + str(np.mean(totalCompleteness)) + "," + str(stop-start) + "\n")


if calcBrownComp == True:
    # folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts'))
    # filename = 'HabEx_CKL2_PPKL2.json'
    # filename = 'WFIRSTcycle6core.json'
    # filename = 'HabEx_CSAG13_PPSAG13_compSubtype.json'
    # #filename = 'HabEx_CSAG13_PPSAG13_compSubtypeHighEccen.json'
    # scriptfile = os.path.join(folder,filename)
    # sim = EXOSIMS.MissionSim.MissionSim(scriptfile=scriptfile,nopar=True)
    # PPop = sim.PlanetPopulation
    # comp = sim.Completeness
    # TL = sim.TargetList

    # comp = sim.Completeness()
    #Nplanets=1e3
    # Ns = [1e3,1e3+1,1e3+2,1e3+3,1e3+4,1e3+4\
    #     5*1e3,5*1e3+1,5*1e3+2,5*1e3+3,5*1e3+4,5*1e3+5,\
    #     1e4,1e4+1,1e4+2,1e4+3,1e4+4,1e4+5,\
    #     1e5,1e5+1,1e5+2,1e5+3,1e5+4,1e5+5,\
    #     1e6,1e6+1,1e6+2,1e6+3,1e6+4,1e6+5,1e6+6,\
    #     1e7,1e7+1,1e7+2,1e7+3,1e7+4,1e7+5,1e7+6,\
    #     1e8,1e8+1,1e8+2,1e8+3,1e8+4,1e8+5,1e8+6]
    Ns = np.floor(np.logspace(start=3,stop=9,num=200)).astype('int')
    Ns = np.zeros(10**3,dtype=int)+10**5
    comp = list()
    for i in np.arange(len(Ns)):
        if os.path.exists("/home/dean/.EXOSIMS/cache/SAG13ForecasterBrownCompleteness100000quasiLambertPhaseFunction4b2f4888cbba16dc9eb52ea645798829.comp"):
            os.remove("/home/dean/.EXOSIMS/cache/SAG13ForecasterBrownCompleteness100000quasiLambertPhaseFunction4b2f4888cbba16dc9eb52ea645798829.comp")
        else:
            print("The file does not exist") 
        start = time.time()
        sim = EXOSIMS.MissionSim.MissionSim(cachedir=None,Nplanets=Ns[i],**{"erange": [0,0.35],"arange":[0.09, 12.4],"Rprange":[0.5, 11.6],"scaleOrbits": False,"constrainOrbits": True,"whichPlanetPhaseFunction": "quasiLambertPhaseFunction",\
            "scienceInstruments": [{"name": "imagingEMCCD","QE": 0.9,"optics": 0.28,"FoV": 0.75,"pixelNumber": 105,"pixelSize": 13e-6,"sread": 0,"idark": 3e-5,"CIC": 1.3e-3,"texp": 100,"ENF": 1,"PCeff": 0.75}],\
            "starlightSuppressionSystems": [{"name": "VVC500","lam": 500,"IWA": 0.045,"OWA": 2.127,"ohTime": 0.1,"BW": 0.20,"optics": 0.95,"optics_comment": "contamination",
            "core_platescale": 0.1,"occ_trans": "$HOME/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_occ_trans_asec.fits",
            "core_thruput": "$HOME/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_core_thruput_asec.fits",           
            "core_mean_intensity": "$HOME/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_core_mean_intensity_asec.fits",
            "occ_trans_local": "$HOME/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_occ_trans_asec.fits",
            "core_thruput_local": "$HOME/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_core_thruput_asec.fits",           
            "core_mean_intensity_local": "$HOME/Documents/exosims/fitFilesFolder/HabExMay3/G_VC6_core_mean_intensity_asec.fits"}],
            "modules":{"PlanetPopulation": "SAG13","StarCatalog": "EXOCAT1","OpticalSystem": "Nemati","ZodiacalLight": "Stark","BackgroundSources": "GalaxiesFaintStars",\
            "PlanetPhysicalModel": "Forecaster","Observatory": "WFIRSTObservatoryL2","TimeKeeping": " ","PostProcessing": " ","Completeness": "BrownCompleteness","TargetList": " ",\
            "SimulatedUniverse": "SAG13Universe","SurveySimulation": " ","SurveyEnsemble": " "}})
        #CompMod = BrownCompleteness.BrownCompleteness(cachedir=None,Nplanets=Ns[i],**{"erange": [0,0.35],"arange":[0.09, 12.4],"Rprange":[0.5, 11.6],"scaleOrbits": False,"constrainOrbits": True,"modules":{"PlanetPopulation": "SAG13","StarCatalog": "EXOCAT1","OpticalSystem": "Nemati","PlanetPhysicalModel": "Forecaster","Completeness": "BrownCompleteness","SimulatedUniverse": "SAG13Universe","TargetList": " "}})
        comp.append(sim.SurveySimulation.Completeness.comp_calc(s_inner,s_outer,dmag_upper))#,-2)
        stop = time.time()
        print('Time: ' + str(stop-start))
        del sim
        with open("./brownCompConvergence3.csv", "a") as myfile: #htere is a nice Brown Comp Convergence file
            myfile.write(str(Ns[i]) + "," + str(comp[i]) + "," + str(stop-start) + " \n")



###########################################################





# Plot Convergence Plot
if plotBool==True:
    #### Load Keithly Completeness CSV File 
    with open('./keithlyCompConvergence.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = np.asarray(data,dtype='float') #convert data into an array
    sortInds = np.argsort(data[:,0]) #sort data by number of planets in comp calc
    data = data[sortInds] #rearrange data
    cumComp = [data[0,1]]
    cumPlans = [data[0,0]]
    for i in np.arange(len(data)-1):
        cumPlans.append(cumPlans[i]+data[i+1,0])
        cumComp.append((cumComp[i]*cumPlans[i] + data[i+1,0]*data[i+1,1])/(cumPlans[i+1]))
    #### Load Brown Completeness
    with open('./brownCompConvergence.csv', newline='') as f:
        reader = csv.reader(f)
        data2 = list(reader)
    data2 = np.asarray(data2,dtype='float') #convert data into an array
    sortInds = np.argsort(data2[:,0]) #sort data by number of planets in comp calc
    data2 = data2[sortInds] #rearrange data
    cumComp2 = [data2[0,1]]
    cumPlans2 = [data2[0,0]]
    for i in np.arange(len(data2)-1):
        cumPlans2.append(cumPlans2[i]+data2[i+1,0])
        cumComp2.append((cumComp2[i]*cumPlans2[i] + data2[i+1,0]*data2[i+1,1])/(cumPlans2[i+1]))


    num=9999
    plt.figure(num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    plt.plot(cumPlans,np.abs((cumComp-cumComp[-1]))/cumComp[-1]*100.,color='purple',label='Keithly')
    plt.plot(cumPlans2,np.abs((cumComp2-cumComp2[-1]))/cumComp2[-1]*100.,color='green',label='Brown')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Planets', weight='bold')
    plt.ylabel('% Error in Completeness', weight='bold')
    plt.legend()
    plt.show(block=False)
    plt.gcf().canvas.draw()
    # Save to a File
    date = str(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
    fname = 'ConvergencePercentErrorComp' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
    print('Done plotting Convergence Percentage')

    num=7777777777
    plt.figure(num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    plt.plot(cumPlans,np.abs((cumComp-cumComp[-1])),color='purple',label='Keithly')
    plt.plot(cumPlans2,np.abs((cumComp2-cumComp2[-1])),color='green',label='Brown')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Planets', weight='bold')
    plt.ylabel('Absolute Error in Completeness', weight='bold')
    plt.legend()
    plt.show(block=False)
    plt.gcf().canvas.draw()
    # Save to a File
    date = str(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
    fname = 'ConvergenceAbsErrorComp' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
    print('Done plotting Convergence bsolute')

    num=8888
    plt.figure(num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    plt.plot(cumPlans,cumComp,color='purple',label='Keithly')
    plt.plot(cumPlans2,cumComp2,color='green',label='Brown')
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Number of Planets', weight='bold')
    plt.ylabel('Completeness', weight='bold')
    plt.legend()
    plt.show(block=False)
    plt.gcf().canvas.draw()
    # Save to a File
    date = str(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
    fname = 'ConvergenceValueComp' + folder.split('/')[-1] + '_' + date
    plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
    plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
    plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
    print('Done plotting Convergence Value')



    #### Plot 10^5 Keithly Comp histograms and stuffs
    keithlyComp_sigma1 = data[:,1].std()
    keithlyComp_mean = data[:,1].mean()
    keithlyTime_sigma1 = data[:,2].std()
    keithlyTime_mean = data[:,2].mean()
    num=124
    plt.figure(num=num)
    plt.hist(data[:,1])
    plt.xlabel('Completeness',weight='bold')
    plt.ylabel('Frequency',weight='bold')
    plt.show(block=False)

    num=125
    plt.figure(num=num)
    plt.hist(data[:,2])
    plt.xlabel('Time (in seconds)',weight='bold')
    plt.ylabel('Frequency',weight='bold')
    plt.show(block=False)


#For find mean and std of datasets
with open('./brownCompConvergence3.csv', newline='') as f:
    reader = csv.reader(f)
    data3 = list(reader)
data3 = np.asarray(data3,dtype='float') #convert data into an array


#For find mean and std of datasets
with open('./keithlyCompConvergence.csv', newline='') as f:
    reader = csv.reader(f)
    data4 = list(reader)
data4 = np.asarray(data4,dtype='float') #convert data into an array


