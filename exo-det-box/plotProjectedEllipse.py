# plotProjectedEllipse.py
# Writen By: Dean Keithly
# Written on June 12, 2020

import numpy as np
import matplotlib.pyplot as plt
from projectedEllipse import xyz_3Dellipse
from projectedEllipse import timeFromTrueAnomaly
import astropy.units as u

##########################################################################################################
def plotProjectedEllipse(ind, sma, e, W, w, inc, Phi, dmajorp, dminorp, Op, num):
    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    ca = plt.gca()
    ca.axis('equal')
    ## Central Sun
    plt.scatter([0],[0],color='orange')
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    plt.plot(x_3Dellipse,y_3Dellipse,color='black')

    #plot 3D Ellipse Center
    plt.scatter(Op[0][ind],Op[1][ind],color='black')

    #ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
    ang2 = Phi[ind]
    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-')
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-')
    plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)
    ####

def plot3DEllipseto2DEllipseProjectionDiagram(ind, sma, e, W, w, inc, Op, Phi,\
    dmajorp, dminorp, num):
    """
    """
    plt.close(num)
    fig = plt.figure(num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    ax = fig.add_subplot(111, projection='3d')

    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    z_3Dellipse = r[2,0,:]
    ax.plot(x_3Dellipse,y_3Dellipse,z_3Dellipse,color='black',label='Planet Orbit',linewidth=2)
    min_z = np.min(z_3Dellipse)

    ## Central Sun
    ax.scatter(0,0,0,color='orange',marker='o',s=25) #of 3D ellipse
    ax.text(0,0,0.15*np.abs(min_z), 'F', None)
    ax.plot([0,0],[0,0],[0,1.3*min_z],color='orange',linestyle='--',linewidth=2) #connecting line
    ax.scatter(0,0,1.3*min_z,color='orange',marker='x',s=25) #of 2D ellipse
    ax.text(0,0,1.5*min_z, 'F\'', None)

    ## Plot 3D Ellipse semi-major/minor axis
    rper = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],0.) #planet position perigee
    rapo = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],np.pi) #planet position apogee
    ax.plot([rper[0][0],rapo[0][0]],[rper[1][0],rapo[1][0]],[rper[2][0],rapo[2][0]],color='purple', linestyle='-',linewidth=2) #3D Ellipse Semi-major axis
    ax.scatter(rper[0][0],rper[1][0],rper[2][0],color='grey',marker='D',s=25) #3D Ellipse Perigee Diamond
    ax.text(1.2*rper[0][0],1.2*rper[1][0],rper[2][0], 'A', None)
    ax.scatter(rper[0][0],rper[1][0],1.3*min_z,color='blue',marker='D',s=25) #2D Ellipse Perigee Diamond
    ax.text(1.1*rper[0][0],1.1*rper[1][0],1.3*min_z, 'A\'', None)
    ax.plot([rper[0][0],rper[0][0]],[rper[1][0],rper[1][0]],[rper[2][0],1.3*min_z],color='grey',linestyle='--',linewidth=2) #3D to 2D Ellipse Perigee Diamond
    ax.scatter(rapo[0][0],rapo[1][0],rapo[2][0],color='grey', marker='D',s=25) #3D Ellipse Apogee Diamond
    ax.text(1.1*rapo[0][0],1.1*rapo[1][0],1.2*rapo[2][0], 'B', None)
    ax.scatter(rapo[0][0],rapo[1][0],1.3*min_z,color='blue',marker='D',s=25) #2D Ellipse Perigee Diamond
    ax.text(1.1*rapo[0][0],1.1*rapo[1][0],1.3*min_z, 'B\'', None)

    ax.plot([rapo[0][0],rapo[0][0]],[rapo[1][0],rapo[1][0]],[rapo[2][0],1.3*min_z],color='grey',linestyle='--',linewidth=2) #3D to 2D Ellipse Apogee Diamond
    rbp = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],np.arccos((np.cos(np.pi/2)-e[ind])/(1-e[ind]*np.cos(np.pi/2)))) #3D Ellipse E=90
    rbm = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],-np.arccos((np.cos(-np.pi/2)-e[ind])/(1-e[ind]*np.cos(-np.pi/2)))) #3D Ellipse E=-90
    ax.plot([rbp[0][0],rbm[0][0]],[rbp[1][0],rbm[1][0]],[rbp[2][0],rbm[2][0]],color='purple', linestyle='-',linewidth=2) #
    ax.scatter(rbp[0][0],rbp[1][0],rbp[2][0],color='grey',marker='D',s=25) #3D ellipse minor +
    ax.text(1.1*rbp[0][0],1.1*rbp[1][0],1.2*rbp[2][0], 'C', None)
    ax.scatter(rbp[0][0],rbp[1][0],1.3*min_z,color='blue',marker='D',s=25) #2D ellipse minor+ projection
    ax.text(1.1*rbp[0][0],1.1*rbp[1][0],1.3*min_z, 'C\'', None)
    ax.plot([rbp[0][0],rbp[0][0]],[rbp[1][0],rbp[1][0]],[rbp[2][0],1.3*min_z],color='grey',linestyle='--',linewidth=2) #3D to 2D Ellipse minor + Diamond
    ax.scatter(rbm[0][0],rbm[1][0],rbm[2][0],color='grey', marker='D',s=25) #3D ellipse minor -
    ax.text(1.1*rbm[0][0],0.5*(rbm[1][0]-Op[1][ind]),rbm[2][0]+0.05, 'D', None)
    ax.scatter(rbm[0][0],rbm[1][0],1.3*min_z,color='blue', marker='D',s=25) #2D ellipse minor- projection
    ax.text(1.1*rbm[0][0],0.5*(rbm[1][0]-Op[1][ind]),1.3*min_z, 'D\'', None)
    ax.plot([rbm[0][0],rbm[0][0]],[rbm[1][0],rbm[1][0]],[rbm[2][0],1.3*min_z],color='grey',linestyle='--',linewidth=2) #3D to 2D Ellipse minor - Diamond

    ## Plot K, H, P
    ax.scatter(0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            0.6*(rapo[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,color='green', marker='x',s=36) #Point along OB, point H
    ax.text(0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            0.6*(rapo[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2+0.1,'H', None) #Point along OB, point H
    xscaletmp = np.sqrt(1-.6**2)
    ax.scatter(xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            xscaletmp*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,color='green',marker='x',s=36) #point along OC, point K
    ax.text(xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            xscaletmp*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2+0.1,'K',None) #point along OC, point K
    angtmp = np.arctan2(0.6,xscaletmp)
    ax.scatter(np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2,\
        np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2,\
        np.cos(angtmp)*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rapo[2][0] - (rper[2][0] + rapo[2][0])/2)*np.sin(angtmp) + (rper[2][0] + rapo[2][0])/2,color='green',marker='o',s=25) #Point P on 3D Ellipse
    ax.text(np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2,\
        np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2,\
        np.cos(angtmp)*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rapo[2][0] - (rper[2][0] + rapo[2][0])/2)*np.sin(angtmp) + (rper[2][0] + rapo[2][0])/2+0.1,'P',None) #Point P on 3D Ellipse
    ## Plot KP, HP
    ax.plot([0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2],\
            [0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2],\
            [0.6*(rapo[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,np.cos(angtmp)*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rapo[2][0] - (rper[2][0] + rapo[2][0])/2)*np.sin(angtmp) + (rper[2][0] + rapo[2][0])/2],linestyle=':',color='black') #H to P line
    ax.plot([xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2],\
            [xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2],\
            [xscaletmp*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,np.cos(angtmp)*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rapo[2][0] - (rper[2][0] + rapo[2][0])/2)*np.sin(angtmp) + (rper[2][0] + rapo[2][0])/2],linestyle=':',color='black') #K to P line
    ## Plot K', H', P'
    ax.scatter(0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            1.3*min_z,color='magenta', marker='x',s=36) #Point along O'B', point H'
    ax.text(0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            1.3*min_z-0.1,'H\'',None) #Point along O'B', point H'
    xscaletmp = np.sqrt(1-.6**2)
    ax.scatter(xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            1.3*min_z,color='magenta',marker='x',s=36) #point along O'C', point K'
    ax.text(xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,\
            xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,\
            1.3*min_z-0.1,'K\'',None) #point along O'C', point K'
    angtmp = np.arctan2(0.6,xscaletmp)
    ax.scatter(np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2,\
        np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2,\
        1.3*min_z,color='magenta',marker='o',s=25) #Point P' on 2D Ellipse
    ax.text(np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2,\
        np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2,\
        1.3*min_z-0.1,'P\'',None) #Point P' on 2D Ellipse
    ## Plot K'P', H'P'
    ax.plot([0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2],\
            [0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2],\
            [1.3*min_z,1.3*min_z],linestyle=':',color='black') #H to P line
    ax.plot([xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2],\
            [xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2],\
            [1.3*min_z,1.3*min_z],linestyle=':',color='black') #K to P line
    ## Plot PP', KK', HH'
    ax.plot([np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2,np.cos(angtmp)*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rapo[0][0] - (rper[0][0] + rapo[0][0])/2)*np.sin(angtmp) + (rper[0][0] + rapo[0][0])/2],\
        [np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2,np.cos(angtmp)*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rapo[1][0] - (rper[1][0] + rapo[1][0])/2)*np.sin(angtmp) + (rper[1][0] + rapo[1][0])/2],\
        [np.cos(angtmp)*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rapo[2][0] - (rper[2][0] + rapo[2][0])/2)*np.sin(angtmp) + (rper[2][0] + rapo[2][0])/2,1.3*min_z],color='black',linestyle=':') #PP'
    ax.plot([0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,0.6*(rapo[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2],\
            [0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,0.6*(rapo[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2],\
            [0.6*(rapo[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,1.3*min_z],color='black',linestyle=':') #HH'
    ax.plot([xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2,xscaletmp*(rbp[0][0] - (rper[0][0] + rapo[0][0])/2) + (rper[0][0] + rapo[0][0])/2],\
            [xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2,xscaletmp*(rbp[1][0] - (rper[1][0] + rapo[1][0])/2) + (rper[1][0] + rapo[1][0])/2],\
            [xscaletmp*(rbp[2][0] - (rper[2][0] + rapo[2][0])/2) + (rper[2][0] + rapo[2][0])/2,1.3*min_z],color='black',linestyle=':') #KK'

    ## Plot Conjugate Diameters
    ax.plot([rbp[0][0],rbm[0][0]],[rbp[1][0],rbm[1][0]],[1.3*min_z,1.3*min_z],color='blue',linestyle='-',linewidth=2) #2D ellipse minor+ projection
    ax.plot([rper[0][0],rapo[0][0]],[rper[1][0],rapo[1][0]],[1.3*min_z,1.3*min_z],color='blue',linestyle='-',linewidth=2) #2D Ellipse Perigee Diamond

    ## Plot Ellipse Center
    ax.scatter((rper[0][0] + rapo[0][0])/2,(rper[1][0] + rapo[1][0])/2,(rper[2][0] + rapo[2][0])/2,color='grey',marker='o',s=36) #3D Ellipse
    ax.text(1.2*(rper[0][0] + rapo[0][0])/2,1.2*(rper[1][0] + rapo[1][0])/2,1.31*(rper[2][0] + rapo[2][0])/2, 'O', None)
    ax.scatter(Op[0][ind],Op[1][ind], 1.3*min_z, color='grey', marker='o',s=25) #2D Ellipse Center
    ax.text(1.2*(rper[0][0] + rapo[0][0])/2,1.2*(rper[1][0] + rapo[1][0])/2,1.4*min_z, 'O\'', None)
    ax.plot([(rper[0][0] + rapo[0][0])/2,Op[0][ind]],[(rper[1][0] + rapo[1][0])/2,Op[1][ind]],[(rper[2][0] + rapo[2][0])/2,1.3*min_z],color='grey',linestyle='--',linewidth=2) #Plot ) to )''


    #ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
    ang2 = Phi[ind]
    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    ax.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    ax.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)

    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    ax.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],[1.3*min_z,1.3*min_z],color='purple',linestyle='-',linewidth=2)
    ax.scatter([dmajorpx1,dmajorpx2,dminorpx1,dminorpx2],[dmajorpy1,dmajorpy2,dminorpy1,dminorpy2],[1.3*min_z,1.3*min_z,1.3*min_z,1.3*min_z],color='black',marker='o',s=25,zorder=6)
    ax.text(1.05*dmajorpx1,1.05*dmajorpy1,1.3*min_z, 'I', None)#(dmajorpx1,dmajorpy1,0))
    ax.text(1.1*dmajorpx2,1.1*dmajorpy2,1.3*min_z, 'R', None)#(dmajorpx2,dmajorpy2,0))
    ax.text(1.05*dminorpx1,0.1*(dminorpy1-Op[1][ind]),1.3*min_z, 'S', None)#(dminorpx1,dminorpy1,0))
    ax.text(1.05*dminorpx2,1.05*dminorpy2,1.3*min_z, 'T', None)#(dminorpx2,dminorpy2,0))
    #ax.text(x,y,z, label, zdir)
    x_projEllipse = Op[0][ind] + dmajorp[ind]*np.cos(vs)*np.cos(ang2) - dminorp[ind]*np.sin(vs)*np.sin(ang2)
    y_projEllipse = Op[1][ind] + dmajorp[ind]*np.cos(vs)*np.sin(ang2) + dminorp[ind]*np.sin(vs)*np.cos(ang2)
    ax.plot(x_projEllipse,y_projEllipse,1.3*min_z*np.ones(len(vs)), color='red', linestyle='-',zorder=5,linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(False)
    #artificial box
    xmax = np.max([np.abs(rper[0][0]),np.abs(rapo[0][0]),np.abs(1.3*min_z)])
    ax.scatter([-xmax,xmax],[-xmax,xmax],[-0.2-np.abs(1.3*min_z),0.2+1.3*min_z],color=None,alpha=0)
    ax.set_xlim3d(-0.99*xmax+Op[0][ind],0.99*xmax+Op[0][ind])
    ax.set_ylim3d(-0.99*xmax+Op[1][ind],0.99*xmax+Op[1][ind])
    ax.set_zlim3d(-0.99*xmax,0.99*xmax)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) #remove background color
    ax.set_axis_off() #removes axes
    plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)
    ####

def plotEllipseMajorAxisFromConjugate(ind, sma, e, W, w, inc, Op, Phi,\
    dmajorp, dminorp, num):
    """ Plots the Q and Q' points as well as teh line 
    """
    plt.close(num)
    fig = plt.figure(num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    ax = plt.gca()

    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    z_3Dellipse = r[2,0,:]
    ax.plot(x_3Dellipse,y_3Dellipse,color='black',label='Planet Orbit',linewidth=2)
    min_z = np.min(z_3Dellipse)

    ## Central Sun
    ax.scatter(0,0,color='orange',marker='x',s=25,zorder=20) #of 2D ellipse
    ax.text(0-.1,0-.1, 'F\'', None)

    ## Plot 3D Ellipse semi-major/minor axis
    rper = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],0.) #planet position perigee
    rapo = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],np.pi) #planet position apogee
    ax.scatter(rper[0][0],rper[1][0],color='blue',marker='D',s=25,zorder=25) #2D Ellipse Perigee Diamond
    ax.text(1.1*rper[0][0],1.1*rper[1][0], 'A\'', None)
    ax.scatter(rapo[0][0],rapo[1][0],color='blue',marker='D',s=25,zorder=25) #2D Ellipse Perigee Diamond
    ax.text(1.1*rapo[0][0]-0.1,1.1*rapo[1][0], 'B\'', None)

    rbp = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],np.arccos((np.cos(np.pi/2)-e[ind])/(1-e[ind]*np.cos(np.pi/2)))) #3D Ellipse E=90
    rbm = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],-np.arccos((np.cos(-np.pi/2)-e[ind])/(1-e[ind]*np.cos(-np.pi/2)))) #3D Ellipse E=-90
    ax.plot([rbp[0][0],rbm[0][0]],[rbp[1][0],rbm[1][0]],color='purple', linestyle='-',linewidth=2) #
    ax.scatter(rbp[0][0],rbp[1][0],color='blue',marker='D',s=25,zorder=20) #2D ellipse minor+ projection
    ax.text(1.1*rbp[0][0]-.01,1.1*rbp[1][0]-.05, 'C\'', None)
    ax.scatter(rbm[0][0],rbm[1][0],color='blue', marker='D',s=25,zorder=20) #2D ellipse minor- projection
    ax.text(1.1*rbm[0][0],0.5*(rbm[1][0]-Op[1][ind])-.05, 'D\'', None)

    ## Plot QQ' Line
    #rapo[0][0],rapo[1][0] #B'
    #rbp[0][0],rbp[1][0] #C'
    #Op[0][ind],Op[1][ind] #O'
    tmp = np.asarray([-(rbp[1][0]-Op[1][ind]),(rbp[0][0]-Op[0][ind])])
    QQp_hat = tmp/np.linalg.norm(tmp)
    dOpCp = np.sqrt((rbp[0][0]-Op[0][ind])**2 + (rbp[1][0]-Op[1][ind])**2)
    #Q = Bp - dOpCp*QQp_hat
    Qx = rapo[0][0] - dOpCp*QQp_hat[0]
    Qy = rapo[1][0] - dOpCp*QQp_hat[1]
    #Qp = Bp + DOpCp*QQp_hat
    Qpx = rapo[0][0] + dOpCp*QQp_hat[0]
    Qpy = rapo[1][0] + dOpCp*QQp_hat[1]
    ax.plot([Op[0][ind],Qx],[Op[1][ind],Qy],color='black',linestyle='-',linewidth=2,zorder=29) #OpQ
    ax.plot([Op[0][ind],Qpx],[Op[1][ind],Qpy],color='black',linestyle='-',linewidth=2,zorder=29) #OpQp
    ax.plot([Qx,Qpx],[Qy,Qpy],color='grey',linestyle='-',linewidth=2,zorder=29)
    ax.scatter([Qx,Qpx],[Qy,Qpy],color='grey',marker='s',s=36,zorder=30)
    ax.text(Qx,Qy-0.1,'Q', None)
    ax.text(Qpx,Qpy+0.05,'Q\'', None)

    ## Plot Conjugate Diameters
    ax.plot([rbp[0][0],rbm[0][0]],[rbp[1][0],rbm[1][0]],color='blue',linestyle='-',linewidth=2) #2D ellipse minor+ projection
    ax.plot([rper[0][0],rapo[0][0]],[rper[1][0],rapo[1][0]],color='blue',linestyle='-',linewidth=2) #2D Ellipse Perigee Diamond

    ## Plot Ellipse Center
    ax.scatter(Op[0][ind],Op[1][ind], color='grey', marker='o',s=25,zorder=30) #2D Ellipse Center
    ax.text(1.2*(rper[0][0] + rapo[0][0])/2,1.2*(rper[1][0] + rapo[1][0])/2+0.05, 'O\'', None)

    #ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
    ang2 = Phi[ind]
    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    ax.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-',linewidth=2)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    ax.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-',linewidth=2)

    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    ax.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-',linewidth=2)
    ax.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-',linewidth=2)
    ax.scatter([dmajorpx1,dmajorpx2,dminorpx1,dminorpx2],[dmajorpy1,dmajorpy2,dminorpy1,dminorpy2],color='black',marker='o',s=25,zorder=6)
    ax.text(1.05*dmajorpx1,1.05*dmajorpy1, 'I', None)
    ax.text(1.1*dmajorpx2,1.1*dmajorpy2, 'R', None)
    ax.text(1.05*dminorpx1,0.1*(dminorpy1-Op[1][ind])-.05, 'S', None)
    ax.text(1.05*dminorpx2-0.1,1.05*dminorpy2-.075, 'T', None)
    x_projEllipse = Op[0][ind] + dmajorp[ind]*np.cos(vs)*np.cos(ang2) - dminorp[ind]*np.sin(vs)*np.sin(ang2)
    y_projEllipse = Op[1][ind] + dmajorp[ind]*np.cos(vs)*np.sin(ang2) + dminorp[ind]*np.sin(vs)*np.cos(ang2)
    ax.plot(x_projEllipse,y_projEllipse, color='red', linestyle='-',zorder=5,linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(False)
    xmax = np.max([np.abs(rper[0][0]),np.abs(rapo[0][0]),np.abs(1.3*min_z), np.abs(Qpx), np.abs(Qx)])
    ax.scatter([-xmax,xmax],[-xmax,xmax],color=None,alpha=0)
    ax.set_xlim(-0.99*xmax+Op[0][ind],0.99*xmax+Op[0][ind])
    ax.set_ylim(-0.99*xmax+Op[1][ind],0.99*xmax+Op[1][ind])
    ax.set_axis_off() #removes axes
    ax.axis('equal')
    plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)

def plotDerotatedEllipse(ind, sma, e, W, w, inc, Phi, dmajorp, dminorp, Op, x, y, num=879):
    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    ca = plt.gca()
    ca.axis('equal')
    plt.scatter([0],[0],color='orange')
    ## Plot 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    plt.plot(x_3Dellipse,y_3Dellipse,color='black')
    ## Plot 3D Ellipse Center
    plt.scatter(Op[0][ind],Op[1][ind],color='black')
    ## Plot Rotated Ellipse
    #ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
    ang2 = Phi[ind]
    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-')
    #new plot stuff
    Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
    plt.plot([-dmajorp[ind],dmajorp[ind]],[0,0],color='purple',linestyle='--') #major
    plt.plot([0,0],[-dminorp[ind],dminorp[ind]],color='purple',linestyle='--') #minor
    xellipsetmp = dmajorp[ind]*np.cos(Erange)
    yellipsetmp = dminorp[ind]*np.sin(Erange)
    plt.plot(xellipsetmp,yellipsetmp,color='black')
    plt.scatter(x[ind],y[ind],color='orange',marker='x')

    c_ae = dmajorp[ind]*np.sqrt(1-dminorp[ind]**2/dmajorp[ind]**2)
    plt.scatter([-c_ae,c_ae],[0,0],color='blue')

    plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)

def plotReorientationMethod(ind, sma, e, W, w, inc, x, y, Phi, Op, dmajorp, dminorp,\
    minSepPoints_x, minSepPoints_y, num):
    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    ca = plt.gca()
    ca.axis('equal')
    plt.scatter([0],[0],color='orange')
    ## Plot 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    plt.plot(x_3Dellipse,y_3Dellipse,color='black')
    ## Plot 3D Ellipse Center
    plt.scatter(Op[0][ind],Op[1][ind],color='black')
    ## Plot Rotated Ellipse
    #ang2 = (theta_OpQ_X[ind]+theta_OpQp_X[ind])/2
    ang2 = Phi[ind]
    dmajorpx1 = Op[0][ind] + dmajorp[ind]*np.cos(ang2)
    dmajorpy1 = Op[1][ind] + dmajorp[ind]*np.sin(ang2)
    dmajorpx2 = Op[0][ind] + dmajorp[ind]*np.cos(ang2+np.pi)
    dmajorpy2 = Op[1][ind] + dmajorp[ind]*np.sin(ang2+np.pi)
    dminorpx1 = Op[0][ind] + dminorp[ind]*np.cos(ang2+np.pi/2)
    dminorpy1 = Op[1][ind] + dminorp[ind]*np.sin(ang2+np.pi/2)
    dminorpx2 = Op[0][ind] + dminorp[ind]*np.cos(ang2-np.pi/2)
    dminorpy2 = Op[1][ind] + dminorp[ind]*np.sin(ang2-np.pi/2)
    plt.plot([Op[0][ind],dmajorpx1],[Op[1][ind],dmajorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dmajorpx2],[Op[1][ind],dmajorpy2],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dminorpx1],[Op[1][ind],dminorpy1],color='purple',linestyle='-')
    plt.plot([Op[0][ind],dminorpx2],[Op[1][ind],dminorpy2],color='purple',linestyle='-')
    #new plot stuff
    Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
    plt.plot([-dmajorp[ind],dmajorp[ind]],[0,0],color='purple',linestyle='--') #major
    plt.plot([0,0],[-dminorp[ind],dminorp[ind]],color='purple',linestyle='--') #minor
    xellipsetmp = dmajorp[ind]*np.cos(Erange)
    yellipsetmp = dminorp[ind]*np.sin(Erange)
    plt.plot(xellipsetmp,yellipsetmp,color='black')
    plt.scatter(x[ind],y[ind],color='orange',marker='x')

    c_ae = dmajorp[ind]*np.sqrt(1-dminorp[ind]**2/dmajorp[ind]**2)
    plt.scatter([-c_ae,c_ae],[0,0],color='blue')
    plt.scatter(minSepPoints_x[ind],minSepPoints_y[ind],color='magenta')
    ux = np.cos(Phi[ind])*minSepPoints_x[ind] - np.sin(Phi[ind])*minSepPoints_y[ind] + Op[0][ind] 
    uy = np.sin(Phi[ind])*minSepPoints_x[ind] + np.cos(Phi[ind])*minSepPoints_y[ind] + Op[1][ind] 
    plt.scatter(ux,uy,color='green')

    plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)

def plotDerotatedIntersectionsMinMaxStarLocBounds(ind, sma, e, W, w, inc, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
    minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
    lmaxSepPoints_x, lmaxSepPoints_y, twoIntSameYInds,\
    maxSepPoints_x, maxSepPoints_y, twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
    type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num):
    """
    """
    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    ca = plt.gca()
    ca.axis('equal')
    #DELETEplt.scatter([xreal[ind,0],xreal[ind,1],xreal[ind,2],xreal[ind,3]], [yreal[ind,0],yreal[ind,1],yreal[ind,2],yreal[ind,3]], color='purple')
    plt.scatter([0],[0],color='orange')
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    #new plot stuff
    Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
    plt.plot([-dmajorp[ind],dmajorp[ind]],[0,0],color='purple',linestyle='--') #major
    plt.plot([0,0],[-dminorp[ind],dminorp[ind]],color='purple',linestyle='--') #minor
    xellipsetmp = dmajorp[ind]*np.cos(Erange)
    yellipsetmp = dminorp[ind]*np.sin(Erange)
    plt.plot(xellipsetmp,yellipsetmp,color='black')
    plt.scatter(x[ind],y[ind],color='orange',marker='x')
    if ind in only2RealInds[typeInds0]:
        plt.scatter(x[ind],y[ind],edgecolors='teal',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds1]:
        plt.scatter(x[ind],y[ind],edgecolors='red',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds2]:
        plt.scatter(x[ind],y[ind],edgecolors='blue',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds3]:
        plt.scatter(x[ind],y[ind],edgecolors='magenta',marker='o',s=64,facecolors='none')

    c_ae = dmajorp[ind]*np.sqrt(1-dminorp[ind]**2/dmajorp[ind]**2)
    plt.scatter([-c_ae,c_ae],[0,0],color='blue')

    # #Plot Min Sep Circle
    # x_circ = minSep[ind]*np.cos(vs)
    # y_circ = minSep[ind]*np.sin(vs)
    # plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='teal')
    # #Plot Max Sep Circle
    # x_circ2 = maxSep[ind]*np.cos(vs)
    # y_circ2 = maxSep[ind]*np.sin(vs)
    # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='red')
    #Plot Min Sep Ellipse Intersection
    plt.scatter(minSepPoints_x[ind],minSepPoints_y[ind],color='teal',marker='D')
    #Plot Max Sep Ellipse Intersection
    plt.scatter(maxSepPoints_x[ind],maxSepPoints_y[ind],color='red',marker='D')

    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        # #Plot lminSep Circle
        # x_circ2 = lminSep[tind]*np.cos(vs)
        # y_circ2 = lminSep[tind]*np.sin(vs)
        # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='magenta')
        # #Plot lmaxSep Circle
        # x_circ2 = lmaxSep[tind]*np.cos(vs)
        # y_circ2 = lmaxSep[tind]*np.sin(vs)
        # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='gold')
        #### Plot Local Min
        plt.scatter(lminSepPoints_x[tind], lminSepPoints_y[tind],color='magenta',marker='D')
        #### Plot Local Max Points
        plt.scatter(lmaxSepPoints_x[tind], lmaxSepPoints_y[tind],color='gold',marker='D')

    #### r Intersection test
    x_circ2 = np.cos(vs)
    y_circ2 = np.sin(vs)
    plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='green')
    #### Intersection Points
    if ind in yrealAllRealInds[fourIntInds]:
        yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
        plt.scatter(fourInt_x[yind],fourInt_y[yind], color='green',marker='o')
    elif ind in yrealAllRealInds[twoIntSameYInds]: #Same Y
        yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
        plt.scatter(twoIntSameY_x[yind],twoIntSameY_y[yind], color='green',marker='o')
    elif ind in yrealAllRealInds[twoIntOppositeXInds]: #Same X
        yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
        plt.scatter(twoIntOppositeX_x[yind],twoIntOppositeX_y[yind], color='green',marker='o')
        #### Type0
    elif ind in only2RealInds[type0_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
        print('plotted')
    elif ind in only2RealInds[type0_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type0_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type0_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type0_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
        #### Type1
    elif ind in only2RealInds[type1_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type1_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type1_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type1_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type1_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
        #### Type2
    elif ind in only2RealInds[type2_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type2_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type2_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type2_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type2_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
        #### Type3
    elif ind in only2RealInds[type3_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type3_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type3_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type3_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type3_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')

    # Plot Star Location Type Dividers
    xran = np.linspace(start=(dmajorp[ind]*(dmajorp[ind]**2*(dmajorp[ind] - dminorp[ind])*(dmajorp[ind] + dminorp[ind]) - dminorp[ind]**2*np.sqrt(3*dmajorp[ind]**4 + 2*dmajorp[ind]**2*dminorp[ind]**2 + 3*dminorp[ind]**4))/(2*(dmajorp[ind]**4 + dminorp[ind]**4))),\
        stop=(dmajorp[ind]*(dmajorp[ind]**2*(dmajorp[ind] - dminorp[ind])*(dmajorp[ind] + dminorp[ind]) + dminorp[ind]**2*np.sqrt(3*dmajorp[ind]**4 + 2*dmajorp[ind]**2*dminorp[ind]**2 + 3*dminorp[ind]**4))/(2*(dmajorp[ind]**4 + dminorp[ind]**4))), num=3, endpoint=True)
    ylineQ1 = xran*dmajorp[ind]/dminorp[ind] - dmajorp[ind]**2/(2*dminorp[ind]) + dminorp[ind]/2 #between first quadrant a,b
    ylineQ4 = -xran*dmajorp[ind]/dminorp[ind] + dmajorp[ind]**2/(2*dminorp[ind]) - dminorp[ind]/2 #between 4th quadrant a,b
    plt.plot(xran, ylineQ1, color='brown', linestyle='-.', )
    plt.plot(-xran, ylineQ4, color='grey', linestyle='-.')
    plt.plot(-xran, ylineQ1, color='orange', linestyle='-.')
    plt.plot(xran, ylineQ4, color='red', linestyle='-.')
    plt.xlim([-1.2*dmajorp[ind],1.2*dmajorp[ind]])
    plt.ylim([-1.2*dminorp[ind],1.2*dminorp[ind]])
    plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)

def plotDerotatedExtrema(ind, sma, e, W, w, inc, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
    maxSepPoints_x, maxSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
    minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
    twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, num):
    """
    """
    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    ca = plt.gca()
    ca.axis('equal')
    plt.scatter([0],[0],color='orange',zorder=25)
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    #new plot stuff
    Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
    plt.plot([-dmajorp[ind],dmajorp[ind]],[0,0],color='purple',linestyle='--') #major
    plt.plot([0,0],[-dminorp[ind],dminorp[ind]],color='purple',linestyle='--') #minor
    xellipsetmp = dmajorp[ind]*np.cos(Erange)
    yellipsetmp = dminorp[ind]*np.sin(Erange)
    plt.plot(xellipsetmp,yellipsetmp,color='black')
    plt.scatter(x[ind],y[ind],color='orange',marker='x',zorder=30)
    if ind in only2RealInds[typeInds0]:
        plt.scatter(x[ind],y[ind],edgecolors='teal',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds1]:
        plt.scatter(x[ind],y[ind],edgecolors='red',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds2]:
        plt.scatter(x[ind],y[ind],edgecolors='blue',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds3]:
        plt.scatter(x[ind],y[ind],edgecolors='magenta',marker='o',s=64,facecolors='none')

    c_ae = dmajorp[ind]*np.sqrt(1-dminorp[ind]**2/dmajorp[ind]**2)
    plt.scatter([-c_ae,c_ae],[0,0],color='blue')

    # #Plot Min Sep Circle
    # x_circ = minSep[ind]*np.cos(vs)
    # y_circ = minSep[ind]*np.sin(vs)
    # plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='teal')
    # #Plot Max Sep Circle
    # x_circ2 = maxSep[ind]*np.cos(vs)
    # y_circ2 = maxSep[ind]*np.sin(vs)
    # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='red')
    #Plot Min Sep Ellipse Intersection
    plt.scatter(minSepPoints_x[ind],minSepPoints_y[ind],color='teal',marker='D',zorder=25)
    #Plot Max Sep Ellipse Intersection
    plt.scatter(maxSepPoints_x[ind],maxSepPoints_y[ind],color='red',marker='D',zorder=25)
    #### Plot star to min line
    plt.plot([x[ind],minSepPoints_x[ind]], [y[ind],minSepPoints_y[ind]],color='teal',zorder=25)
    #### Plot star to max line
    plt.plot([x[ind],maxSepPoints_x[ind]], [y[ind],maxSepPoints_y[ind]],color='red',zorder=25)

    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        # #Plot lminSep Circle
        # x_circ2 = lminSep[tind]*np.cos(vs)
        # y_circ2 = lminSep[tind]*np.sin(vs)
        # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='magenta')
        # #Plot lmaxSep Circle
        # x_circ2 = lmaxSep[tind]*np.cos(vs)
        # y_circ2 = lmaxSep[tind]*np.sin(vs)
        # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='gold')
        #### Plot Local Min
        plt.scatter(lminSepPoints_x[tind], lminSepPoints_y[tind],color='magenta',marker='D',zorder=25)
        #### Plot Local Max Points
        plt.scatter(lmaxSepPoints_x[tind], lmaxSepPoints_y[tind],color='gold',marker='D',zorder=25)
        #### Plot star to local min line
        plt.plot([x[ind],lminSepPoints_x[tind]], [y[ind],lminSepPoints_y[tind]],color='magenta',zorder=25)
        #### Plot star to local max line
        plt.plot([x[ind],lmaxSepPoints_x[tind]], [y[ind],lmaxSepPoints_y[tind]],color='gold',zorder=25)

    plt.xlim([-1.2*dmajorp[ind],1.2*dmajorp[ind]])
    plt.ylim([-1.2*dminorp[ind],1.2*dminorp[ind]])
    plt.xlabel('X Position In Image Plane (AU)', weight='bold')
    plt.ylabel('Y Position In Image Plane (AU)', weight='bold')
    plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)

def plotRerotatedFromNus(ind, sma, e, W, w, inc, Op, yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds,\
    nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    twoIntSameY_x_dr, twoIntSameY_y_dr, num):
    """
    """
    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    ca = plt.gca()
    ca.axis('equal')
    plt.scatter([0],[0],color='orange')
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    rs = xyz_3Dellipse(sma,e,W,w,inc,vs)
    plt.plot(rs[0,0],rs[1,0],color='black')

    ## Plot Intersection circle
    plt.plot(1*np.cos(vs),1*np.sin(vs),color='green')

    ## Plot Intersections
    if ind in yrealAllRealInds[fourIntInds]:
        yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,0])
        plt.scatter(r_int0[0],r_int0[1],color='green',marker='o')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='green',marker='o')
        r_int2 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,2])
        plt.scatter(r_int2[0],r_int2[1],color='green',marker='o')
        r_int3 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,3])
        plt.scatter(r_int3[0],r_int3[1],color='green',marker='o')


        # r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntSameY[yind,0]+np.pi)
        # plt.scatter(r_int0[0],r_int0[1],color='magenta',marker='o')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,1]+np.pi)
        plt.scatter(r_int1[0],r_int1[1],color='magenta',marker='o')

        # r_int0 = xyz_3Dellipse(sma,e,W,w,inc,np.pi - nu_twoIntSameY[yind,0])
        # plt.scatter(r_int0[0],r_int0[1],color='green',marker='x')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,np.pi - nu_fourInt[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='green',marker='x')

        # r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntSameY[yind,0]+np.pi/6)
        # plt.scatter(r_int0[0],r_int0[1],color='red',marker='x')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_fourInt[yind,1]+np.pi/6)
        plt.scatter(r_int1[0],r_int1[1],color='red',marker='x')

        # r_int0 = xyz_3Dellipse(sma,e,W,w,inc,-nu_twoIntSameY[yind,0])
        # plt.scatter(r_int0[0],r_int0[1],color='blue',marker='x')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,-nu_fourInt[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='blue',marker='x')


    if ind in yrealAllRealInds[twoIntSameYInds]:
        yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntSameY[yind,0])
        plt.scatter(r_int0[0],r_int0[1],color='green',marker='o')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntSameY[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='green',marker='o')



        #plt.scatter(twoIntSameY_x[yind], twoIntSameY_y[yind],color='blue',marker='o')

    if ind in yrealAllRealInds[twoIntOppositeXInds]:
        yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntOppositeX[yind,0])
        plt.scatter(r_int0[0],r_int0[1],color='green',marker='o')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntOppositeX[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='green',marker='o')

        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntOppositeX[yind,0]+np.pi)
        plt.scatter(r_int0[0],r_int0[1],color='red',marker='D')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_twoIntOppositeX[yind,1]+np.pi)
        plt.scatter(r_int1[0],r_int1[1],color='blue',marker='D')

        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,-nu_twoIntOppositeX[yind,0])
        plt.scatter(r_int0[0],r_int0[1],color='red',marker='x')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,-nu_twoIntOppositeX[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='blue',marker='x')

        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,-nu_twoIntOppositeX[yind,0]+np.pi)
        plt.scatter(r_int0[0],r_int0[1],color='red',marker='^')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,-nu_twoIntOppositeX[yind,1]+np.pi)
        plt.scatter(r_int1[0],r_int1[1],color='blue',marker='^')

    if ind in only2RealInds:
        yind = np.where(only2RealInds == ind)[0]
        r_int0 = xyz_3Dellipse(sma,e,W,w,inc,nu_IntersectionsOnly2[yind,0])
        plt.scatter(r_int0[0],r_int0[1],color='green',marker='o')
        r_int1 = xyz_3Dellipse(sma,e,W,w,inc,nu_IntersectionsOnly2[yind,1])
        plt.scatter(r_int1[0],r_int1[1],color='green',marker='o')

    ## Plot Smin Smax Diamonds
    r_min = xyz_3Dellipse(sma,e,W,w,inc,nu_minSepPoints[ind])
    plt.scatter(r_min[0],r_min[1],color='teal',marker='D',s=64)
    r_max = xyz_3Dellipse(sma,e,W,w,inc,nu_maxSepPoints[ind])
    plt.scatter(r_max[0],r_max[1],color='red',marker='D',s=64)

    ## Plot Slmin Slmax Diamonds
    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        r_lmin = xyz_3Dellipse(sma,e,W,w,inc,nu_lminSepPoints[tind])
        plt.scatter(r_lmin[0],r_lmin[1],color='magenta',marker='D',s=64)
        r_lmax = xyz_3Dellipse(sma,e,W,w,inc,nu_lmaxSepPoints[tind])
        plt.scatter(r_lmax[0],r_lmax[1],color='gold',marker='D',s=64)

    plt.title('sma: ' + str(np.round(sma,4)) + ' e: ' + str(np.round(e,4)) + ' W: ' + str(np.round(W,4)) + '\nw: ' + str(np.round(w,4)) + ' inc: ' + str(np.round(inc,4)))
    plt.show(block=False)

def errorLinePlot(fourIntInds,errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,\
    twoIntSameYInds,errors_twoIntSameY0,errors_twoIntSameY1,twoIntOppositeXInds,errors_twoIntOppositeX0,errors_twoIntOppositeX1,\
    only2RealInds,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,num):
    plt.close(num)
    plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    plt.yscale('log')
    plt.xscale('log')
    # yrealAllRealInds[fourIntInds]
    plt.plot(np.arange(len(fourIntInds)),np.abs(np.sort(-errors_fourInt0)[np.arange(len(fourIntInds))]),label='Four Int 0')
    plt.plot(np.arange(len(fourIntInds)),np.abs(np.sort(-errors_fourInt1)[np.arange(len(fourIntInds))]),label='Four Int 1')
    plt.plot(np.arange(len(fourIntInds)),np.abs(np.sort(-errors_fourInt2)[np.arange(len(fourIntInds))]),label='Four Int 2')
    plt.plot(np.arange(len(fourIntInds)),np.abs(np.sort(-errors_fourInt3)[np.arange(len(fourIntInds))]),label='Four Int 3')
    # yrealAllRealInds[twoIntSameYInds]
    plt.plot(np.arange(len(twoIntSameYInds)),np.abs(np.sort(-errors_twoIntSameY0)),label='Two Int Same Y 0')
    plt.plot(np.arange(len(twoIntSameYInds)),np.abs(np.sort(-errors_twoIntSameY1)),label='Two Int Same Y 1')
    # yrealAllRealInds[twoIntOppositeXInds]
    plt.plot(np.arange(len(twoIntOppositeXInds)),np.abs(np.sort(-errors_twoIntOppositeX0)),label='Two Int Opposite X 0')
    plt.plot(np.arange(len(twoIntOppositeXInds)),np.abs(np.sort(-errors_twoIntOppositeX1)),label='Two Int Opposite X 1')
    # only2RealInds
    plt.plot(np.arange(len(only2RealInds)),np.abs(np.sort(-errors_IntersectionsOnly2X0)),label='Only 2 Int 0')
    plt.plot(np.arange(len(only2RealInds)),np.abs(np.sort(-errors_IntersectionsOnly2X1)),label='Only 2 Int 1')

    plt.legend()
    plt.ylabel('Absolute Separation Error (AU)', weight='bold')
    plt.xlabel('Planet Orbit Index', weight='bold')
    plt.show(block=False)

def plotErrorHistogramAlpha(errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,errors_twoIntSameY1,\
    errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,num):
    plt.close(num)
    plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    plt.yscale('log')
    plt.xscale('log')

    plt.hist(np.abs(errors_fourInt0)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Four Int 0',alpha=0.2)
    plt.hist(np.abs(errors_fourInt1)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Four Int 1',alpha=0.2)
    plt.hist(np.abs(errors_fourInt2)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Four Int 2',alpha=0.2)
    plt.hist(np.abs(errors_fourInt3)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Four Int 1',alpha=0.2)

    plt.hist(np.abs(errors_twoIntSameY0)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Two Int Same Y 0',alpha=0.2)
    plt.hist(np.abs(errors_twoIntSameY1)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Two Int Same Y 1',alpha=0.2)

    plt.hist(np.abs(errors_twoIntOppositeX0)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Two Int Opposite X 0',alpha=0.2)
    plt.hist(np.abs(errors_twoIntOppositeX1)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Two Int Opposite X 1',alpha=0.2)

    plt.hist(np.abs(errors_IntersectionsOnly2X0)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Only 2 Int 0',alpha=0.2)
    plt.hist(np.abs(errors_IntersectionsOnly2X1)+1e-17, bins=np.logspace(start=-17.,stop=-1,num=17),label='Only 2 Int 1',alpha=0.2)
    plt.xlabel('Absolute Error (AU)', weight='bold')
    plt.ylabel('Number of Planets', weight='bold') #Switch to fraction
    plt.legend()
    plt.show(block=False)

def plotErrorHistogram(errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,\
    errors_twoIntSameY0,errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,\
    errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,num):
    plt.close(num)
    plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    plt.yscale('log')
    plt.xscale('log')

    MIN = np.min(np.concatenate((np.abs(errors_fourInt0)+1e-17, np.abs(errors_fourInt1)+1e-17,np.abs(errors_fourInt2)+1e-17,np.abs(errors_fourInt3)+1e-17,\
            np.abs(errors_twoIntSameY0)+1e-17, np.abs(errors_twoIntSameY1)+1e-17,np.abs(errors_twoIntOppositeX0)+1e-17,\
            np.abs(errors_twoIntOppositeX1)+1e-17,np.abs(errors_IntersectionsOnly2X0)+1e-17,np.abs(errors_IntersectionsOnly2X1)+1e-17)))
    MAX = np.max(np.concatenate((np.abs(errors_fourInt0)+1e-17, np.abs(errors_fourInt1)+1e-17,np.abs(errors_fourInt2)+1e-17,np.abs(errors_fourInt3)+1e-17,\
            np.abs(errors_twoIntSameY0)+1e-17, np.abs(errors_twoIntSameY1)+1e-17,np.abs(errors_twoIntOppositeX0)+1e-17,\
            np.abs(errors_twoIntOppositeX1)+1e-17,np.abs(errors_IntersectionsOnly2X0)+1e-17,np.abs(errors_IntersectionsOnly2X1)+1e-17)))
    numBins = int(np.ceil(np.log10(MAX))) - int(np.floor(np.log10(MIN))) - 1
    bins = 10 ** np.linspace(np.floor(np.log10(MIN)), np.ceil(np.log10(MAX)), numBins)
    plt.hist(np.concatenate((np.abs(errors_fourInt0)+1e-17, np.abs(errors_fourInt1)+1e-17,np.abs(errors_fourInt2)+1e-17,np.abs(errors_fourInt3)+1e-17,\
            np.abs(errors_twoIntSameY0)+1e-17, np.abs(errors_twoIntSameY1)+1e-17,np.abs(errors_twoIntOppositeX0)+1e-17,\
            np.abs(errors_twoIntOppositeX1)+1e-17,np.abs(errors_IntersectionsOnly2X0)+1e-17,np.abs(errors_IntersectionsOnly2X1)+1e-17)), color='purple', bins=bins)# bins=np.logspace(start=-17.,stop=-1,num=17))
    plt.xlabel('Planet-Star Separation Error in (AU)', weight='bold')
    plt.ylabel('Number of Planets', weight='bold') #Switch to fraction
    plt.show(block=False)

def plotProjectedEllipseWithNu(ind,sma,e,W,w,inc,nu_minSepPoints,nu_maxSepPoints, yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds,\
    only2RealInds, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2, num):
    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')

    ## Central Sun
    plt.scatter([0],[0],color='orange')
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    r = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],vs)
    x_3Dellipse = r[0,0,:]
    y_3Dellipse = r[1,0,:]
    plt.plot(x_3Dellipse,y_3Dellipse,color='black')

    #Plot Separation Limits
    r_minSep = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_minSepPoints[ind])
    tmp_minSep = np.sqrt(r_minSep[0]**2 + r_minSep[1]**2)
    r_maxSep = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_maxSepPoints[ind])
    tmp_maxSep = np.sqrt(r_maxSep[0]**2 + r_maxSep[1]**2)
    plt.scatter(r_minSep[0],r_minSep[1],color='teal',marker='D')
    plt.scatter(r_maxSep[0],r_maxSep[1],color='red',marker='D')
    if ind in yrealAllRealInds:
        print('All Real')
        tind = np.where(yrealAllRealInds == ind)[0]
        r_lminSep = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_lminSepPoints[tind])#[yrealAllRealInds[tind]])
        tmp_lminSep = np.sqrt(r_lminSep[0]**2 + r_lminSep[1]**2)
        r_lmaxSep = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_lmaxSepPoints[tind])#[yrealAllRealInds[tind]]+np.pi)
        tmp_lmaxSep = np.sqrt(r_lmaxSep[0]**2 + r_lmaxSep[1]**2)
        plt.scatter(r_lminSep[0],r_lminSep[1],color='magenta',marker='D')
        plt.scatter(r_lmaxSep[0],r_lmaxSep[1],color='gold',marker='D')

    if ind in yrealAllRealInds[fourIntInds]:
        #WORKING
        print('All Real 4 Int')
        yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
        r_fourInt0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,0])
        r_fourInt1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,1])
        r_fourInt2 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,2])
        r_fourInt3 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,3])
        plt.scatter(r_fourInt0[0],r_fourInt0[1],color='green',marker='o')
        plt.scatter(r_fourInt1[0],r_fourInt1[1],color='green',marker='o')
        plt.scatter(r_fourInt2[0],r_fourInt2[1],color='green',marker='o')
        plt.scatter(r_fourInt3[0],r_fourInt3[1],color='green',marker='o')
    elif ind in yrealAllRealInds[twoIntSameYInds]: #Same Y
        print('All Real 2 Int Same Y')
        yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
        r_twoIntSameY0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,0])
        r_twoIntSameY1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,1])
        plt.scatter(r_twoIntSameY0[0],r_twoIntSameY0[1],color='green',marker='o')
        plt.scatter(r_twoIntSameY1[0],r_twoIntSameY1[1],color='green',marker='o')
    elif ind in yrealAllRealInds[twoIntOppositeXInds]: #Same X
        print('All Real 2 Int Opposite X')
        yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
        r_twoIntOppositeX0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,0])
        r_twoIntOppositeX1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,1])
        plt.scatter(r_twoIntOppositeX0[0],r_twoIntOppositeX0[1],color='green',marker='o')
        plt.scatter(r_twoIntOppositeX1[0],r_twoIntOppositeX1[1],color='green',marker='o')
    elif ind in only2RealInds:
        print('All Real 2 Int')
        yind = np.where(only2RealInds == ind)[0]
        r_IntersectionOnly20 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,0])
        r_IntersectionOnly21 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,1])
        plt.scatter(r_IntersectionOnly20[0],r_IntersectionOnly20[1],color='green',marker='o')
        plt.scatter(r_IntersectionOnly21[0],r_IntersectionOnly21[1],color='green',marker='o')

        r_IntersectionOnly20 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,0])
        plt.scatter(r_IntersectionOnly20[0],r_IntersectionOnly20[1],color='grey',marker='x')

        r_IntersectionOnly21 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,1])
        plt.scatter(r_IntersectionOnly21[0],r_IntersectionOnly21[1],color='blue',marker='x')

    #Plot lmaxSep Circle
    x_circ2 = 1.*np.cos(vs)
    y_circ2 = 1.*np.sin(vs)
    plt.plot(x_circ2,y_circ2,color='green')
    ca = plt.gca()
    ca.axis('equal')
    plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)

def plotSeparationvsnu(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep, \
    nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints,\
    nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, num):
    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    nurange = np.linspace(start=0.,stop=2.*np.pi,num=100)
    prs = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nurange)
    pseps = np.sqrt(prs[0,0]**2+prs[1,0]**2)
    plt.plot(nurange,pseps,color='black')
    plt.plot([0,2.*np.pi],[0,0],color='black',linestyle='--') #0 sep line
    plt.plot([0,2*np.pi],[minSep[ind],minSep[ind]],color='teal')
    plt.plot([0,2*np.pi],[maxSep[ind],maxSep[ind]],color='red')
    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        #plt.plot([0,2*np.pi],[lminSep[tind],lminSep[tind]],color='magenta')
        #plt.plot([0,2*np.pi],[lmaxSep[tind],lmaxSep[tind]],color='gold')
    plt.plot([0,2*np.pi],[1,1],color='green') #the plot intersection line

    #Plot Separation Limits
    plt.scatter(nu_minSepPoints[ind],minSep[ind],color='teal',marker='D',zorder=25)
    plt.scatter(nu_maxSepPoints[ind],maxSep[ind],color='red',marker='D',zorder=25)
    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        plt.scatter(nu_lminSepPoints[tind],lminSep[tind],color='magenta',marker='D',zorder=25)
        plt.scatter(nu_lmaxSepPoints[tind],lmaxSep[tind],color='gold',marker='D',zorder=25)

    if ind in yrealAllRealInds[fourIntInds]:
        yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
        r_fourInt0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,0])
        r_fourInt1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,1])
        r_fourInt2 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,2])
        r_fourInt3 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,3])
        plt.scatter(nu_fourInt[yind,0],np.sqrt(r_fourInt0[0]**2 + r_fourInt0[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(nu_fourInt[yind,1],np.sqrt(r_fourInt1[0]**2 + r_fourInt1[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(nu_fourInt[yind,2],np.sqrt(r_fourInt2[0]**2 + r_fourInt2[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(nu_fourInt[yind,3],np.sqrt(r_fourInt3[0]**2 + r_fourInt3[1]**2),color='green',marker='o',zorder=25)
    elif ind in yrealAllRealInds[twoIntSameYInds]: #Same Y
        yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
        r_twoIntSameY0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,0])
        r_twoIntSameY1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,1])
        plt.scatter(nu_twoIntSameY[yind,0],np.sqrt(r_twoIntSameY0[0]**2 + r_twoIntSameY0[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(nu_twoIntSameY[yind,1],np.sqrt(r_twoIntSameY1[0]**2 + r_twoIntSameY1[1]**2),color='green',marker='o',zorder=25)
    elif ind in yrealAllRealInds[twoIntOppositeXInds]: #Same X
        yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
        r_twoIntOppositeX0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,0])
        r_twoIntOppositeX1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,1])
        plt.scatter(nu_twoIntOppositeX[yind,0],np.sqrt(r_twoIntOppositeX0[0]**2 + r_twoIntOppositeX0[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(nu_twoIntOppositeX[yind,1],np.sqrt(r_twoIntOppositeX1[0]**2 + r_twoIntOppositeX1[1]**2),color='green',marker='o',zorder=25)
    elif ind in only2RealInds:
        yind = np.where(only2RealInds == ind)[0]
        r_IntersectionOnly20 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,0])
        r_IntersectionOnly21 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,1])
        plt.scatter(nu_IntersectionsOnly2[yind,0],np.sqrt(r_IntersectionOnly20[0]**2 + r_IntersectionOnly20[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(nu_IntersectionsOnly2[yind,1],np.sqrt(r_IntersectionOnly21[0]**2 + r_IntersectionOnly21[1]**2),color='green',marker='o',zorder=25)

    plt.xlim([0.,2.*np.pi])
    plt.ylim([0.,1.05*maxSep[ind]])
    plt.ylabel('Projected Separation, s, in AU',weight='bold')
    plt.xlabel('True Anomaly, ' + r'$\nu$' + ', (rad)',weight='bold')
    plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)

def plotSeparationVsTime(ind, sma, e, W, w, inc, minSep, maxSep, lminSep, lmaxSep,\
    t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,\
    t_twoIntSameY0,t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
    nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2,\
    yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, periods, num):

    plt.close(num)
    fig = plt.figure(num=num)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    # Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
    # Mrange = Erange - e[ind]*np.sin(Erange)
    #periods = (2*np.pi*np.sqrt((sma*u.AU)**3/(const.G.to('AU3 / (kg s2)')*const.M_sun))).to('year').value
    # xellipsetmp = a[ind]*np.cos(Erange)
    # yellipsetmp = b[ind]*np.sin(Erange)
    # septmp = np.sqrt((xellipsetmp - x[ind])**2 + (yellipsetmp - y[ind])**2)
    #plt.plot(Erange,septmp,color='black')
    nurange = np.linspace(start=0.,stop=2.*np.pi,num=400)
    trange = timeFromTrueAnomaly(nurange,periods[ind],e[ind])
    rs = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nurange)
    seps = np.sqrt(rs[0,0]**2+rs[1,0]**2)

    plt.plot(trange,seps,color='black')
    plt.plot([0,periods[ind]],[0,0],color='black',linestyle='--') #0 sep line
    plt.plot([0,periods[ind]],[minSep[ind],minSep[ind]],color='teal')
    plt.plot([0,periods[ind]],[maxSep[ind],maxSep[ind]],color='red')
    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        #plt.plot([0,periods[ind]],[lminSep[tind],lminSep[tind]],color='magenta')
        #plt.plot([0,periods[ind]],[lmaxSep[tind],lmaxSep[tind]],color='gold')
    plt.plot([0,periods[ind]],[1,1],color='green') #the plot intersection line

    #Plot Separation Limits
    #DELETEt_minSep = timeFromTrueAnomaly(nu_minSepPoints[ind],periods[ind],e[ind])
    #DELETEt_maxSep = timeFromTrueAnomaly(nu_maxSepPoints[ind],periods[ind],e[ind])
    plt.scatter(t_minSep[ind],minSep[ind],color='teal',marker='D',zorder=25)
    plt.scatter(t_maxSep[ind],maxSep[ind],color='red',marker='D',zorder=25)
    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        #DELETEt_lminSep = timeFromTrueAnomaly(nu_lminSepPoints[tind],periods[ind],e[ind])
        #DELETEt_lmaxSep = timeFromTrueAnomaly(nu_lmaxSepPoints[tind],periods[ind],e[ind])
        plt.scatter(t_lminSep[tind],lminSep[tind],color='magenta',marker='D',zorder=25)
        plt.scatter(t_lmaxSep[tind],lmaxSep[tind],color='gold',marker='D',zorder=25)

    if ind in yrealAllRealInds[fourIntInds]:
        yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
        #DELETEt_fourInt0 = timeFromTrueAnomaly(nu_fourInt[yind,0],periods[ind],e[ind])
        #DELETEt_fourInt1 = timeFromTrueAnomaly(nu_fourInt[yind,1],periods[ind],e[ind])
        #DELETEt_fourInt2 = timeFromTrueAnomaly(nu_fourInt[yind,2],periods[ind],e[ind])
        #DELETEt_fourInt3 = timeFromTrueAnomaly(nu_fourInt[yind,3],periods[ind],e[ind])
        r_fourInt0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,0])
        r_fourInt1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,1])
        r_fourInt2 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,2])
        r_fourInt3 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_fourInt[yind,3])
        plt.scatter(t_fourInt0[yind],np.sqrt(r_fourInt0[0]**2 + r_fourInt0[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(t_fourInt1[yind],np.sqrt(r_fourInt1[0]**2 + r_fourInt1[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(t_fourInt2[yind],np.sqrt(r_fourInt2[0]**2 + r_fourInt2[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(t_fourInt3[yind],np.sqrt(r_fourInt3[0]**2 + r_fourInt3[1]**2),color='green',marker='o',zorder=25)
    elif ind in yrealAllRealInds[twoIntSameYInds]: #Same Y
        yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
        #DELETEt_twoIntSameY0 = timeFromTrueAnomaly(nu_twoIntSameY[yind,0],periods[ind],e[ind])
        #DELETEt_twoIntSameY1 = timeFromTrueAnomaly(nu_twoIntSameY[yind,1],periods[ind],e[ind])
        r_twoIntSameY0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,0])
        r_twoIntSameY1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntSameY[yind,1])
        plt.scatter(t_twoIntSameY0[yind],np.sqrt(r_twoIntSameY0[0]**2 + r_twoIntSameY0[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(t_twoIntSameY1[yind],np.sqrt(r_twoIntSameY1[0]**2 + r_twoIntSameY1[1]**2),color='green',marker='o',zorder=25)
    elif ind in yrealAllRealInds[twoIntOppositeXInds]: #Same X
        yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
        #DELETEt_twoIntOppositeX0 = timeFromTrueAnomaly(nu_twoIntOppositeX[yind,0],periods[ind],e[ind])
        #DELETEt_twoIntOppositeX1 = timeFromTrueAnomaly(nu_twoIntOppositeX[yind,1],periods[ind],e[ind])
        r_twoIntOppositeX0 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,0])
        r_twoIntOppositeX1 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_twoIntOppositeX[yind,1])
        plt.scatter(t_twoIntOppositeX0[yind],np.sqrt(r_twoIntOppositeX0[0]**2 + r_twoIntOppositeX0[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(t_twoIntOppositeX1[yind],np.sqrt(r_twoIntOppositeX1[0]**2 + r_twoIntOppositeX1[1]**2),color='green',marker='o',zorder=25)
    elif ind in only2RealInds:
        yind = np.where(only2RealInds == ind)[0]
        #DELETEt_IntersectionOnly20 = timeFromTrueAnomaly(nu_IntersectionsOnly2[yind,0],periods[ind],e[ind])
        #DELETEt_IntersectionOnly21 = timeFromTrueAnomaly(nu_IntersectionsOnly2[yind,1],periods[ind],e[ind])
        r_IntersectionOnly20 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,0])
        r_IntersectionOnly21 = xyz_3Dellipse(sma[ind],e[ind],W[ind],w[ind],inc[ind],nu_IntersectionsOnly2[yind,1])
        plt.scatter(t_IntersectionOnly20[yind],np.sqrt(r_IntersectionOnly20[0]**2 + r_IntersectionOnly20[1]**2),color='green',marker='o',zorder=25)
        plt.scatter(t_IntersectionOnly21[yind],np.sqrt(r_IntersectionOnly21[0]**2 + r_IntersectionOnly21[1]**2),color='green',marker='o',zorder=25)

    plt.xlim([0.,periods[ind]])
    plt.ylim([0.,1.05*maxSep[ind]])
    plt.ylabel('Projected Separation, s, in AU',weight='bold')
    plt.xlabel('Time Past Periastron, t, (years)',weight='bold')
    plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)


def plotDerotatedEllipseStarLocDividers(ind, sma, e, W, w, inc, x, y, dmajorp, dminorp, only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3, minSepPoints_x,\
    minSepPoints_y, yrealAllRealInds, lminSepPoints_x, lminSepPoints_y, fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
    lmaxSepPoints_x, lmaxSepPoints_y, twoIntSameYInds,\
    maxSepPoints_x, maxSepPoints_y, twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    type0_0Inds, type0_1Inds, type0_2Inds, type0_3Inds, type0_4Inds, type1_0Inds, type1_1Inds, type1_2Inds, type1_3Inds, type1_4Inds,\
    type2_0Inds, type2_1Inds, type2_2Inds, type2_3Inds, type2_4Inds, type3_0Inds, type3_1Inds, type3_2Inds, type3_3Inds, type3_4Inds, num):
    plt.close(num)
    fig = plt.figure(num=num)
    ca = plt.gca()
    ca.axis('equal')
    #DELETEplt.scatter([xreal[ind,0],xreal[ind,1],xreal[ind,2],xreal[ind,3]], [yreal[ind,0],yreal[ind,1],yreal[ind,2],yreal[ind,3]], color='purple')
    plt.scatter([0],[0],color='orange')
    ## 3D Ellipse
    vs = np.linspace(start=0,stop=2*np.pi,num=300)
    #new plot stuff
    Erange = np.linspace(start=0.,stop=2*np.pi,num=400)
    plt.plot([-dmajorp[ind],dmajorp[ind]],[0,0],color='purple',linestyle='--') #major
    plt.plot([0,0],[-dminorp[ind],dminorp[ind]],color='purple',linestyle='--') #minor
    xellipsetmp = dmajorp[ind]*np.cos(Erange)
    yellipsetmp = dminorp[ind]*np.sin(Erange)
    plt.plot(xellipsetmp,yellipsetmp,color='black')
    plt.scatter(x[ind],y[ind],color='orange',marker='x')
    if ind in only2RealInds[typeInds0]:
        plt.scatter(x[ind],y[ind],edgecolors='teal',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds1]:
        plt.scatter(x[ind],y[ind],edgecolors='red',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds2]:
        plt.scatter(x[ind],y[ind],edgecolors='blue',marker='o',s=64,facecolors='none')
    if ind in only2RealInds[typeInds3]:
        plt.scatter(x[ind],y[ind],edgecolors='magenta',marker='o',s=64,facecolors='none')

    c_ae = dmajorp[ind]*np.sqrt(1-dminorp[ind]**2/dmajorp[ind]**2)
    plt.scatter([-c_ae,c_ae],[0,0],color='blue')

    # #Plot Min Sep Circle
    # x_circ = minSep[ind]*np.cos(vs)
    # y_circ = minSep[ind]*np.sin(vs)
    # plt.plot(x[ind]+x_circ,y[ind]+y_circ,color='teal')
    # #Plot Max Sep Circle
    # x_circ2 = maxSep[ind]*np.cos(vs)
    # y_circ2 = maxSep[ind]*np.sin(vs)
    # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='red')
    #Plot Min Sep Ellipse Intersection
    plt.scatter(minSepPoints_x[ind],minSepPoints_y[ind],color='teal',marker='D')
    #Plot Max Sep Ellipse Intersection
    plt.scatter(maxSepPoints_x[ind],maxSepPoints_y[ind],color='red',marker='D')

    if ind in yrealAllRealInds:
        tind = np.where(yrealAllRealInds == ind)[0]
        # #Plot lminSep Circle
        # x_circ2 = lminSep[tind]*np.cos(vs)
        # y_circ2 = lminSep[tind]*np.sin(vs)
        # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='magenta')
        # #Plot lmaxSep Circle
        # x_circ2 = lmaxSep[tind]*np.cos(vs)
        # y_circ2 = lmaxSep[tind]*np.sin(vs)
        # plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='gold')
        #### Plot Local Min
        plt.scatter(lminSepPoints_x[tind], lminSepPoints_y[tind],color='magenta',marker='D')
        #### Plot Local Max Points
        plt.scatter(lmaxSepPoints_x[tind], lmaxSepPoints_y[tind],color='gold',marker='D')

    #### r Intersection test
    x_circ2 = np.cos(vs)
    y_circ2 = np.sin(vs)
    plt.plot(x[ind]+x_circ2,y[ind]+y_circ2,color='green')
    #### Intersection Points
    if ind in yrealAllRealInds[fourIntInds]:
        yind = np.where(yrealAllRealInds[fourIntInds] == ind)[0]
        plt.scatter(fourInt_x[yind],fourInt_y[yind], color='green',marker='o')
    elif ind in yrealAllRealInds[twoIntSameYInds]: #Same Y
        yind = np.where(yrealAllRealInds[twoIntSameYInds] == ind)[0]
        plt.scatter(twoIntSameY_x[yind],twoIntSameY_y[yind], color='green',marker='o')
    elif ind in yrealAllRealInds[twoIntOppositeXInds]: #Same X
        yind = np.where(yrealAllRealInds[twoIntOppositeXInds] == ind)[0]
        plt.scatter(twoIntOppositeX_x[yind],twoIntOppositeX_y[yind], color='green',marker='o')
        #### Type0
    elif ind in only2RealInds[type0_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
        print('plotted')
    elif ind in only2RealInds[type0_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type0_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type0_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type0_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
        #### Type1
    elif ind in only2RealInds[type1_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type1_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type1_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type1_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type1_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
        #### Type2
    elif ind in only2RealInds[type2_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type2_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type2_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type2_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type2_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
        #### Type3
    elif ind in only2RealInds[type3_0Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type3_1Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type3_2Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type3_3Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')
    elif ind in only2RealInds[type3_4Inds]:
        gind = np.where(only2RealInds == ind)[0]
        plt.scatter(xIntersectionsOnly2[gind],yIntersectionsOnly2[gind], color='green',marker='o')

    # Plot Star Location Type Dividers
    xran = np.linspace(start=(dmajorp[ind]*(dmajorp[ind]**2*(dmajorp[ind] - dminorp[ind])*(dmajorp[ind] + dminorp[ind]) - dminorp[ind]**2*np.sqrt(3*dmajorp[ind]**4 + 2*dmajorp[ind]**2*dminorp[ind]**2 + 3*dminorp[ind]**4))/(2*(dmajorp[ind]**4 + dminorp[ind]**4))),\
        stop=(dmajorp[ind]*(dmajorp[ind]**2*(dmajorp[ind] - dminorp[ind])*(dmajorp[ind] + dminorp[ind]) + dminorp[ind]**2*np.sqrt(3*dmajorp[ind]**4 + 2*dmajorp[ind]**2*dminorp[ind]**2 + 3*dminorp[ind]**4))/(2*(dmajorp[ind]**4 + dminorp[ind]**4))), num=3, endpoint=True)
    ylineQ1 = xran*dmajorp[ind]/dminorp[ind] - dmajorp[ind]**2/(2*dminorp[ind]) + dminorp[ind]/2 #between first quadrant a,b
    ylineQ4 = -xran*dmajorp[ind]/dminorp[ind] + dmajorp[ind]**2/(2*dminorp[ind]) - dminorp[ind]/2 #between 4th quadrant a,b
    plt.plot(xran, ylineQ1, color='brown', linestyle='-.', )
    plt.plot(-xran, ylineQ4, color='grey', linestyle='-.')
    plt.plot(-xran, ylineQ1, color='orange', linestyle='-.')
    plt.plot(xran, ylineQ4, color='red', linestyle='-.')
    plt.xlim([-1.2*dmajorp[ind],1.2*dmajorp[ind]])
    plt.ylim([-1.2*dminorp[ind],1.2*dminorp[ind]])
    plt.title('sma: ' + str(np.round(sma[ind],4)) + ' e: ' + str(np.round(e[ind],4)) + ' W: ' + str(np.round(W[ind],4)) + '\nw: ' + str(np.round(w[ind],4)) + ' inc: ' + str(np.round(inc[ind],4)))
    plt.show(block=False)


def plotSepsHistogram(minSep,maxSep,lminSep,lmaxSep,sma,yrealAllRealInds,num):
    
    plt.close(num)
    fig2 = plt.figure(constrained_layout=True, num=num, figsize=(6,10))
    gs = fig2.add_gridspec(4, 1)
    bins = np.linspace(start=0.,stop=1.5,num=16)
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    plt.rcParams['axes.linewidth']=2
    plt.rc('font',weight='bold')
    ax0 = fig2.add_subplot(gs[0, :])
    ax0.set_yscale('log')
    ax1 = fig2.add_subplot(gs[1, :])
    ax1.set_yscale('log')
    ax2 = fig2.add_subplot(gs[2, :])
    ax2.set_yscale('log')
    ax3 = fig2.add_subplot(gs[3, :])
    ax3.set_yscale('log')

    ax0.hist(minSep[yrealAllRealInds]/sma[yrealAllRealInds],bins=bins)
    ax0.set_xlabel('Minimum Separation Normalized by Semi-major Axis', weight='bold')
    ax0.set_xlim([0,np.max(maxSep/sma)])
    ax0.set_ylim([10**0,10**5])
    ax1.hist(lminSep/sma[yrealAllRealInds],bins=bins)
    ax1.set_xlabel('Local Minimum Separation Normalized by Semi-major Axis', weight='bold')
    ax1.set_xlim([0,np.max(maxSep/sma)])
    ax1.set_ylim([10**0,10**5])
    ax2.hist(lmaxSep/sma[yrealAllRealInds],bins=bins)
    ax2.set_xlabel('Local Maximum Separation Normalized by Semi-major Axis', weight='bold')
    ax2.set_xlim([0,np.max(maxSep/sma)])
    ax2.set_ylim([10**0,10**5])
    ax3.hist(maxSep[yrealAllRealInds]/sma[yrealAllRealInds],bins=bins)
    ax3.set_xlabel('Maximum Separation Normalized by Semi-major Axis', weight='bold')
    ax3.set_xlim([0,np.max(maxSep/sma)])
    ax3.set_ylim([10**0,10**5])
    plt.show(block=False)

  


