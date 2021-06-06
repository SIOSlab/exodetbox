####
import numpy as np
import time
from astropy import constants as const
import astropy.units as u
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.planet_star_separation import planet_star_separation

def projected_apbpPsipsi(a,e,W,w,inc):
    """ Given the KOE of a planet, calculate the semi-paramters of the projected ellipse
    and the angle from two construction lines to the x-axis (construction lines are 
    intermediaries to determine the angle of the projected semi-major axis from the x-axis) 
    Args:
        a (numpy array):
            semi-major axis in AU
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the Ascending Node in Radians
        w (numpy array):
            Argument of periapsis in Radians
        inc (numpy array):
            Inclination in Radians
    Returns:
        dmajorp (numpy array):
            Semi-major axis of the projected ellipse in AU
        dminorp (numpy array):
            Semi-minor axis of the projected ellipse in AU
        theta_OpQ_X (numpy array):
            Angle from the x-axis to line OpQ
        theta_OpQp_X (numpy array):
            Angle from the x-axis to line OpQp
    """
    Gamma = e*(1. - e**2.)
    gamma = (np.sin(W)*np.cos(w) + np.sin(w)*np.cos(W)*np.cos(inc))
    Phi = np.sqrt((e+1.)/(1.-e))
    phi = a**2.*(e**2. - 1.)**2.
    lam2 = (-np.sin(W)*np.sin(w)*np.cos(inc) + np.cos(W)*np.cos(w))
    Omicron = (np.sin(W)*np.cos(w + 2.*np.arctan(Phi)) + np.sin(w + 2.*np.arctan(Phi))*np.cos(W)*np.cos(inc))
    Eeps2 = a*(1. - e**2.)
    Gorgon = (-np.sin(W)*np.sin(w + 2.*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2.*np.arctan(Phi)))
    Gemini = (e*np.cos(2.*np.arctan(Phi)) + 1.)

    #Specific Calc Substitutions to shorten the code
    Ramgy = (np.sqrt(phi*gamma**2. + phi*(np.sin(W)*np.sin(w)*np.cos(inc) - np.cos(W)*np.cos(w))**2. + phi*np.sin(inc)**2.*np.sin(w)**2.))
    Affinity1 = a**2.*Gamma*gamma/Ramgy
    Yolo1 = (np.sin(W)*np.cos(w + np.pi) + np.sin(w + np.pi)*np.cos(W)*np.cos(inc))
    Kolko1 = (-np.sin(W)*np.sin(w + np.pi)*np.cos(inc) + np.cos(W)*np.cos(w + np.pi))
    #Semi-major axis length
    dmajorp = np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1. - e) - (a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.))**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1. - e) - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2 + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.))**2.)/2.\
             + np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1. - e) + (a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.))**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1. - e) + (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.))**2.)/2.
    #Semi-minor axis length
    dminorp = -np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1. - e) - (a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.))**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1. - e) - (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.))**2.)/2.\
             + np.sqrt(np.abs(Affinity1 + Eeps2*Yolo1/(1. - e) + (a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.))**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Kolko1/(1. - e) + (-Affinity1 - Eeps2*Omicron/Gemini)*np.sqrt(np.abs(Affinity1 + Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.)/np.sqrt(np.abs(-Affinity1 - Eeps2*Omicron/Gemini)**2. + np.abs(a**2.*Gamma*(lam2)/Ramgy + Eeps2*Gorgon/Gemini)**2.))**2.)/2.
    #theta_OpQ_X: Angle between x-axis and line OpQ
    theta_OpQ_X = np.arctan2(Affinity1 + Eeps2*Yolo1/(1. - e) - (a**2.*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2.*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2.*np.arctan(Phi)))/(e*np.cos(2.*np.arctan(Phi)) + 1.))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2.*np.arctan(Phi)) + np.sin(w + 2.*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2. + np.abs(a**2.*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2.*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2.*np.arctan(Phi)))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2.)/np.sqrt(np.abs(-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2.*np.arctan(Phi)) + np.sin(w + 2.*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2.\
         + np.abs(a**2.*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2.*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2.*np.arctan(Phi)))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2.), a**2.*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1. - e) - (-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2.*np.arctan(Phi)) + np.sin(w + 2.*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2.*np.arctan(Phi)) + 1.))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2.*np.arctan(Phi)) + np.sin(w + 2.*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2. + np.abs(a**2.*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2.*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2.*np.arctan(Phi)))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2.)/np.sqrt(np.abs(-Affinity1\
          - Eeps2*(np.sin(W)*np.cos(w + 2.*np.arctan(Phi)) + np.sin(w + 2.*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2. + np.abs(a**2.*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2.*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2.*np.arctan(Phi)))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2.))

    #theta_OpQp_X: Angle between x-axis and line OpQp
    theta_OpQp_X = np.arctan2(Affinity1 + Eeps2*Yolo1/(1. - e) + (a**2.*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2.*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2.*np.arctan(Phi)))/(e*np.cos(2.*np.arctan(Phi)) + 1.))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2.*np.arctan(Phi)) + np.sin(w + 2.*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2. + np.abs(a**2.*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2.*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2.*np.arctan(Phi)))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2.)/np.sqrt(np.abs(-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2.*np.arctan(Phi)) + np.sin(w + 2.*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2. + np.abs(a**2.*Gamma*lam2/Ramgy\
         + Eeps2*(-np.sin(W)*np.sin(w + 2.*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2.*np.arctan(Phi)))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2.), a**2.*Gamma*lam2/Ramgy + Eeps2*Kolko1/(1. - e) + (-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2.*np.arctan(Phi)) + np.sin(w + 2.*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2.*np.arctan(Phi)) + 1.))*np.sqrt(np.abs(Affinity1 + Eeps2*(np.sin(W)*np.cos(w + 2.*np.arctan(Phi)) + np.sin(w + 2.*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2. + np.abs(a**2.*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2.*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2.*np.arctan(Phi)))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2.)/np.sqrt(np.abs(-Affinity1 - Eeps2*(np.sin(W)*np.cos(w + 2.*np.arctan(Phi))\
          + np.sin(w + 2.*np.arctan(Phi))*np.cos(W)*np.cos(inc))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2. + np.abs(a**2.*Gamma*lam2/Ramgy + Eeps2*(-np.sin(W)*np.sin(w + 2.*np.arctan(Phi))*np.cos(inc) + np.cos(W)*np.cos(w + 2.*np.arctan(Phi)))/(e*np.cos(2.*np.arctan(Phi)) + 1.))**2.))

    return dmajorp, dminorp, theta_OpQ_X, theta_OpQp_X

def xyz_3Dellipse(a,e,W,w,inc,v):
    """ Returns XYZ cartesian coordinate representation of the Keplerian Orbital Elements
    Args:
        a (numpy array):
            semi-major axis in AU
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the Ascending Node in Radians
        w (numpy array):
            Argument of periapsis in Radians
        inc (numpy array):
            Inclination in Radians
        v (numpy array):
            true anomaly in Radians
    Returns:
        r (numpy array):
            x, y, z by n
    """
    eqnr = a*(1.-e**2.)/(1.+e*np.cos(v))
    eqnX = eqnr*(np.cos(W)*np.cos(w+v) - np.sin(W)*np.sin(w+v)*np.cos(inc))
    eqnY = eqnr*(np.sin(W)*np.cos(w+v) + np.cos(W)*np.sin(w+v)*np.cos(inc))
    eqnZ = eqnr*(np.sin(inc)*np.sin(w+v))
    return np.asarray([[eqnX],[eqnY],[eqnZ]])

def projected_Op(a,e,W,w,inc):
    """ Calculates the location of the geometric center of the projected ellipse
    Args:
        a (numpy array):
            semi-major axis in AU
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the Ascending Node in Radians
        w (numpy array):
            Argument of periapsis in Radians
        inc (numpy array):
            Inclination in Radians
    Returns:
        r_center (numpy array):
            Projected ellipse center in AU. [x_center, y_center]
    """
    r1 = xyz_3Dellipse(a,e,W,w,inc,0.)
    r2 = xyz_3Dellipse(a,e,W,w,inc,np.pi)
    r_center = (r1+r2)/2.
    return np.asarray([r_center[0][0],r_center[1][0]])

def projected_BpAngle(a,e,W,w,inc):
    """
    Args:
        a (numpy array):
            semi-major axis in AU
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the Ascending Node in Radians
        w (numpy array):
            Argument of periapsis in Radians
        inc (numpy array):
            Inclination in Radians
    Returns:
        theta (numpy array):
            Angle of semi-major axis from X-axis in Radians
    """
    eqnr = a*(1-e**2)/(1+e)
    eqnX = eqnr*(np.cos(W)*np.cos(w) - np.sin(W)*np.sin(w)*np.cos(inc))
    eqnY = eqnr*(np.sin(W)*np.cos(w) + np.cos(W)*np.sin(w)*np.cos(inc))
    eqnZ = eqnr*(np.sin(inc)*np.sin(w))
    rhat = np.asarray([[eqnX],[eqnY],[eqnZ]])/np.linalg.norm(np.asarray([[eqnX],[eqnY],[eqnZ]]), ord=2, axis=0, keepdims=True)

    return np.arctan(rhat[1][0],rhat[0][0])

def projected_projectedLinearEccentricity(a,e,W,w,inc):
    """
    Args:
        a (numpy array):
            semi-major axis in AU
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the Ascending Node in Radians
        w (numpy array):
            Argument of periapsis in Radians
        inc (numpy array):
            Inclination in Radians
    Returns:
        c_3D_projected (numpy array):
            linear eccentricity of 3D ellipse's projected distance on plane of 2D ellipse in AU
    """
    O = projected_Op(a,e,W,w,inc)
    c_3D_projected = np.linalg.norm(O, ord=2, axis=0, keepdims=True)
    return c_3D_projected[0]

def derotate_arbitraryPoint(x,y,phi):
    """ Derotates an arbitrary point by angle Phi
    Args:
        x (numpy array):
            distance from focus (star) to Op along X-axis
        y (numpy array):
            distance from focus (star) to Op along Y-axis
        phi (numpy array):
            angle from X-axis to semi-minor axis of projected ellipse 
    Returns:
        x2 (numpy array):
            de-rotated distance from origin to star focus along semi-major axis
        y2 (numpy array):
            de-rotated distance from origin to star focus along semi-minor axis
    """
    ahat_x = np.cos(phi)
    ahat_y = np.sin(phi)
    bhat_x = -np.sin(phi)
    bhat_y = np.cos(phi)

    #portion of FOp in direction of ahat
    x2 = ahat_x*x + ahat_y*y
    y2 = bhat_x*x + bhat_y*y

    return -x2, -y2

def derotatedEllipse(theta_OpQ_X, theta_OpQp_X, Op):
    """ Take the position and angle of the geometric center of the projected ellipse (relative to the orbiting Foci located at 0,0)
    and, assuming the geometric center of the ellipse is now at (0,0) return the location of the orbiting foci relative to (0,0)
    of the derotated ellipse
    Args:
        theta_OpQ_X (numpy array):
            angle of OpQ from x-axis
        theta_OpQp_X (numpy array):
            angle of OpQp from x-axis
        Op (numpy array):
            projected ellipse position [x,y] relative to star (origin)
    Return:
        x (numpy array):
            x coordinates of host star relative to projected ellipse center
        y (numpy array):
            y coordinates of host star relative to projected ellipse center
        Phi (numpy array):
            Angle of projected ellipse semi-major axis from x-axis
    """
    Phi = (theta_OpQ_X+theta_OpQp_X)/2.
    x, y = derotate_arbitraryPoint(Op[0],Op[1],Phi)
    return x, y, Phi

def rerotateEllipsePoints(xpts,ypts,Phi,Ox,Oy):
    """ A function designed to take a point on the derotated ellipse and convert it to the corresponding point on the rotated ellipse
    Args:
        xpts (numpy array):
            the x points in the derotated frame to rerotate
        ypts (numpy array):
            the y points in the derotated frame to rerotate
        Phi (numpy array):
            the angle to rotate these points by
        Ox (numpy array):
            the x component of the geometric origin of the projected ellipse
        Oy (numpy array):
            the y component of the geometric origin of the projected ellipse
    Returns:
        ux (numpy array):
            the rerotated x points
        uy (numpy array):
            the rerotated y points
    """
    ux = np.cos(Phi)*xpts - np.sin(Phi)*ypts + Ox
    uy = np.sin(Phi)*xpts + np.cos(Phi)*ypts + Oy
    return ux, uy



def roots_vec(p):
    p = np.atleast_1d(p)
    n = p.shape[-1]
    A = np.zeros(p.shape[:1] + (n-1, n-1), float)
    A[...,1:,:-1] = np.eye(n-2)
    A[...,0,:] = -p[...,1:]/p[...,None,0]
    return np.linalg.eigvals(A)

def roots_loop(p):
    r = []
    for pp in p:
        r.append(np.roots(pp))
    return r

def quarticCoefficients(a,b,mx,my):
    """ Calculates coefficients to the quartic polynomial from projected ellipse semi parameters and
    projected strictly positive star coordinates
    Args:
        a (numpy array):
            semi-major axis of projected ellipse with length n planets
        b (numpy array):
            semi-minor axis of projected ellipse with length n planets
        mx (numpy array):
            positive x coordinate of projected star position with length n planets
        my (numpy array):
            positive y coordinate of projected star position with length n planets
    Returns:
        parr (numpy array):
            quartic coefficients for each star with shape n planets by 5 coefficients
    """
    A = -(2 - 2*b**2/a**2)**2/a**2
    B = 4*mx*(2 - 2*b**2/a**2)/a**2
    C = -4*my**2*b**2/a**4 - 4*mx**2/a**2 + (2 - 2*b**2/a**2)**2
    D = -4*mx*(2-2*b**2/a**2)
    E = 4*mx**2
    return np.asarray([A, B, C, D, E]).T

def quarticSolutions(a,b,mx,my):
    """ Runs and separates out real and imaginary components of the quartic solutions
    Args:
        a (numpy array):
            semi-major axis of projected ellipse with length n planets
        b (numpy array):
            semi-minor axis of projected ellipse with length n planets
        mx (numpy array):
            positive x coordinate of projected star position with length n planets
        my (numpy array):
            positive y coordinate of projected star position with length n planets
    Returns:
        xreal (numpy array):
            real component of x solutions to the quartic
        imag (numpy array):
            imaginary component of x solutions to the quartic
    """
    out = np.asarray(roots_loop(quarticCoefficients(a,b,mx,my)))
    xreal = np.real(out)
    imag = np.imag(out)
    return xreal, imag

def ellipseYFromX(xreal, a, b):
    """ Calculates y values in the positive quadrant of the ellipse
    Args:
        xreal (numpy array):
            shape n planets by 4
        a (numpy array):
            semi-major axis of projected ellipse with length n planets
        b (numpy array):
            semi-minor axis of projected ellipse with length n planets
    return:
        yreal (numpy array):
            numpy array of ellipse quadrant 1 y values, shape n planets by 4
    """
    return np.asarray([np.sqrt(b**2*(1-xreal[:,0]**2/a**2)), np.sqrt(b**2*(1-xreal[:,1]**2/a**2)), np.sqrt(b**2*(1-xreal[:,2]**2/a**2)), np.sqrt(b**2*(1-xreal[:,3]**2/a**2))]).T

def calculateSeparations(xreal, yreal, mx, my):
    """ Calculate absolute and local minimums and maximums
    Args:
        xreal (numpy array):
            x positions of zero solutions to the quartic solely in quadrant 1 with shape n planets by 4
        xreal (numpy array):
            x positions of zero solutions to the quartic solely in quadrant 1 with shape n planets by 4
        mx (numpy array):
            x position of the star in the derotated, projected ellipse with length n planets
        my (numpy array):
            y position of the star in the derotated, projected ellipse with length n planets
    """ 
    #Local min and Local max must lie in the same quadrant immediately above or below the quadrant the star belongs in
    s_mp = np.asarray([np.sqrt((xreal[:,0]-mx)**2 + (yreal[:,0]+my)**2), np.sqrt((xreal[:,1]-mx)**2 + (yreal[:,1]+my)**2), np.sqrt((xreal[:,2]-mx)**2 + (yreal[:,2]+my)**2), np.sqrt((xreal[:,3]-mx)**2 + (yreal[:,3]+my)**2)]).T
    s_pm = np.asarray([np.sqrt((xreal[:,0]+mx)**2 + (yreal[:,0]-my)**2), np.sqrt((xreal[:,1]+mx)**2 + (yreal[:,1]-my)**2), np.sqrt((xreal[:,2]+mx)**2 + (yreal[:,2]-my)**2), np.sqrt((xreal[:,3]+mx)**2 + (yreal[:,3]-my)**2)]).T

    #Using Abs because some terms are negative???
    s_absmin = np.asarray([np.sqrt((np.abs(xreal[:,0])-mx)**2 + (np.abs(yreal[:,0])-my)**2), np.sqrt((np.abs(xreal[:,1])-mx)**2 + (np.abs(yreal[:,1])-my)**2), np.sqrt((np.abs(xreal[:,2])-mx)**2 + (np.abs(yreal[:,2])-my)**2), np.sqrt((np.abs(xreal[:,3])-mx)**2 + (np.abs(yreal[:,3])-my)**2)]).T
    s_absmax = np.asarray([np.sqrt((np.abs(xreal[:,0])+mx)**2 + (np.abs(yreal[:,0])+my)**2), np.sqrt((np.abs(xreal[:,1])+mx)**2 + (np.abs(yreal[:,1])+my)**2), np.sqrt((np.abs(xreal[:,2])+mx)**2 + (np.abs(yreal[:,2])+my)**2), np.sqrt((np.abs(xreal[:,3])+mx)**2 + (np.abs(yreal[:,3])+my)**2)]).T

    return s_mp, s_pm, s_absmin, s_absmax

def sepsMinMaxLminLmax(s_absmin, s_absmax, s_mp, xreal, yreal, x, y):
    """ Calculates Minimum, Maximum, Local Minimum, Local Maximum for Each star
    Args:
        s_absmin (numpy array):
        s_absmax (numpy array):
        s_mp (numpy array):
        xreal (numpy array):
            x points of solutions on ellipse shape n planets by 4
        yreal (numpy array):
            y points of solutions on ellipse shape n planets by 4
        x (numpy array):
            x positions of projected star in derotated ellipse with length n
        y (numpy array):
            y positions of projected star in derotated ellipse with length n
    Returns:
        minSepPoints_x (numpy array):
        minSepPoints_y (numpy array):
        maxSepPoints_x (numpy array):
        maxSepPoints_y (numpy array):
        lminSepPoints_x (numpy array):
        lminSepPoints_y (numpy array):
        lmaxSepPoints_x (numpy array):
        lmaxSepPoints_y (numpy array):
    """
    #### Minimum Separations and x, y of minimum separation
    minSepInd = np.nanargmin(s_absmin,axis=1)
    minSep = s_absmin[np.arange(len(minSepInd)),minSepInd] #Minimum Planet-StarSeparations
    minSep_x = xreal[np.arange(len(minSepInd)),minSepInd] #Minimum Planet-StarSeparations x coord
    minSep_y = yreal[np.arange(len(minSepInd)),minSepInd] #Minimum Planet-StarSeparations y coord

    #### Maximum Separations and x,y of maximum separation
    maxSepInd = np.nanargmax(s_absmax,axis=1)
    maxSep = s_absmax[np.arange(len(maxSepInd)),maxSepInd] #Maximum Planet-StarSeparations
    maxSep_x = xreal[np.arange(len(minSepInd)),maxSepInd] #Maximum Planet-StarSeparations x coord
    maxSep_y = yreal[np.arange(len(minSepInd)),maxSepInd] #Maximum Planet-StarSeparations y coord

    #Sort arrays Minimum
    sminOrderInds = np.argsort(s_absmin, axis=1) #argsort sorts from min to max with last terms being nan
    minSeps = s_absmin[np.arange(len(minSepInd)),sminOrderInds[:,0]]
    minSeps_x = xreal[np.arange(len(minSepInd)),sminOrderInds[:,0]]
    minSeps_y = yreal[np.arange(len(minSepInd)),sminOrderInds[:,0]]

    #Sort arrays Maximum
    smaxOrderInds = np.argsort(-s_absmax, axis=1) #-argsort sorts from max to min with last terms being nan #Note: Last 2 indicies will be Nan
    maxSeps = s_absmax[np.arange(len(maxSepInd)),smaxOrderInds[:,0]]
    maxSeps_x = np.abs(xreal[np.arange(len(maxSepInd)),smaxOrderInds[:,0]])
    maxSeps_y = yreal[np.arange(len(maxSepInd)),smaxOrderInds[:,0]]

    #Masking
    mask = np.zeros((len(maxSepInd),4), dtype=bool)
    #assert ~np.any(sminOrderInds[:,0] == smaxOrderInds[:,0]), 'Exception: A planet has smin == smax'
    print('Number of smaxInd == sminInd: ' + str(np.count_nonzero(sminOrderInds[:,0] == smaxOrderInds[:,0])))
    mask[np.arange(len(minSepInd)),sminOrderInds[:,0]] = 1 #contains 1's where the minimum separation occurs
    mask[np.arange(len(maxSepInd)),smaxOrderInds[:,0]] = 1 #contains 1's where the maximum separation occurs
    assert np.all(np.isnan(s_absmin) == np.isnan(s_absmax)), 'Exception: absmin and absmax have different nan values'
    mask += np.isnan(s_absmin) #Adds nan solutions to mask

    #### Account For Stars with Local Minimum and Maximum
    #For Stars with 2 Inds Accounted For
    s_mptmp = ~mask*s_mp
    s_mptmp[mask] = np.nan

    #Sort arrays Local Minimum
    s_mpOrderInds = np.argsort(s_mptmp, axis=1) #argsort sorts from min to max with last terms being nan
    s_mplminSeps = s_mptmp[np.arange(len(s_mptmp)),s_mpOrderInds[:,0]]
    lminSeps_x = xreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,0]]
    lminSeps_y = yreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,0]]
    s_mplmaxSeps = s_mptmp[np.arange(len(s_mptmp)),s_mpOrderInds[:,1]]
    lmaxSeps_x = np.abs(xreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,1]])
    lmaxSeps_y = yreal[np.arange(len(s_mptmp)),s_mpOrderInds[:,1]]

    #Quadrant Star Belongs to
    bool1 = x > 0
    bool2 = y > 0
    #Quadrant 1 if T,T
    #Quadrant 2 if F,T
    #Quadrant 3 if F,F
    #Quadrant 4 if T,F

    #### Min Sep Point (Points on plot of Min Sep)
    minSepPoints_x = minSeps_x*(2*bool1-1)
    minSepPoints_y = minSeps_y*(2*bool2-1)

    #### Max Sep Point (Points on plot of max sep)
    maxSepPoints_x = maxSeps_x*(-2*bool1+1)
    maxSepPoints_y = maxSeps_y*(-2*bool2+1)

    #### Local Min Sep Points
    lminSepPoints_x = lminSeps_x*(2*bool1-1)
    lminSepPoints_y = lminSeps_y*(-2*bool2+1)

    #### Local Max Sep Points
    lmaxSepPoints_x = lmaxSeps_x*(2*bool1-1)
    lmaxSepPoints_y = lmaxSeps_y*(-2*bool2+1)

    return minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, s_mplminSeps, s_mplmaxSeps

def checkResiduals(A,B,C,D,xreals2,inds,numSols):
    residual_0 = xreals2[inds,0]**4 + A[inds]*xreals2[inds,0]**3 + B[inds]*xreals2[inds,0]**2 + C[inds]*xreals2[inds,0] + D[inds]
    residual_1 = xreals2[inds,1]**4 + A[inds]*xreals2[inds,1]**3 + B[inds]*xreals2[inds,1]**2 + C[inds]*xreals2[inds,1] + D[inds]
    residual_2 = np.zeros(residual_0.shape)
    residual_3 = np.zeros(residual_0.shape)
    if numSols > 2:
        residual_2 = xreals2[inds,2]**4 + A[inds]*xreals2[inds,2]**3 + B[inds]*xreals2[inds,2]**2 + C[inds]*xreals2[inds,2] + D[inds]
        if numSols > 3:
            residual_3 = xreals2[inds,3]**4 + A[inds]*xreals2[inds,3]**3 + B[inds]*xreals2[inds,3]**2 + C[inds]*xreals2[inds,3] + D[inds]
    residual = np.asarray([residual_0, residual_1, residual_2, residual_3]).T
    isAll = np.all((np.real(residual) < 1e-7)*(np.imag(residual) < 1e-7))
    maxRealResidual = np.max(np.real(residual))
    maxImagResidual = np.max(np.imag(residual))
    return residual, isAll, maxRealResidual, maxImagResidual

def quarticCoefficients_ellipse_to_Quarticipynb(a, b, x, y, r):
    """ Calculates coefficients of the quartic expression solving for the intersection between a circle with radius r and ellipse with semi-major axis a
    semi-minor axis b, and the center of the circle at x and y.
    Coefficients for the quartic of form x**4 + A*x**3 + B*x**2 + C*x + D = 0
    """
    A = -4*a**2*x/(a**2 - b**2)
    B = 2*a**2*(a**2*b**2 - a**2*r**2 + 3*a**2*x**2 + a**2*y**2 - b**4 + b**2*r**2 - b**2*x**2 + b**2*y**2)/(a**4 - 2*a**2*b**2 + b**4)
    C = 4*a**4*x*(-b**2 + r**2 - x**2 - y**2)/(a**4 - 2*a**2*b**2 + b**4)
    D = a**4*(b**4 - 2*b**2*r**2 + 2*b**2*x**2 - 2*b**2*y**2 + r**4 - 2*r**2*x**2 - 2*r**2*y**2 + x**4 + 2*x**2*y**2 + y**4)/(a**4 - 2*a**2*b**2 + b**4)
    return A, B, C, D

def quarticCoefficients_smin_smax_lmin_lmax(a, b, x, y):
    """ Calculates coefficients of the quartic equation solving where ds2/dxe = 0 for the distance between a point and the ellipse
    for an ellipse with semi-major axis a, semi-minor axis b, and point at x, y
    Args:
        a (numpy array):
            semi-major axis of the projected ellipse
        b (numpy array):
            semi-minor axis of the projected ellipse
        x (numpy array):
            x position of the center of the projected ellipse
        y (numpy array):
            y position of the center of the projected ellipse
    Returns:
        A (numpy array):
            coefficients of x^3
        B (numpy array):
            coefficients of x^2
        C (numpy array):
            coefficients of x
        D (numpy array):
            constants
    """
    Gamma = np.zeros(len(a),dtype='complex128')
    Gamma = (4.*a**4. - 8.*a**2.*b**2. + 4.*b**4.)/a**2.
    A = (-8.*a**2.*x + 8.*b**2.*x)/Gamma
    B = (-4.*a**4. + 8.*a**2.*b**2. + 4.*a**2.*x**2. - 4.*b**4. + 4.*b**2.*y**2.)/Gamma
    C = (8.*a**4.*x - 8.*a**2.*b**2.*x)/Gamma
    D = (-4.*a**4.*x**2.)/Gamma
    return A, B, C, D

def quarticSolutions_ellipse_to_Quarticipynb(A, B, C, D):
    """ Equations from ellipse_to_Quartic.ipynb solves the quartic 
    Uses the coefficients of the quartic to find
    Args:
        A (numpy array):
            coefficients of x^3
        B (numpy array):
            coefficients of x^2
        C (numpy array):
            coefficients of x
        D (numpy array):
            constants
    Returns:
        xreal (numpy array):
            an nx4 array contianing the solutions to the quartic expression
        delta (numpy array):
            indicator parameter for quartic solution types
        P (numpy array):
            indicator parameter for quartic solution types
        D2 (numpy array):
            indicator parameter for quartic solution types
        R (numpy array):
            indicator parameter for quartic solution types
        delta_0 (numpy array):
            indicator parameter for quartic solution types
    """
    #A bunch of simplifications
    p0 = (-3.*A**2./8.+B)**3.
    p1 = (A*(A**2./8.-B/2.)+C)**2.
    p2 = -A*(A*(3.*A**2./256.-B/16.)+C/4.)+D
    p3 = -3.*A**2./8.+B
    p4 = 2.*A*(A**2./8.-B/2.)
    p5 = -p0/108.-p1/8.+p2*p3/3.
    p6 = (p0/216.+p1/16.-p2*p3/6.+np.sqrt(p5**2./4.+(-p2-p3**2./12.)**3./27.))**(1./3.)
    p6Inds = np.where(p6==0.0)[0] #p6 being 0 causes divide by 0
    p6[p6Inds] = np.ones(len(p6Inds))*10**-9
    p7 = A**2./4.-2.*B/3.
    p8 = (2.*p2+p3**2./6.)/(3.*p6)
    #, (-2*p2-p3**2/6)/(3*p6)
    p9 = np.sqrt(-2.*p5**(1./3.)+p7)
    p10 = np.sqrt(2.*p6+p7+p8)
    p11 = A**2./2.-4.*B/3.

    #otherwise case
    x0 = -A/4. - p10/2. - np.sqrt(p11 - 2.*p6 - p8 + (2.*C + p4)/p10)/2.
    x1 = -A/4. - p10/2. + np.sqrt(p11 - 2.*p6 - p8 + (2.*C + p4)/p10)/2.
    x2 = -A/4. + p10/2. - np.sqrt(p11 - 2.*p6 - p8 + (-2.*C - p4)/p10)/2.
    x3 = -A/4. + p10/2. + np.sqrt(p11 - 2.*p6 - p8 + (-2.*C - p4)/p10)/2.
    zeroInds = np.where(p2 + p3**2./12. == 0)[0] #piecewise condition
    if len(zeroInds) != 0.:
        x0[zeroInds] = -A[zeroInds]/4. - p9[zeroInds]/2. - np.sqrt(p11[zeroInds] + 2.*np.cbrt(np.real(p5[zeroInds])) + (2.*C[zeroInds] + p4[zeroInds])/p9[zeroInds])/2.
        x1[zeroInds] = -A[zeroInds]/4. - p9[zeroInds]/2. + np.sqrt(p11[zeroInds] + 2.*np.cbrt(np.real(p5[zeroInds])) + (2.*C[zeroInds] + p4[zeroInds])/p9[zeroInds])/2.
        x2[zeroInds] = -A[zeroInds]/4. + p9[zeroInds]/2. - np.sqrt(p11[zeroInds] + 2.*np.cbrt(np.real(p5[zeroInds])) + (-2.*C[zeroInds] - p4[zeroInds])/p9[zeroInds])/2.
        x3[zeroInds] = -A[zeroInds]/4. + p9[zeroInds]/2. + np.sqrt(p11[zeroInds] + 2.*np.cbrt(np.real(p5[zeroInds])) + (-2.*C[zeroInds] - p4[zeroInds])/p9[zeroInds])/2.

    delta = 256.*D**3. - 192.*A*C*D**2. - 128.*B**2.*D**2. + 144.*B*C**2.*D - 27.*C**4.\
        + 144.*A**2.*B*D**2. - 6.*A**2.*C**2.*D - 80.*A*B**2.*C*D + 18.*A*B*C**3. + 16.*B**4.*D\
        - 4.*B**3.*C**2. - 27.*A**4.*D**2. + 18.*A**3.*B*C*D - 4.*A**3.*C**3. - 4.*A**2.*B**3.*D + A**2.*B**2.*C**2. #verified against wikipedia multiple times
    assert 0 == np.count_nonzero(np.imag(delta)), 'Not all delta are real'
    delta = np.real(delta)
    P = 8.*B - 3.*A**2.
    assert 0 == np.count_nonzero(np.imag(P)), 'Not all P are real'
    P = np.real(P)
    D2 = 64.*D - 16.*B**2. + 16.*A**2.*B - 16.*A*C - 3.*A**4. #is 0 if the quartic has 2 double roots 
    assert 0 == np.count_nonzero(np.imag(D2)), 'Not all D2 are real'
    D2 = np.real(D2)
    R = A**3. + 8.*C - 4.*A*B
    assert 0 == np.count_nonzero(np.imag(R)), 'Not all R are real'
    R = np.real(R)
    delta_0 = B**2. - 3.*A*C + 12.*D
    assert 0 == np.count_nonzero(np.imag(delta_0)), 'Not all delta_0 are real'
    delta_0 = np.real(delta_0)

    return np.asarray([x0, x1, x2, x3]).T, delta, P, D2, R, delta_0



### LEAVING THIS HERE, HAVE ALTERNATIVE
# #### SOLUTIONS TO ds2_dxe = 0
# phi = (a**2 - b**2)
# theta = (-a**6 + 2*a**4*b**2 + a**4*x**2 - a**2*b**4 + a**2*b**2*y**2)
# gamma = (a**4*x**2/(2*phi**2) - theta/(2*(a**4 - 2*a**2*b**2 + b**4)))
# beta = (3*a**4*x**2/(64*phi**2) - theta/(16*(a**4 - 2*a**2*b**2 + b**4)))
# alpha = (-3*a**4*x**2/(2*phi**2) + theta/(a**4 - 2*a**2*b**2 + b**4))
# epsilon = (-a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) + 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi)
# gamma = (-(2*a**4*x/phi - 2*a**2*x*gamma/phi)**2/8 - alpha**3/108 + alpha*epsilon/3)
# gorg = (a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) - 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi - alpha**2/12)
# mini = ((2*a**4*x/phi - 2*a**2*x*gamma/phi)**2/16 + alpha**3/216 - alpha*epsilon/6 + sqrt(gorg**3/27 + gamma**2/4))

# Piecewise((a**2*x/(2*phi) - sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 - sqrt(2*a**4*x**2/phi**2 + (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gamma**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 

# Eq(-a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) + 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi + alpha**2/12, 0)),
# (a**2*x/(2*phi) - sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 - sqrt(2*a**4*x**2/phi**2 + (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gorg/(3*mini**(1/3)) - 2*mini**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, True)),


# Piecewise((a**2*x/(2*phi) - sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 + sqrt(2*a**4*x**2/phi**2 + (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gamma**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, 

# Eq(-a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) + 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi + alpha**2/12, 0)), 
# (a**2*x/(2*phi) - sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 + sqrt(2*a**4*x**2/phi**2 + (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gorg/(3*mini**(1/3)) - 2*mini**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, True)),


# Piecewise((a**2*x/(2*phi) + sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 - sqrt(2*a**4*x**2/phi**2 - (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gamma**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, 

# Eq(-a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) + 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi + alpha**2/12, 0)), 
# (a**2*x/(2*phi) + sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 - sqrt(2*a**4*x**2/phi**2 - (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gorg/(3*mini**(1/3)) - 2*mini**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, True))


# Piecewise((a**2*x/(2*phi) + sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 + sqrt(2*a**4*x**2/phi**2 - (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gamma**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gamma**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, 

# Eq(-a**6*x**2/(a**4 - 2*a**2*b**2 + b**4) + 2*a**2*x*(a**4*x/(2*phi) - 2*a**2*x*beta/phi)/phi + alpha**2/12, 0)), 
# (a**2*x/(2*phi) + sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2 + sqrt(2*a**4*x**2/phi**2 - (4*a**4*x/phi - 4*a**2*x*gamma/phi)/sqrt(a**4*x**2/phi**2 - 2*gorg/(3*mini**(1/3)) + 2*mini**(1/3) - 2*theta/(3*(a**4 - 2*a**2*b**2 + b**4))) + 2*gorg/(3*mini**(1/3)) - 2*mini**(1/3) - 4*theta/(3*(a**4 - 2*a**2*b**2 + b**4)))/2, True))


def smin_smax_slmin_slmax(n, xreal, yreal, mx, my, x, y):
    """ Calculates the planet-star separation extrema, the x/y coordinates that each of these extrema occur at, the planet inds which have 4 extrema (min, max, local min, local max), and 
    the planet inds which have 2 extrema (min and max)
    Args:
        n (integer):
            number of planets
        xreal (numpy array):
            n x 4 array of the x-coordinate solutions to the quartic 
        yreal (numpy array):
            n x 4 array of the y-coordinate solutions corresponding to |xreal|
        mx (numpy array):
            star center x componentx located solely in the first quadrant, |x|
        my (numpy array):
            star center y componentx located solely in the first quadrant, |y|
        x (numpy array):
            star center x components
        y (numpy array):
            star cetner y components
    Returns:
        minSepPoints_x (numpy array):
            the first quadrant x-coordinates of the minimum separations (with length n)
        minSepPoints_y (numpy array):
            the first quadrant y-coordinates of the minimum separations (with length n)
        maxSepPoints_x (numpy array):
            the first quadrant x-coordinates of the maximum separations (with length n)
        maxSepPoints_y (numpy array):
            the first quadrant y-coordinates of the maximum separations (with length n)
        lminSepPoints_x (numpy array):
            the first quadrant x-coordinates of the local minimum separations (with same length as yrealImagInds)
        lminSepPoints_y (numpy array):
            the first quadrant y-coordinates of the local minimum separations (with same length as yrealImagInds)
        lmaxSepPoints_x (numpy array):
            the first quadrant x-coordinates of the local maximum separations (with same length as yrealImagInds)
        lmaxSepPoints_y (numpy array):
            the first quadrant y-coordinates of the local maximum separations (with same length as yrealImagInds)
        minSep (numpy array):
            the minimum planet-star separations for each star with length n
        maxSep (numpy array):
            the maximum planet-star separations for each star with length n
        lminSep (numpy array):
            the local minimum planet-star separations for each star with same length as yrealImagInds
        lmaxSep (numpy array):
            the local maximum planet-star separations for each star with same length as yrealImagInds
        yrealAllRealInds (numpy array):
            an array of integers acting as indicies of planets which have min, max, local min, local max
        yrealImagInds (numpy array):
            an array of integers acting as indicies of planets whihc only have min, max (no local min or local max)
    """
    yrealAllRealInds = np.where(np.all(np.abs(np.imag(yreal)) < 1e-5,axis=1))[0]
    yreal[np.abs(np.imag(yreal)) < 1e-5] = np.real(yreal[np.abs(np.imag(yreal)) < 1e-5]) #eliminate any unreasonably small imaginary components
    yrealImagInds = np.where(np.any(np.abs(np.imag(yreal)) >= 1e-5,axis=1))[0] #inds where any of the values are imaginary
    assert len(yrealImagInds) + len(yrealAllRealInds) == n, 'For some reason, this sum does not account for all planets'
    assert len(np.intersect1d(yrealImagInds,yrealAllRealInds)) == 0, 'For some reason, this sum does not account for all planets'
    #The following 7 lines can be deleted. it just says the first 2 cols of yreal have the smallest imaginary component
    yrealImagArgsortInds = np.argsort(np.abs(np.imag(yreal[yrealImagInds])),axis=1)
    assert len(yrealImagInds) == np.count_nonzero(yrealImagArgsortInds[:,0] == 0), "Not all first indicies have smallest Imag component"
    assert len(yrealImagInds) == np.count_nonzero(yrealImagArgsortInds[:,1] == 1), "Not all first indicies have second smallest Imag component"
    #maxImagFirstCol = np.max(np.imag(yreal[yrealImagInds,0]))
    #DELETEassert np.max(np.imag(yreal[yrealImagInds,0])) == 0, 'max y imag component of column 0 is not 0'
    if len(yrealImagInds) > 0:
        assert np.max(np.imag(yreal[yrealImagInds,0])) < 1e-4, 'max y imag component of column 0 is not 0'
    #maxImagSecondCol = np.max(np.imag(yreal[yrealImagInds,1]))
    #DELETEassert np.max(np.imag(yreal[yrealImagInds,1])) == 0, 'max y imag component of column 1 is not 0'
    if len(yrealImagInds) > 0:
        assert np.max(np.imag(yreal[yrealImagInds,1])) < 1e-4, 'max y imag component of column 1 is not 0'
    # np.max(np.imag(yreal[yrealImagInds,2])) #this is quite large
    # np.max(np.imag(yreal[yrealImagInds,3])) #this is quite large

    #Initialize minSep, lminSep, lmaxSep, maxSep
    minSep = np.zeros(xreal.shape[0])
    maxSep = np.zeros(xreal.shape[0])
    lminSep = np.zeros(len(yrealAllRealInds))
    lmaxSep = np.zeros(len(yrealAllRealInds))
    #Initialize minSepPoints_x, minSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, maxSepPoints_x, maxSepPoints_y
    minSepPoints_x = np.zeros(xreal.shape[0])
    minSepPoints_y = np.zeros(xreal.shape[0])
    lminSepPoints_x = np.zeros(len(yrealAllRealInds))
    lminSepPoints_y = np.zeros(len(yrealAllRealInds))
    lmaxSepPoints_x = np.zeros(len(yrealAllRealInds))
    lmaxSepPoints_y = np.zeros(len(yrealAllRealInds))
    maxSepPoints_x = np.zeros(xreal.shape[0])
    maxSepPoints_y = np.zeros(xreal.shape[0])
    ###################################################################################
    #### Smin and Smax Two Real Solutions Two Imaginary Solutions #####################
    #Smin and Smax need to be calculated separately for x,y with imaginary solutions vs those without
    if len(yrealImagInds) > 0:
        #For yrealImagInds. Smin and Smax must be either first column of second column
        assert np.all(np.real(xreal[yrealImagInds,0]) >= 0), 'not all xreal components are strictly negative'
        #assert np.all(np.real(xreal[yrealImagInds,0]) < 0), 'not all xreal components are strictly negative'
        assert np.all(np.real(yreal[yrealImagInds,0]) >= 0), 'not all yreal components are strictly positive'
        assert np.all(np.real(xreal[yrealImagInds,1]) >= 0), 'not all xreal components are strictly positive'
        assert np.all(np.real(yreal[yrealImagInds,1]) >= 0), 'not all yreal components are strictly positive'
        smm0 = np.sqrt((np.real(xreal[yrealImagInds,0])-mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,0])-my[yrealImagInds])**2)
        smp0 = np.sqrt((np.real(xreal[yrealImagInds,0])-mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,0])+my[yrealImagInds])**2)
        spm0 = np.sqrt((np.real(xreal[yrealImagInds,0])+mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,0])-my[yrealImagInds])**2)
        spp0 = np.sqrt((np.real(xreal[yrealImagInds,0])+mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,0])+my[yrealImagInds])**2)
        smm1 = np.sqrt((np.real(xreal[yrealImagInds,1])-mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,1])-my[yrealImagInds])**2)
        smp1 = np.sqrt((np.real(xreal[yrealImagInds,1])-mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,1])+my[yrealImagInds])**2)
        spm1 = np.sqrt((np.real(xreal[yrealImagInds,1])+mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,1])-my[yrealImagInds])**2)
        spp1 = np.sqrt((np.real(xreal[yrealImagInds,1])+mx[yrealImagInds])**2 + (np.real(yreal[yrealImagInds,1])+my[yrealImagInds])**2)

        #Search for Smallest
        smm = np.asarray([smm0,smm1])
        #DELETEassert np.all(np.argmin(smm,axis=0) == 1), 'mins are not all are smm1'
        if not np.all(np.argmin(smm,axis=0) == 1):
            #do some additional checking
            print('mins are not all are smm1')
            inds = np.where(np.argmin(smm,axis=0) == 1)[0] #inds to check
            if np.all(np.abs(smm0[inds] - smm1[inds]) < 1e-8):
                tmp = smm0[inds]
                smm0[inds] = smm1[inds]
                smm1[inds] = tmp
            smm = np.asarray([smm0,smm1])
        smp = np.asarray([smp0,smp1])
        if not np.all(np.argmin(smp,axis=0) == 0):
            indsOf0 = np.where(np.argmin(smp,axis=0) == 0)[0]
            for i in np.arange(len(indsOf0)):
                if np.abs(smp0[indsOf0[i]] - smp1[indsOf0[i]]) < 1e-5:
                    #just swap them
                    tmp = smp0[indsOf0[i]]
                    smp0[indsOf0[i]] = smp1[indsOf0[i]]
                    smp1[indsOf0[i]] = tmp
        smp = np.asarray([smp0,smp1])
        assert np.all(np.argmin(smp,axis=0) == 0), 'mins are not all are smp0'
        #myInd = yrealImagInds[np.where(np.logical_not(np.argmin(smp,axis=0) == 0))[0]]
        #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
        spm = np.asarray([spm0,spm1])
        if not np.all(np.argmin(spm,axis=0) == 1):
            indsOf0 = np.where(np.argmin(spm,axis=0) == 0)[0]
            for i in np.arange(len(indsOf0)):
                if np.abs(spm0[indsOf0[i]] - spm1[indsOf0[i]]) < 1e-5:
                    #just swap them
                    tmp = spm0[indsOf0[i]]
                    spm0[indsOf0[i]] = spm1[indsOf0[i]]
                    spm1[indsOf0[i]] = tmp
        spm = np.asarray([spm0,spm1])
        assert np.all(np.argmin(spm,axis=0) == 1), 'mins are not all are spm1'
        #myInd = yrealImagInds[np.where(np.logical_not(np.argmin(spm,axis=0) == 1))[0]]
        #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
        spp = np.asarray([spp0,spp1])
        #DELETEassert np.all(np.argmin(spp,axis=0) == 1), 'mins are not all are spp1'
        if not np.all(np.argmin(spp,axis=0) == 1):
            #do some additional checking
            print('mins are not all are spp1')
            inds = np.where(np.argmin(spp,axis=0) == 1)[0] #inds to check
            if np.all(np.abs(spp0[inds] - spp1[inds]) < 1e-8):
                tmp = spp0[inds]
                spp0[inds] = spp1[inds]
                spp1[inds] = tmp
            spp = np.asarray([spp0,spp1])
        #above says smallest must be one of these: smm1, smp0, spm1, spp1
        #The following are where each of these separations are 0
        smm1Inds = np.where((smm1 <= smp0)*(smm1 <= spm1)*(smm1 <= spp1))[0]
        smp0Inds = np.where((smp0 < smm1)*(smp0 <= spm1)*(smp0 <= spp1))[0]
        #spp1Inds = np.where((smp0 < smm1)*(smp0 < spm1)*(smp0 < spp1))[0]
        spm1Inds = np.where((spm1 < smp0)*(spm1 < smm1)*(spm1 <= spp1))[0]
        spp1Inds = np.where((spp1 < smp0)*(spp1 < spm1)*(spp1 < smm1))[0]
        assert len(yrealImagInds) == len(smm1Inds) + len(smp0Inds) + len(spm1Inds) + len(spp1Inds), 'Have not covered all cases'
        #set(np.arange(len(yrealImagInds)))^set(smm1Inds).union(set(smp0Inds).union(set(spm1Inds).union(set(spp1Inds))))
        if len(smm1Inds) > 0:
            minSep[yrealImagInds[smm1Inds]] = smm1[smm1Inds]
            minSepPoints_x[yrealImagInds[smm1Inds]] = np.real(xreal[yrealImagInds[smm1Inds],1])
            minSepPoints_y[yrealImagInds[smm1Inds]] = np.real(yreal[yrealImagInds[smm1Inds],1])
        if len(smp0Inds) > 0:
            minSep[yrealImagInds[smp0Inds]] = smp0[smp0Inds]
            minSepPoints_x[yrealImagInds[smp0Inds]] = np.real(xreal[yrealImagInds[smp0Inds],0])
            minSepPoints_y[yrealImagInds[smp0Inds]] = np.real(yreal[yrealImagInds[smp0Inds],0])
        if len(spm1Inds) > 0:
            minSep[yrealImagInds[spm1Inds]] = spm1[spm1Inds]#smp0[spm1Inds] #the commented is the original which appeared to work flawlessly
            minSepPoints_x[yrealImagInds[spm1Inds]] = np.real(xreal[yrealImagInds[spm1Inds],1])
            minSepPoints_y[yrealImagInds[spm1Inds]] = np.real(yreal[yrealImagInds[spm1Inds],1])
        if len(spp1Inds) > 0:
            minSep[yrealImagInds[spp1Inds]] = spp1[spp1Inds]
            minSepPoints_x[yrealImagInds[spp1Inds]] = np.real(xreal[yrealImagInds[spp1Inds],1])
            minSepPoints_y[yrealImagInds[spp1Inds]] = np.real(yreal[yrealImagInds[spp1Inds],1])
        #above says largest must be one of these: smm0, smp0, spm1, spp1
        smm0Inds = np.where((smm0 >= smp0)*(smm0 >= spm1)*(smm0 >= spp0))[0] #used to not have equals
        smp0Inds = np.where((smp0 > smm0)*(smp0 >= spm1)*(smp0 >= spp0))[0] #used to not have equals
        spm1Inds = np.where((spm1 > smp0)*(spm1 > smm0)*(spm1 >= spp0))[0] #used to not have equals
        spp0Inds = np.where((spp0 > smp0)*(spp0 > spm1)*(spp0 > smm0))[0]
        if len(smm0Inds) > 0:
            maxSep[yrealImagInds[smm0Inds]] = smm0[smm0Inds]
            maxSepPoints_x[yrealImagInds[smm0Inds]] = np.real(xreal[yrealImagInds[smm0Inds],0])
            maxSepPoints_y[yrealImagInds[smm0Inds]] = np.real(yreal[yrealImagInds[smm0Inds],0])
        if len(smp0Inds) > 0:
            maxSep[yrealImagInds[smp0Inds]] = smp0[smp0Inds]
            maxSepPoints_x[yrealImagInds[smp0Inds]] = np.real(xreal[yrealImagInds[smp0Inds],0])
            maxSepPoints_y[yrealImagInds[smp0Inds]] = np.real(yreal[yrealImagInds[smp0Inds],0])
        if len(spm1Inds) > 0:
            maxSep[yrealImagInds[spm1Inds]] = spm1[spm1Inds]
            maxSepPoints_x[yrealImagInds[spm1Inds]] = np.real(xreal[yrealImagInds[spm1Inds],1])
            maxSepPoints_y[yrealImagInds[spm1Inds]] = np.real(yreal[yrealImagInds[spm1Inds],1])
        if len(spp0Inds) > 0:
            maxSep[yrealImagInds[spp0Inds]] = spp0[spp0Inds]
            maxSepPoints_x[yrealImagInds[spp0Inds]] = np.real(xreal[yrealImagInds[spp0Inds],0])
            maxSepPoints_y[yrealImagInds[spp0Inds]] = np.real(yreal[yrealImagInds[spp0Inds],0])

        assert len(yrealImagInds) == len(smm0Inds) + len(smp0Inds) + len(spm1Inds) + len(spp0Inds), 'Not all Inds accounted for'
        #len(smm1Inds) + len(spm1Inds) + len(spp1Inds) + len()

    #not currentyl assigning x,y values or lmin lmax for 2 solutions with 2 complex
    ########################################################
    #### 4 Real Solutions ##################################
    smm = np.zeros((4,len(yrealAllRealInds)))
    smp = np.zeros((4,len(yrealAllRealInds)))
    #spm = np.zeros((4,len(yrealAllRealInds))) #removed for efficiency
    spp = np.zeros((4,len(yrealAllRealInds)))
    for i in [0,1,2,3]:
        smm[i] = np.sqrt((np.real(xreal[yrealAllRealInds,i])-mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,i]))-my[yrealAllRealInds])**2)
        smp[i] = np.sqrt((np.real(xreal[yrealAllRealInds,i])-mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,i]))+my[yrealAllRealInds])**2)
        #spm[i] = np.sqrt((np.real(xreal[yrealAllRealInds,i])+mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,i]))-my[yrealAllRealInds])**2) #removed for efficiency
        spp[i] = np.sqrt((np.real(xreal[yrealAllRealInds,i])+mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,i]))+my[yrealAllRealInds])**2)
    smm = smm.T
    smp = smp.T
    #spm = spm.T #removed for efficiency
    spp = spp.T

    #### minSep
    minSep[yrealAllRealInds] = smm[:,1]
    minSepPoints_x[yrealAllRealInds] = np.real(xreal[yrealAllRealInds,1])
    minSepPoints_y[yrealAllRealInds] = np.real(yreal[yrealAllRealInds,1])
    ####

    #### maxSep
    maxSep[yrealAllRealInds] = spp[:,0]
    maxSepPoints_x[yrealAllRealInds] = np.real(xreal[yrealAllRealInds,0])
    maxSepPoints_y[yrealAllRealInds] = np.real(yreal[yrealAllRealInds,0])
    ####

    #Calcs for local extrema
    smp2 = np.sqrt((np.real(xreal[yrealAllRealInds,2])-mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,2]))+my[yrealAllRealInds])**2)
    smp3 = np.sqrt((np.real(xreal[yrealAllRealInds,3])-mx[yrealAllRealInds])**2 + (np.abs(np.real(yreal[yrealAllRealInds,3]))+my[yrealAllRealInds])**2)
    smp = np.asarray([smp2,smp3]).T
    #### slmax
    slmaxInds = np.argmax(smp,axis=1)
    lmaxSep = smp[np.arange(len(yrealAllRealInds)),slmaxInds]
    lmaxSepPoints_x = xreal[yrealAllRealInds, 2+slmaxInds]
    lmaxSepPoints_y = yreal[yrealAllRealInds, 2+slmaxInds]
    #### slmin
    slminInds = np.argmin(smp,axis=1)
    lminSep = smp[np.arange(len(yrealAllRealInds)),slminInds]
    lminSepPoints_x = xreal[yrealAllRealInds, 2+slminInds]
    lminSepPoints_y = yreal[yrealAllRealInds, 2+slminInds]
    ####

    #Comment
    #Checks on minSep <= lminSep <= lmaxSep <= maxSep
    assert ~np.any(minSep == 0), 'Oops, a minSep was missed'
    assert ~np.any(maxSep == 0), 'Oops, a maxSep was missed'
    assert np.all(minSep[yrealAllRealInds] <= lminSep), 'Not all minSep < lminSep'
    #DELETEassert np.all(maxSep[yrealAllRealInds] >= lmaxSep), 'Not all maxSep > lmaxSep'
    assert np.all(maxSep[yrealAllRealInds] - lmaxSep > -1e12), 'Not all maxSep > lmaxSep'
    assert np.all(lminSep <= lmaxSep), 'Not all lminSep < lmaxSep'
    ################################################################################################# Done with separations

    #KEEP
    #Quadrant Star Belongs to
    bool1 = x > 0
    bool2 = y > 0
    #Quadrant 1 if T,T
    #Quadrant 2 if F,T
    #Quadrant 3 if F,F
    #Quadrant 4 if T,F
    #### Min Sep Point (Points on plot of Min Sep)
    minSepPoints_x = minSepPoints_x*(2*bool1-1)
    minSepPoints_y = minSepPoints_y*(2*bool2-1)
    #### Max Sep Point (Points on plot of max sep)
    maxSepPoints_x = maxSepPoints_x*(-2*bool1+1)
    maxSepPoints_y = maxSepPoints_y*(-2*bool2+1)
    #### Local Min Sep Points
    lminSepPoints_x = np.real(lminSepPoints_x*(2*bool1[yrealAllRealInds]-1))
    lminSepPoints_y = np.real(lminSepPoints_y*(-2*bool2[yrealAllRealInds]+1))
    #### Local Max Sep Points
    lmaxSepPoints_x = np.real(lmaxSepPoints_x*(2*bool1[yrealAllRealInds]-1))
    lmaxSepPoints_y = np.real(lmaxSepPoints_y*(-2*bool2[yrealAllRealInds]+1))

    return minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds

def trueAnomalyFromXY(X,Y,W,w,inc):
    """ Calculated true anomaly from X, Y, and KOE
    These nus are not necessarily the actual value
    due to the use of arctan. A subsequent fixer function exists
    Args:
        X (numpy array): 
            x component of body in 3D elliptical orbit
        Y (numpy array): 
            y component of body in 3D elliptical orbit
        W (numpy array): 
            Longitude of the ascending node of the body
        w (numpy array): 
            argument of periapsis of the body
        inc (numpy array): 
            inclination of the body's orbit
    Returns:
        nu (numpy array):
            true anomalies in radians
    """
    #nu = np.arctan2( -X/Y*np.sin(W)*np.cos(w) -X/Y*np.cos(W)*np.cos(inc)*np.sin(w) + np.cos(W)*np.cos(w) - np.sin(W)*np.cos(inc)*np.cos(w),\
    #            -X/Y*np.sin(W)*np.sin(w) + X/Y*np.cos(W)*np.cos(inc)*np.cos(w) + np.cos(W)*np.sin(w) + np.sin(W)*np.cos(inc)*np.cos(w) ) #Manual Typing

    #Jupyter Notebook
    nu = np.arctan2(np.sin(W)*np.cos(w)/Y + np.sin(w)*np.cos(W)*np.cos(inc)/Y + np.sin(W)*np.sin(w)*np.cos(inc)/X - np.cos(W)*np.cos(w)/X,\
                    np.sin(W)*np.sin(w)/Y - np.cos(W)*np.cos(inc)*np.cos(w)/Y - np.sin(W)*np.cos(inc)*np.cos(w)/X - np.sin(w)*np.cos(W)/X)

    return nu

def timeFromTrueAnomaly(nu,T,e):
    """ Time (since periastron) from true anomaly
    Args:
        nu (numpy array):
            true anomalies
        T (numpy array):
            orbital periods
        e (numpy array):
            planet eccentricities
    Returns:
        time (numpy array):
            time past periastron corresponding to the input true anomaly 
    """

    E = np.arctan2(np.sqrt(1.-e**2.)*np.sin(nu),e+np.cos(nu))
    E = np.mod(E,2.*np.pi)
    t = (E-e*np.sin(E))/(2.*np.pi/T)
    return t

def printKOE(ind,a,e,W,w,inc):
    print('a: ' + str(np.round(sma[ind],2)) + ' e: ' + str(np.round(e[ind],2)) + ' W: ' + str(np.round(W[ind],2)) + ' w: ' + str(np.round(w[ind],2)) + ' i: ' + str(np.round(inc[ind],2)))# + ' theta: ' + str(np.round(theta[ind],2)))
    return None

#### Ellipse Circle Intersection
def ellipseCircleIntersections(s_circle, a, b, mx, my, x, y, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds, sma=None, e=None, W=None, w=None, inc=None):
    """ Calculates the intersections between a circle centered at x,y with radius s_circle and an ellipse centered at x=0, y=0 with semi-major axis aligned with x-axis
    Args:
        s_circle (float):
            the circle radius in AU PASSING IN S_CIRCLE AS AN ARRAY IS DUMB
        a (numpy array):
            semi-major axis of the projected ellipse
        b (numpy array):
            semi-minor axis of the projected ellipse
        mx (numpy array):
            star center x componentx located solely in the first quadrant, |x|
        my (numpy array):
            star center y componentx located solely in the first quadrant, |y|
        x (numpy array):
            star center x components
        y (numpy array):
            star cetner y components
        minSep (numpy array):
            the minimum planet-star separations for each star with length n
        maxSep (numpy array):
            the maximum planet-star separations for each star with length n
        lminSep (numpy array):
            the local minimum planet-star separations for each star with same length as yrealImagInds
        lmaxSep (numpy array):
            the local maximum planet-star separations for each star with same length as yrealImagInds
        yrealAllRealInds (numpy array):
            an array of integers acting as indicies of planets which have min, max, local min, local max
        yrealImagInds (numpy array):
            an array of integers acting as indicies of planets whihc only have min, max (no local min or local max)
        sma, e, W, w, inc are all able to be deleted. For debugging purposes only
    Returns: 
        only2RealInds (numpy array):
            indicies where there can only ever by 2 circle-ellipse intersections
        typeInds0 (numpy array):
            inds of only2RealInds where s_x,b-y < s_a-x,y < s_a+x,y < s_x,b+y
            used to determine solutions for only2RealInds 
        typeInds1 (numpy array):
            inds of only2RealInds where s_x,b-y < s_x,b+y < s_a-x,y < s_a+x,y
            used to determine solutions for only2RealInds 
        typeInds2 (numpy array):
            inds of only2RealInds where s_x,b-y < s_a-x,y < s_x,b+y < s_a+x,y
            used to determine solutions for only2RealInds 
        typeInds3 (numpy array):
            inds of only2RealInds where s_a-x,y < s_x,b-y < s_x,b+y < s_a+x,y
            used to determine solutions for only2RealInds 
        fourIntInds (numpy array):
            indicies of yrealAllRealInds which should have 4 intersections
        fourInt_x (numpy array):
            x coordinates of fourIntInds
        fourInt_y (numpy array):
            y coordinates of fourIntInds
        twoIntSameY_x (numpy array):
            x components of intersections which must occur on same Y side of the projected ellipse as the star
        twoIntSameY_y (numpy array):
            y components of intersections which must occur on same Y side of the projected ellipse as the star
        twoIntOppositeXInds (numpy array):
            indicies of yrealAllRealInds which should have 2 intersections on the 
            opposite X side of the projected ellipse as the star
        twoIntOppositeX_x (numpy array):
            x components of intersections which must occur on opposite X side of the projected ellipse as the star
        twoIntOppositeX_y (numpy array):
            y components of intersections which must occur on opposite X side of the projected ellipse as the star
        xIntersectionsOnly2 (numpy array):
            x components of intersections where there must be only 2 intersections
        yIntersectionsOnly2 (numpy array):
            y components of intersections where there must be only 2 intersections
        twoIntSameYInds (numpy array):
            indicies of yrealAllRealInds which should have 2 intersections on the 
            same Y side of the projected ellipse as the star
        type0_0Inds (numpy array):
        type0_1Inds (numpy array):
        type0_2Inds (numpy array):
        type0_3Inds (numpy array):
        type0_4Inds (numpy array):
        type1_0Inds (numpy array):
        type1_1Inds (numpy array):
        type1_2Inds (numpy array):
        type1_3Inds (numpy array):
        type1_4Inds (numpy array):
        type2_0Inds (numpy array):
        type2_1Inds (numpy array):
        type2_2Inds (numpy array):
        type2_3Inds (numpy array):
        type2_4Inds (numpy array):
        type3_0Inds (numpy array):
        type3_1Inds (numpy array):
        type3_2Inds (numpy array):
        type3_3Inds (numpy array):
        type3_4Inds (numpy array):
        allIndsUsed (numpy array):
            contains all typeA_BInds concatenated. typeA_BInds can be deleted once everything works 100%
    """

    #### Quadrant Star Belongs to #######
    bool1 = x > 0
    bool2 = y > 0
    #Quadrant 1 if T,T
    #Quadrant 2 if F,T
    #Quadrant 3 if F,F
    #Quadrant 4 if T,F
    ####################################

    #### Testing ellipse_to_Quartic solution
    #DELETEif r == None:
    #DELETEr = np.ones(len(a),dtype='complex128')
    #ARE THESE NECESSARY?
    #DELETEa.astype('complex128')
    #DELETEab.astype('complex128')
    #DELETEamx.astype('complex128')
    #DELETEamy.astype('complex128')
    #DELETEas_circle.astype('complex128')
    A, B, C, D = quarticCoefficients_ellipse_to_Quarticipynb(a, b, mx, my, s_circle) #extract quartic coefficients from the given star location, ellipse, and circle radius
    xreal2, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D) #Solve the quartic expression
    yreal2 = ellipseYFromX(xreal2.astype('complex128'), a, b) #Calculate Y components of intersection x solutions

    #### Of All Real Inds (yrealAllRealInds) find bool array where r is... ################
    #Output bool arrays are used in the next section
    gtMinSepBool = (minSep[yrealAllRealInds] < s_circle[yrealAllRealInds])
    ltMaxSepBool = (maxSep[yrealAllRealInds] >= s_circle[yrealAllRealInds])
    gtLMaxSepBool = (lmaxSep < s_circle[yrealAllRealInds])
    ltLMaxSepBool = (lmaxSep > s_circle[yrealAllRealInds])
    gtLMinSepBool = (lminSep <= s_circle[yrealAllRealInds])
    ltLMinSepBool = (lminSep > s_circle[yrealAllRealInds])

    #Two intersections on same y-side of projected ellipse
    twoIntSameYInds = np.where(gtMinSepBool*ltLMinSepBool)[0]
    #Four intersections total
    fourIntInds = np.where(gtLMinSepBool*ltLMaxSepBool)[0]
    #Two intersections opposite x-side
    twoIntOppositeXInds = np.where(ltMaxSepBool*gtLMaxSepBool)[0]
    del gtMinSepBool, ltMaxSepBool, gtLMaxSepBool, ltLMaxSepBool, gtLMinSepBool, ltLMinSepBool #for memory efficiency

    #Solution Checks
    if not len(fourIntInds) == 0: #If this is the case, np.max does not work on empty sets
        if np.max(np.abs(np.imag(xreal2[yrealAllRealInds[fourIntInds]]))) >= 1e-3:
            myInds = np.where(np.abs(np.imag(xreal2[yrealAllRealInds[fourIntInds],0])) >= 1e-3)[0]
            for myInd in myInds:
                with open("/home/dean/Documents/exosims/twoDetMC/badKOE.py","a") as f:
                    f.write('#ellipseCircleIntersections 1' + '\n')
                    f.write('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]) + '\n')
                    f.write('badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)' + '\n')
                    error = minSep
        assert np.max(np.abs(np.imag(xreal2[yrealAllRealInds[fourIntInds]]))) < 1e-2, 'an Imag component of the all reals is too high!' #uses to be 1e-7 but would occasionally get errors so relaxing
        #Actually, it seems this occurs when there is a high eccentricity and this high error is actually acceptable
    else:
        print('fourIntInds is empty') #Debuggin statement
    ########################################################################################

    #### Four Intersection Points
    fourInt_dx = (np.real(xreal2[yrealAllRealInds[fourIntInds]]).T - mx[yrealAllRealInds[fourIntInds]]).T
    fourIntSortInds = np.argsort(fourInt_dx, axis=1)
    sameYOppositeXInds = fourIntSortInds[:,0]
    #NOTE: There are some rare cases where the solutions
    #assert np.all(sameYOppositeXInds==0), 'not all 0' #Should all be 0, but occasionally this is not the case due to numerical error
    #if not np.all(sameYOppositeXInds==0):
    #    tInds = np.where(sameYOppositeXInds != 0)[0]#inds where the solution does not conform
    sameYXInds = fourIntSortInds[:,3]
    #assert np.all(sameYXInds==3), 'not all 3'
    oppositeYOppositeXInds = fourIntSortInds[:,1]
    #assert np.all(oppositeYOppositeXInds==1), 'not all 1'
    oppositeYSameXInds = fourIntSortInds[:,2]
    #assert np.all(oppositeYSameXInds==2), 'not all 2'
    fourInt_y = np.zeros((len(fourIntInds),4))
    fourInt_x = np.zeros((len(fourIntInds),4))
    fourInt_x[:,0] = np.real(xreal2[yrealAllRealInds[fourIntInds],sameYOppositeXInds])
    fourInt_x[:,1] = np.real(xreal2[yrealAllRealInds[fourIntInds],sameYXInds])
    fourInt_x[:,2] = np.real(xreal2[yrealAllRealInds[fourIntInds],oppositeYOppositeXInds])
    fourInt_x[:,3] = np.real(xreal2[yrealAllRealInds[fourIntInds],oppositeYSameXInds])
    fourInt_y = ellipseYFromX(np.abs(fourInt_x), a[yrealAllRealInds[fourIntInds]], b[yrealAllRealInds[fourIntInds]])
    fourInt_y[:,2] = -fourInt_y[:,2]
    fourInt_y[:,3] = -fourInt_y[:,3]
    #### Four Intercept Points
    fourInt_x = (fourInt_x.T*(2*bool1[yrealAllRealInds[fourIntInds]]-1)).T
    fourInt_y = (fourInt_y.T*(2*bool2[yrealAllRealInds[fourIntInds]]-1)).T
    ####

    #### Two Intersection Points twoIntSameYInds
    twoIntSameY_x = np.zeros((len(twoIntSameYInds),2))
    twoIntSameY_y = np.zeros((len(twoIntSameYInds),2))
    #DELETEassert np.max(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],0])) < 1e-8, 'An Imaginary component was too large' #Was 1e-12, but is now 1e-8
    if len(twoIntSameYInds) > 0:
        #assert np.max(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],0])) < 1e-5, 'An Imaginary component was too large' #Was 1e-12, but is now 1e-5
        # There was a specific case where the imaginary component was as high as 0.26, but th e real components of the solution were correct... so I am not complaining
        # Here are the specifics of the case causing the issue. I verified the real components of the solution were correct via plotting
        # a = 2.4946080925534058
        # b = 0.3800927979713089
        # mx = 0.724828007491803
        # my = 0.09703406084684035
        # s_circle = 0.45
        # We do need to nuke the imaginary component of a solution with imag > 1e-5
        afflictedInds = np.where(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],0]) < 1e-5)[0]
        xreal2[yrealAllRealInds[twoIntSameYInds[afflictedInds]]] = np.real(xreal2[yrealAllRealInds[twoIntSameYInds[afflictedInds]]]) #Removes the imaginary component
    twoIntSameY_x[:,0] = np.real(xreal2[yrealAllRealInds[twoIntSameYInds],0])
    smallImagInds = np.where(np.abs(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],1])) < 1e-9)[0]
    largeImagInds = np.where(np.abs(np.imag(xreal2[yrealAllRealInds[twoIntSameYInds],1])) > 1e-9)[0]
    twoIntSameY_x[smallImagInds,1] = np.real(xreal2[yrealAllRealInds[twoIntSameYInds[smallImagInds]],1])
    twoIntSameY_x[largeImagInds,1] = np.real(xreal2[yrealAllRealInds[twoIntSameYInds[largeImagInds]],3])
    twoIntSameY_y = np.asarray([np.sqrt(b[yrealAllRealInds[twoIntSameYInds]]**2*(1-twoIntSameY_x[:,0]**2/a[yrealAllRealInds[twoIntSameYInds]]**2)),\
            np.sqrt(b[yrealAllRealInds[twoIntSameYInds]]**2*(1-twoIntSameY_x[:,1]**2/a[yrealAllRealInds[twoIntSameYInds]]**2))]).T
    #Adjust for Quadrant Star Belongs to
    twoIntSameY_x = (twoIntSameY_x.T*(2*bool1[yrealAllRealInds[twoIntSameYInds]]-1)).T
    twoIntSameY_y = (twoIntSameY_y.T*(2*bool2[yrealAllRealInds[twoIntSameYInds]]-1)).T
    #### Checking for Duplicates
    twoIntSameY_s = np.sqrt(twoIntSameY_x**2. + twoIntSameY_y**2.)


    #Doing some error checking stuffs.....
    #Pull out the 4 x sols
    x0 = np.real(xreal2[yrealAllRealInds[twoIntSameYInds],0])
    x1 = np.real(xreal2[yrealAllRealInds[twoIntSameYInds],1])
    x2 = np.real(xreal2[yrealAllRealInds[twoIntSameYInds],2])
    x3 = np.real(xreal2[yrealAllRealInds[twoIntSameYInds],3])
    xarray = np.asarray([x0,x0,-x0,-x0,x1,x1,-x1,-x1,x2,x2,-x2,-x2,x3,x3,-x3,-x3]).T
    #Calculate the 4 associated y sols
    y0 = np.sqrt(b[yrealAllRealInds[twoIntSameYInds]]**2*(1-x0**2/a[yrealAllRealInds[twoIntSameYInds]]**2))   
    y1 = np.sqrt(b[yrealAllRealInds[twoIntSameYInds]]**2*(1-x1**2/a[yrealAllRealInds[twoIntSameYInds]]**2))
    y2 = np.sqrt(b[yrealAllRealInds[twoIntSameYInds]]**2*(1-x2**2/a[yrealAllRealInds[twoIntSameYInds]]**2))
    y3 = np.sqrt(b[yrealAllRealInds[twoIntSameYInds]]**2*(1-x3**2/a[yrealAllRealInds[twoIntSameYInds]]**2))
    yarray = np.asarray([y0,-y0,y0,-y0,y1,-y1,y1,-y1,y2,-y2,y2,-y2,y3,-y3,y3,-y3]).T
    #Calculate Sep array
    separray = np.sqrt((xarray-np.tile(x[yrealAllRealInds[twoIntSameYInds]],(16,1)).T)**2.+(yarray-np.tile(y[yrealAllRealInds[twoIntSameYInds]],(16,1)).T)**2.)
    #Calculate Error Array
    errorarray = np.abs(separray - np.tile(s_circle[yrealAllRealInds[twoIntSameYInds]],(16,1)).T)
    #np.histogram(np.sum(errorarray < 1e-5,axis=1))
    errorarraybools = errorarray < 1e-4 #Tells me where the errors are small
    twoIntSameY_x = np.zeros((len(twoIntSameYInds),2)) #Construct the output arrays
    twoIntSameY_y = np.zeros((len(twoIntSameYInds),2))
    summedSolErrors = np.sum(errorarraybools,axis=1) #Find the numbers of errors that are small for the given star
    indsWith0 = np.where(summedSolErrors == 0)[0] #just relax the errors a bit
    indsWith1 = np.where(summedSolErrors == 1)[0] #just relax the errors a bit
    indsWith2 = np.where(summedSolErrors == 2)[0] #exactly the number of solutions we're looking for (HOPE THEYRE UNIQUE)
    indsWith3 = np.where(summedSolErrors == 3)[0] #exactly the number of solutions we're looking for (HOPE THEYRE UNIQUE)
    indsWith4 = np.where(summedSolErrors == 4)[0] #Need to parse out the two different low error solutions
    indsWith7 = np.where(summedSolErrors == 7)[0] #WTF
    indsWith5 = np.where(summedSolErrors == 5)[0] #WTF
    indsWith6 = np.where(summedSolErrors == 6)[0] #WTF
    indsWith8 = np.where(summedSolErrors == 8)[0] #WTF
    #assert len(indsWith0) + len(indsWith1) + len(indsWith2) + len(indsWith3) + len(indsWith4) + len(indsWith7) + len(indsWith5) + len(indsWith6) + len(indsWith8) == xarray.shape[0], "All the solutions don't add correctly"
   
    #### For inds with 9, 10, 11, 12, 13, 14, ....
    if not len(indsWith0) + len(indsWith1) + len(indsWith2) + len(indsWith3) + len(indsWith4) + len(indsWith7) + len(indsWith5) + len(indsWith6) + len(indsWith8) == xarray.shape[0]:
        mySet = set(np.arange(xarray.shape[0])) - (set(indsWith0) | set(indsWith1) | set(indsWith2) | set(indsWith3) | set(indsWith4) | set(indsWith7) | set(indsWith5) | set(indsWith6) | set(indsWith8))
        indsWith = list(mySet)
        indsOfMin = np.nanargmin(errorarray[indsWith],axis=1) #finds the ind of the minimum
        twoIntSameY_x[indsWith,0] = xarray[indsWith,indsOfMin]
        twoIntSameY_y[indsWith,0] = yarray[indsWith,indsOfMin]
        #Nan the absolute minimum in the error array and sep array
        errorarray[indsWith,indsOfMin] = np.nan
        separray[indsWith,indsOfMin] = np.nan
        #Find the ind of the second smallest
        indsOfMin2 = np.nanargmin(errorarray[indsWith],axis=1) #Finds the ind of the second minimum
        #Calculate distance between x_min,y_min and x_2min,y_2min Ensure it is sufficiently large
        xarrayIndsOfMin = xarray[indsWith,indsOfMin]
        yarrayIndsOfMin = yarray[indsWith,indsOfMin]
        xarrayIndsOfMin2 = xarray[indsWith,indsOfMin2]
        yarrayIndsOfMin2 = yarray[indsWith,indsOfMin2]
        ptptSeps = np.sqrt((xarrayIndsOfMin - xarrayIndsOfMin2)**2. + (yarrayIndsOfMin - yarrayIndsOfMin2)**2.) #Calculate the distance between the two points
        #assert np.all(ptptSeps > 1e-5), 'The points selected are too close to one another'
        #Iterete over ptptSeps, nan the solution that is too close, select another
        if np.any(ptptSeps < 1e-5):
            for i in np.where(ptptSeps < 1e-5)[0]: #iterate over all the violating stars
                while ptptSeps[i] < 1e-5 and not np.all(np.isnan(errorarray[indsWith[i]])):
                    errorarray[indsWith[i],indsOfMin2[i]] = np.nan
                    indsOfMin2[i] = np.nanargmin(errorarray[indsWith[i]]) #Finds the ind of the second minimum
                    ptptSeps[i] = np.sqrt((xarray[indsWith[i],indsOfMin[i]] - xarray[indsWith[i],indsOfMin2[i]])**2. + (yarray[indsWith[i],indsOfMin[i]] - yarray[indsWith[i],indsOfMin2[i]])**2.)
                assert not np.all(np.isnan(errorarray[indsWith[i]])), 'Looks like all points were too close for a planet'

        #If the points are all far apart, then
        twoIntSameY_x[indsWith,1] = xarray[indsWith,indsOfMin2]
        twoIntSameY_y[indsWith,1] = yarray[indsWith,indsOfMin2]


    #### For inds with 2
    #pull out 4 smallest solutions
    indsOfMin = np.nanargmin(errorarray[indsWith2],axis=1) #finds the ind of the minimum
    twoIntSameY_x[indsWith2,0] = xarray[indsWith2,indsOfMin]
    twoIntSameY_y[indsWith2,0] = yarray[indsWith2,indsOfMin]
    #Nan the absolute minimum in the error array and sep array
    errorarray[indsWith2,indsOfMin] = np.nan
    separray[indsWith2,indsOfMin] = np.nan
    #Find the ind of the second smallest
    indsOfMin2 = np.nanargmin(errorarray[indsWith2],axis=1) #Finds the ind of the second minimum
    twoIntSameY_x[indsWith2,1] = xarray[indsWith2,indsOfMin2]
    twoIntSameY_y[indsWith2,1] = yarray[indsWith2,indsOfMin2]
    
    #### For inds with 4
    if len(indsWith4) > 0:
        #pull out 4 smallest solutions
        indsOfMin = np.nanargmin(errorarray[indsWith4],axis=1) #finds the ind of the minimum
        twoIntSameY_x[indsWith4,0] = xarray[indsWith4,indsOfMin]
        twoIntSameY_y[indsWith4,0] = yarray[indsWith4,indsOfMin]
        #Nan the absolute minimum in the error array and sep array
        errorarray[indsWith4,indsOfMin] = np.nan
        separray[indsWith4,indsOfMin] = np.nan
        #Find the ind of the second smallest
        indsOfMin2 = np.nanargmin(errorarray[indsWith4],axis=1) #Finds the ind of the second minimum
        #Calculate distance between x_min,y_min and x_2min,y_2min Ensure it is sufficiently large
        xarrayIndsOfMin = xarray[indsWith4,indsOfMin]
        yarrayIndsOfMin = yarray[indsWith4,indsOfMin]
        xarrayIndsOfMin2 = xarray[indsWith4,indsOfMin2]
        yarrayIndsOfMin2 = yarray[indsWith4,indsOfMin2]
        ptptSeps = np.sqrt((xarrayIndsOfMin - xarrayIndsOfMin2)**2. + (yarrayIndsOfMin - yarrayIndsOfMin2)**2.) #Calculate the distance between the two points
        #assert np.all(ptptSeps > 1e-5), 'The points selected are too close to one another'
        #Iterete over ptptSeps, nan the solution that is too close, select another
        if np.any(ptptSeps < 1e-5):
            for i in np.where(ptptSeps < 1e-5)[0]: #iterate over all the violating stars
                while ptptSeps[i] < 1e-5 and not np.all(np.isnan(errorarray[indsWith4[i]])):
                    errorarray[indsWith4[i],indsOfMin2[i]] = np.nan
                    indsOfMin2[i] = np.nanargmin(errorarray[indsWith4[i]]) #Finds the ind of the second minimum
                    ptptSeps[i] = np.sqrt((xarray[indsWith4[i],indsOfMin[i]] - xarray[indsWith4[i],indsOfMin2[i]])**2. + (yarray[indsWith4[i],indsOfMin[i]] - yarray[indsWith4[i],indsOfMin2[i]])**2.)
                assert not np.all(np.isnan(errorarray[indsWith4[i]])), 'Looks like all points were too close for a planet'

        #If the points are all far apart, then
        twoIntSameY_x[indsWith4,1] = xarray[indsWith4,indsOfMin2]
        twoIntSameY_y[indsWith4,1] = yarray[indsWith4,indsOfMin2]

    #### For Inds With 3
    if len(indsWith3) > 0:
        #Select the two smallest seems to be the correct solution
        indsOfMin = np.nanargmin(errorarray[indsWith3],axis=1) #finds the ind of the minimum
        twoIntSameY_x[indsWith3,0] = xarray[indsWith3,indsOfMin]
        twoIntSameY_y[indsWith3,0] = yarray[indsWith3,indsOfMin]
        #Nan the absolute minimum in the error array and sep array
        errorarray[indsWith3,indsOfMin] = np.nan
        separray[indsWith3,indsOfMin] = np.nan
        #Find the ind of the second smallest
        indsOfMin2 = np.nanargmin(errorarray[indsWith3],axis=1) #Finds the ind of the second minimum
        #Calculate distance between x_min,y_min and x_2min,y_2min Ensure it is sufficiently large
        xarrayIndsOfMin = xarray[indsWith3,indsOfMin]
        yarrayIndsOfMin = yarray[indsWith3,indsOfMin]
        xarrayIndsOfMin2 = xarray[indsWith3,indsOfMin2]
        yarrayIndsOfMin2 = yarray[indsWith3,indsOfMin2]
        ptptSeps = np.sqrt((xarrayIndsOfMin - xarrayIndsOfMin2)**2. + (yarrayIndsOfMin - yarrayIndsOfMin2)**2.) #Calculate the distance between the two points
        assert np.all(ptptSeps > 1e-5), 'The points selected are too close to one another'
        #If the points are all far apart, then
        twoIntSameY_x[indsWith3,1] = xarray[indsWith3,indsOfMin2]
        twoIntSameY_y[indsWith3,1] = yarray[indsWith3,indsOfMin2]

    #### For Inds With 1
    if len(indsWith1) > 0:
        #The only solution I can come up with is to preemtively filter planets like this whenever they are encountered
        #Select the two smallest seems to be the correct solution. yrealAllRealInds[twoIntSameYInds[indsWith0]]
        #Find the relevant planet index with 
        indsOfMin = np.nanargmin(errorarray[indsWith1],axis=1) #finds the ind of the minimum
        twoIntSameY_x[indsWith1,0] = xarray[indsWith1,indsOfMin]
        twoIntSameY_y[indsWith1,0] = yarray[indsWith1,indsOfMin]
        #Nan the 4 absolute minimum solutions in the error array and sep array
        for i in np.arange(len(indsWith1)):
            indWith = indsWith1[i]
            indOfMin = indsOfMin[i]
            if indOfMin in [0,1,2,3]:
                for ind in [0,1,2,3]:
                    errorarray[indWith,ind] = np.nan
                    separray[indWith,ind] = np.nan
            if indOfMin in [4,5,6,7]:
                for ind in [4,5,6,7]:
                    errorarray[indWith,ind] = np.nan
                    separray[indWith,ind] = np.nan
            if indOfMin in [8,9,10,11]:
                for ind in [8,9,10,11]:
                    errorarray[indWith,ind] = np.nan
                    separray[indWith,ind] = np.nan
            if indOfMin in [12,13,14,15]:
                for ind in [12,13,14,15]:
                    errorarray[indWith,ind] = np.nan
                    separray[indWith,ind] = np.nan
        #Find the ind of the second smallest
        indsOfMin2 = np.nanargmin(errorarray[indsWith1],axis=1) #Finds the ind of the second minimum
        #Calculate distance between x_min,y_min and x_2min,y_2min Ensure it is sufficiently large
        xarrayIndsOfMin = xarray[indsWith1,indsOfMin]
        yarrayIndsOfMin = yarray[indsWith1,indsOfMin]
        xarrayIndsOfMin2 = xarray[indsWith1,indsOfMin2]
        yarrayIndsOfMin2 = yarray[indsWith1,indsOfMin2]
        ptptSeps = np.sqrt((xarrayIndsOfMin - xarrayIndsOfMin2)**2. + (yarrayIndsOfMin - yarrayIndsOfMin2)**2.) #Calculate the distance between the two points
        assert np.all(ptptSeps > 1e-5), 'The points selected are too close to one another'
        #If the points are all far apart, then
        twoIntSameY_x[indsWith1,1] = xarray[indsWith1,indsOfMin2]
        twoIntSameY_y[indsWith1,1] = yarray[indsWith1,indsOfMin2]

    #### For Inds With 0
    if len(indsWith0) > 0:
        if np.any(np.all(np.isnan(errorarray[indsWith0]),axis=1)):
            tInd = np.where(np.all(np.isnan(errorarray[indsWith0]),axis=1))[0]
            myInd2 = yrealAllRealInds[twoIntSameYInds[indsWith0[tInd]]]
            for myInd in myInd2:
                with open("/home/dean/Documents/exosims/twoDetMC/badKOE.py","a") as f:
                    f.write('#ellipseCircleIntersections 2' + '\n')
                    f.write('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]) + '\n')
                    f.write('badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)' + '\n')
            #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
        #assert not np.any(np.all(np.isnan(xarray[indsWith0]),axis=1)), 'Looks like one of the solutions is all NAN' #when this case was investigated, where xarray had all nans, it was caused by the quartic solver itself
        #The only solution I can come up with is to preemtively filter planets like this whenever they are encountered
        #Select the two smallest seems to be the correct solution. 
        # np.where(np.all(np.isnan(xarray[indsWith0]),axis=1))[0]
        # yrealAllRealInds[twoIntSameYInds[indsWith0]]

        #Checking to see if sep extrema are close to s_circle
        #minSep[yrealAllRealInds[twoIntSameYInds[indsWith0[1]]]]
        #maxSep[yrealAllRealInds[twoIntSameYInds[indsWith0[1]]]]
        #lminSep[twoIntSameYInds[indsWith0[1]]]
        #lmaxSep[twoIntSameYInds[indsWith0[1]]]

        #Find the relevant planet index with 
        indsOfMin = np.nanargmin(errorarray[indsWith0],axis=1) #finds the ind of the minimum
        twoIntSameY_x[indsWith0,0] = xarray[indsWith0,indsOfMin]
        twoIntSameY_y[indsWith0,0] = yarray[indsWith0,indsOfMin]
        #Nan the absolute minimum in the error array and sep array
        # errorarray[indsWith0,indsOfMin] = np.nan
        # separray[indsWith0,indsOfMin] = np.nan
        #Nan the 4 absolute minimum solutions in the error array and sep array
        for i in np.arange(len(indsWith0)):
            indWith = indsWith0[i]
            indOfMin = indsOfMin[i]
            if indOfMin in [0,1,2,3]:
                for ind in [0,1,2,3]:
                    errorarray[indWith,ind] = np.nan
                    separray[indWith,ind] = np.nan
            if indOfMin in [4,5,6,7]:
                for ind in [4,5,6,7]:
                    errorarray[indWith,ind] = np.nan
                    separray[indWith,ind] = np.nan
            if indOfMin in [8,9,10,11]:
                for ind in [8,9,10,11]:
                    errorarray[indWith,ind] = np.nan
                    separray[indWith,ind] = np.nan
            if indOfMin in [12,13,14,15]:
                for ind in [12,13,14,15]:
                    errorarray[indWith,ind] = np.nan
                    separray[indWith,ind] = np.nan
        #Find the ind of the second smallest
        indsOfMin2 = np.nanargmin(errorarray[indsWith0],axis=1) #Finds the ind of the second minimum
        #Calculate distance between x_min,y_min and x_2min,y_2min Ensure it is sufficiently large
        xarrayIndsOfMin = xarray[indsWith0,indsOfMin]
        yarrayIndsOfMin = yarray[indsWith0,indsOfMin]
        xarrayIndsOfMin2 = xarray[indsWith0,indsOfMin2]
        yarrayIndsOfMin2 = yarray[indsWith0,indsOfMin2]
        ptptSeps = np.sqrt((xarrayIndsOfMin - xarrayIndsOfMin2)**2. + (yarrayIndsOfMin - yarrayIndsOfMin2)**2.) #Calculate the distance between the two points
        #assert np.all(ptptSeps > 1e-5), 'The points selected are too close to one another'
        #Iterete over ptptSeps, nan the solution that is too close, select another
        if np.any(ptptSeps < 1e-5):
            for i in np.where(ptptSeps < 1e-5)[0]: #iterate over all the violating stars
                while ptptSeps[i] < 1e-5 and not np.all(np.isnan(errorarray[indsWith0[i]])):
                    errorarray[indsWith0[i],indsOfMin2[i]] = np.nan
                    indsOfMin2[i] = np.nanargmin(errorarray[indsWith0[i]]) #Finds the ind of the second minimum
                    ptptSeps[i] = np.sqrt((xarray[indsWith0[i],indsOfMin[i]] - xarray[indsWith0[i],indsOfMin2[i]])**2. + (yarray[indsWith0[i],indsOfMin[i]] - yarray[indsWith0[i],indsOfMin2[i]])**2.)
                assert not np.all(np.isnan(errorarray[indsWith0[i]])), 'Looks like all points were too close for a planet'

        #If the points are all far apart, then
        twoIntSameY_x[indsWith0,1] = xarray[indsWith0,indsOfMin2]
        twoIntSameY_y[indsWith0,1] = yarray[indsWith0,indsOfMin2]

    #### For Inds With 5
    if len(indsWith5) > 0:
        #Select the two smallest seems to be the correct solution
        indsOfMin = np.nanargmin(errorarray[indsWith5],axis=1) #finds the ind of the minimum
        twoIntSameY_x[indsWith5,0] = xarray[indsWith5,indsOfMin]
        twoIntSameY_y[indsWith5,0] = yarray[indsWith5,indsOfMin]
        #Nan the absolute minimum in the error array and sep array
        errorarray[indsWith5,indsOfMin] = np.nan
        separray[indsWith5,indsOfMin] = np.nan
        #Find the ind of the second smallest
        indsOfMin2 = np.nanargmin(errorarray[indsWith5],axis=1) #Finds the ind of the second minimum
        #Calculate distance between x_min,y_min and x_2min,y_2min Ensure it is sufficiently large
        xarrayIndsOfMin = xarray[indsWith5,indsOfMin]
        yarrayIndsOfMin = yarray[indsWith5,indsOfMin]
        xarrayIndsOfMin2 = xarray[indsWith5,indsOfMin2]
        yarrayIndsOfMin2 = yarray[indsWith5,indsOfMin2]
        ptptSeps = np.sqrt((xarrayIndsOfMin - xarrayIndsOfMin2)**2. + (yarrayIndsOfMin - yarrayIndsOfMin2)**2.) #Calculate the distance between the two points
        #assert np.all(ptptSeps > 1e-5), 'The points selected are too close to one another'
        #myInd = twoIntSameYInds[indsWith5[np.where(ptptSeps <= 1e-5)[0]]]
        #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
        #If the points are all far apart, then
        twoIntSameY_x[indsWith5,1] = xarray[indsWith5,indsOfMin2]
        twoIntSameY_y[indsWith5,1] = yarray[indsWith5,indsOfMin2]

    #### For Inds With 6
    if len(indsWith6) > 0:
        #Select the two smallest seems to be the correct solution
        indsOfMin = np.nanargmin(errorarray[indsWith6],axis=1) #finds the ind of the minimum
        twoIntSameY_x[indsWith6,0] = xarray[indsWith6,indsOfMin]
        twoIntSameY_y[indsWith6,0] = yarray[indsWith6,indsOfMin]
        #Nan the absolute minimum in the error array and sep array
        errorarray[indsWith6,indsOfMin] = np.nan
        separray[indsWith6,indsOfMin] = np.nan
        #Find the ind of the second smallest
        indsOfMin2 = np.nanargmin(errorarray[indsWith6],axis=1) #Finds the ind of the second minimum
        #Calculate distance between x_min,y_min and x_2min,y_2min Ensure it is sufficiently large
        xarrayIndsOfMin = xarray[indsWith6,indsOfMin]
        yarrayIndsOfMin = yarray[indsWith6,indsOfMin]
        xarrayIndsOfMin2 = xarray[indsWith6,indsOfMin2]
        yarrayIndsOfMin2 = yarray[indsWith6,indsOfMin2]
        ptptSeps = np.sqrt((xarrayIndsOfMin - xarrayIndsOfMin2)**2. + (yarrayIndsOfMin - yarrayIndsOfMin2)**2.) #Calculate the distance between the two points
        #assert np.all(ptptSeps > 1e-5), 'The points selected are too close to one another'
        #myInd = twoIntSameYInds[indsWith6[np.where(ptptSeps <= 1e-5)[0]]]
        #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
        #If the points are all far apart, then
        twoIntSameY_x[indsWith6,1] = xarray[indsWith6,indsOfMin2]
        twoIntSameY_y[indsWith6,1] = yarray[indsWith6,indsOfMin2]

    #### For Inds With 7
    if len(indsWith7) > 0:
        #Select the two smallest seems to be the correct solution
        indsOfMin = np.nanargmin(errorarray[indsWith7],axis=1) #finds the ind of the minimum
        twoIntSameY_x[indsWith7,0] = xarray[indsWith7,indsOfMin]
        twoIntSameY_y[indsWith7,0] = yarray[indsWith7,indsOfMin]
        #Nan the absolute minimum in the error array and sep array
        errorarray[indsWith7,indsOfMin] = np.nan
        separray[indsWith7,indsOfMin] = np.nan
        #Find the ind of the second smallest
        indsOfMin2 = np.nanargmin(errorarray[indsWith7],axis=1) #Finds the ind of the second minimum
        #Calculate distance between x_min,y_min and x_2min,y_2min Ensure it is sufficiently large
        xarrayIndsOfMin = xarray[indsWith7,indsOfMin]
        yarrayIndsOfMin = yarray[indsWith7,indsOfMin]
        xarrayIndsOfMin2 = xarray[indsWith7,indsOfMin2]
        yarrayIndsOfMin2 = yarray[indsWith7,indsOfMin2]
        ptptSeps = np.sqrt((xarrayIndsOfMin - xarrayIndsOfMin2)**2. + (yarrayIndsOfMin - yarrayIndsOfMin2)**2.) #Calculate the distance between the two points
        #assert np.all(ptptSeps > 1e-5), 'The points selected are too close to one another'
        #myInd = twoIntSameYInds[indsWith7[np.where(ptptSeps <= 1e-5)[0]]]
        #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
        #If the points are all far apart, then
        twoIntSameY_x[indsWith7,1] = xarray[indsWith7,indsOfMin2]
        twoIntSameY_y[indsWith7,1] = yarray[indsWith7,indsOfMin2]

    #### For Inds With 8
    if len(indsWith8) > 0:
        #Select the two smallest seems to be the correct solution
        indsOfMin = np.nanargmin(errorarray[indsWith8],axis=1) #finds the ind of the minimum
        twoIntSameY_x[indsWith8,0] = xarray[indsWith8,indsOfMin]
        twoIntSameY_y[indsWith8,0] = yarray[indsWith8,indsOfMin]
        #Nan the absolute minimum in the error array and sep array
        errorarray[indsWith8,indsOfMin] = np.nan
        separray[indsWith8,indsOfMin] = np.nan
        #Find the ind of the second smallest
        indsOfMin2 = np.nanargmin(errorarray[indsWith8],axis=1) #Finds the ind of the second minimum
        #Calculate distance between x_min,y_min and x_2min,y_2min Ensure it is sufficiently large
        xarrayIndsOfMin = xarray[indsWith8,indsOfMin]
        yarrayIndsOfMin = yarray[indsWith8,indsOfMin]
        xarrayIndsOfMin2 = xarray[indsWith8,indsOfMin2]
        yarrayIndsOfMin2 = yarray[indsWith8,indsOfMin2]
        ptptSeps = np.sqrt((xarrayIndsOfMin - xarrayIndsOfMin2)**2. + (yarrayIndsOfMin - yarrayIndsOfMin2)**2.) #Calculate the distance between the two points
        #assert np.all(ptptSeps > 1e-5), 'The points selected are too close to one another'
        #myInd = twoIntSameYInds[indsWith8[np.where(ptptSeps <= 1e-5)[0]]]
        #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
        #If the points are all far apart, then
        twoIntSameY_x[indsWith8,1] = xarray[indsWith8,indsOfMin2]
        twoIntSameY_y[indsWith8,1] = yarray[indsWith8,indsOfMin2]


    #Double Verification
    ptptSeps = np.sqrt((twoIntSameY_x[np.arange(twoIntSameY_x.shape[0]),1] - twoIntSameY_x[np.arange(twoIntSameY_x.shape[0]),0])**2. + (twoIntSameY_y[np.arange(twoIntSameY_x.shape[0]),1] - twoIntSameY_y[np.arange(twoIntSameY_x.shape[0]),0])**2.) #Calculate the distance between the two points
    if np.any(np.all(np.isnan(xarray),axis=1)):
        tInd = np.where(np.all(np.isnan(xarray),axis=1))[0]
        myInd = yrealAllRealInds[twoIntSameYInds[tInd]]
        with open("/home/dean/Documents/exosims/twoDetMC/badKOE.py","a") as f:
            f.write('#ellipseCircleIntersections 3' + '\n')
            f.write('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]) + '\n')
            f.write('badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)' + '\n')
    assert np.all(ptptSeps > 1e-5), 'The points selected are too close to one another' #If this is triggered, find the culprit ind, find the number of low error solutions it is producing, check and see if the errorarray has >2 low error solutions (this is likely the cause).
    #myInd = twoIntSameYInds[np.where(ptptSeps <= 1e-5)[0]]
    #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
    #Will likely need to add in a check that removes solutions by clusters. Currently we only nan indsOfMin, but we technically need to nan all indsOfMin in that set of 4 (since we did x0 y0,x0 -y0,-x0 y0,-x0 -y0)


    #### Two Intersection Points twoIntOppositeXInds
    twoIntOppositeX_x = np.zeros((len(twoIntOppositeXInds),2))
    twoIntOppositeX_y = np.zeros((len(twoIntOppositeXInds),2))
    if not len(twoIntOppositeXInds) == 0:
        if np.max(np.abs(np.imag(xreal2[yrealAllRealInds[twoIntOppositeXInds],0]))) >= 1e-3:
            myInds = np.where(np.abs(np.imag(xreal2[yrealAllRealInds[twoIntOppositeXInds],0])) >= 1e-3)[0]
            for myInd in myInds:
                with open("/home/dean/Documents/exosims/twoDetMC/badKOE.py","a") as f:
                    f.write('#ellipseCircleIntersections 4' + '\n')
                    f.write('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]) + '\n')
                    f.write('badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)' + '\n')
        assert np.max(np.abs(np.imag(xreal2[yrealAllRealInds[twoIntOppositeXInds],0]))) < 1e-3, '' #was 1e-12 but caused problems changed to 1e-7
        #Note this error just means a sep extrema is close to the s_circle
    else:
        print('twoIntOppositeXInds has length 0') #this is a debugging statement
    twoIntOppositeX_x[:,0] = np.real(xreal2[yrealAllRealInds[twoIntOppositeXInds],0])
    twoIntOppositeX_x[:,1] = np.real(xreal2[yrealAllRealInds[twoIntOppositeXInds],1])
    twoIntOppositeX_y = np.asarray([np.sqrt(b[yrealAllRealInds[twoIntOppositeXInds]]**2*(1-np.abs(twoIntOppositeX_x[:,0])**2/a[yrealAllRealInds[twoIntOppositeXInds]]**2)),\
            np.sqrt(b[yrealAllRealInds[twoIntOppositeXInds]]**2*(1-np.abs(twoIntOppositeX_x[:,1])**2/a[yrealAllRealInds[twoIntOppositeXInds]]**2))]).T
    #twoIntOppositeX_x = (twoIntOppositeX_x.T*(-2*bool1[yrealAllRealInds[twoIntOppositeXInds]]+1)).T
    twoIntOppositeX_y[:,1] = -twoIntOppositeX_y[:,1]
    #Adjust for Quadrant Star Belongs to
    twoIntOppositeX_x = (twoIntOppositeX_x.T*(2*bool1[yrealAllRealInds[twoIntOppositeXInds]]-1)).T
    twoIntOppositeX_y = (twoIntOppositeX_y.T*(2*bool2[yrealAllRealInds[twoIntOppositeXInds]]-1)).T
    ####

    #### ONLY 2 Real Inds (No Local Min or Local Max) ########################################
    sepsInsideInds = np.where((maxSep[yrealImagInds] >= s_circle[yrealImagInds]) & (s_circle[yrealImagInds] >= minSep[yrealImagInds]))[0] #inds where r is within the minimum and maximum separations
    only2RealInds = yrealImagInds[sepsInsideInds] #indicies of planets with only 2 real interesections
    #lets try usnig separation bounds
    #We will calculate separation at [0,+/-b] and [+/-a,0]
    sepbp = np.sqrt(mx[only2RealInds]**2+(b[only2RealInds]+my[only2RealInds])**2)
    sepbm = np.sqrt(mx[only2RealInds]**2+(b[only2RealInds]-my[only2RealInds])**2)
    sepap = np.sqrt((a[only2RealInds]+mx[only2RealInds])**2+my[only2RealInds]**2)
    sepam = np.sqrt((a[only2RealInds]-mx[only2RealInds])**2+my[only2RealInds]**2)

    #Types of Star Locations In Projected Ellipse
    typeInds0 = np.where(sepap < sepbp)[0]
    typeInds1 = np.where(sepbp < sepam)[0]
    typeInds2 = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam))[0]
    typeInds3 = np.where(sepam < sepbm)[0]
    #DELETE print(len(typeInds0))
    # print(len(typeInds1))
    # print(len(typeInds2))
    # print(len(typeInds3))

    xIntersectionsOnly2 = np.zeros((len(only2RealInds),2))
    yIntersectionsOnly2 = np.zeros((len(only2RealInds),2))
    #Separation Order For Each Location Type with Inds
    #Type0
    type0_0Inds = np.where((sepap < sepbp)*(s_circle[only2RealInds] < sepbm))[0]
    xIntersectionsOnly2[type0_0Inds] = np.real(xreal2[only2RealInds[type0_0Inds],0:2])
    yIntersectionsOnly2[type0_0Inds] = np.real(yreal2[only2RealInds[type0_0Inds],0:2])
    type0_1Inds = np.where((sepap < sepbp)*(sepbm < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepam))[0]
    xIntersectionsOnly2[type0_1Inds] = np.real(np.asarray([xreal2[only2RealInds[type0_1Inds],0],xreal2[only2RealInds[type0_1Inds],1]]).T) #-x is already in solution
    yIntersectionsOnly2[type0_1Inds] = np.real(yreal2[only2RealInds[type0_1Inds],0:2])
    type0_2Inds = np.where((sepap < sepbp)*(sepam < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepap))[0]
    xIntersectionsOnly2[type0_2Inds] = np.real(np.asarray([xreal2[only2RealInds[type0_2Inds],0],xreal2[only2RealInds[type0_2Inds],1]]).T) #-x is already in solution
    yIntersectionsOnly2[type0_2Inds] = np.real(np.asarray([yreal2[only2RealInds[type0_2Inds],0],-yreal2[only2RealInds[type0_2Inds],1]]).T)
    type0_3Inds = np.where((sepap < sepbp)*(sepap < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepbp))[0]
    xIntersectionsOnly2[type0_3Inds] = np.real(np.asarray([xreal2[only2RealInds[type0_3Inds],0],xreal2[only2RealInds[type0_3Inds],1]]).T) #-x is already in solution
    yIntersectionsOnly2[type0_3Inds] = np.real(np.asarray([-yreal2[only2RealInds[type0_3Inds],0],-yreal2[only2RealInds[type0_3Inds],1]]).T)
    type0_4Inds = np.where((sepap < sepbp)*(sepbp < s_circle[only2RealInds]))[0]
    xIntersectionsOnly2[type0_4Inds] = np.real(np.asarray([xreal2[only2RealInds[type0_4Inds],0],xreal2[only2RealInds[type0_4Inds],1]]).T) #-x is already in solution
    yIntersectionsOnly2[type0_4Inds] = np.real(np.asarray([-yreal2[only2RealInds[type0_4Inds],0],-yreal2[only2RealInds[type0_4Inds],1]]).T)
    #TODO FIX ALL THE STUFF HERE. FIRST FIND WHEN TYPE 1 Situations Occur (Should they have 4 real solutions always?)
    type1_0Inds = np.where((sepbp < sepam)*(s_circle[only2RealInds] < sepbm))[0]
    xIntersectionsOnly2[type1_0Inds] = np.real(xreal2[only2RealInds[type1_0Inds],0:2])
    yIntersectionsOnly2[type1_0Inds] = np.real(yreal2[only2RealInds[type1_0Inds],0:2])
    type1_1Inds = np.where((sepbp < sepam)*(sepbm < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepbp))[0]
    xIntersectionsOnly2[type1_1Inds] = np.real(np.asarray([-xreal2[only2RealInds[type1_1Inds],0],xreal2[only2RealInds[type1_1Inds],1]]).T)
    yIntersectionsOnly2[type1_1Inds] = np.real(np.asarray([yreal2[only2RealInds[type1_1Inds],0],yreal2[only2RealInds[type1_1Inds],1]]).T)
    type1_2Inds = np.where((sepbp < sepam)*(sepbp < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepam))[0]
    xIntersectionsOnly2[type1_2Inds] = np.real(np.asarray([-xreal2[only2RealInds[type1_2Inds],0],xreal2[only2RealInds[type1_2Inds],1]]).T)
    yIntersectionsOnly2[type1_2Inds] = np.real(np.asarray([yreal2[only2RealInds[type1_2Inds],0],yreal2[only2RealInds[type1_2Inds],1]]).T)
    type1_3Inds = np.where((sepbp < sepam)*(sepam < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepap))[0]
    xIntersectionsOnly2[type1_3Inds] = np.real(np.asarray([-xreal2[only2RealInds[type1_3Inds],0],-xreal2[only2RealInds[type1_3Inds],1]]).T)
    yIntersectionsOnly2[type1_3Inds] = np.real(np.asarray([yreal2[only2RealInds[type1_3Inds],0],-yreal2[only2RealInds[type1_3Inds],1]]).T)
    type1_4Inds = np.where((sepbp < sepam)*(sepap < s_circle[only2RealInds]))[0]
    xIntersectionsOnly2[type1_4Inds] = np.real(np.asarray([-xreal2[only2RealInds[type1_4Inds],0],-xreal2[only2RealInds[type1_4Inds],1]]).T)
    yIntersectionsOnly2[type1_4Inds] = np.real(np.asarray([-yreal2[only2RealInds[type1_4Inds],0],-yreal2[only2RealInds[type1_4Inds],1]]).T)
    #Type1 sepbm, sepbp, sepam, sepap #NOTE: Type1 should not be yrealImagInds
    #Type2 sepbm, sepam, sepbp, sepap
    type2_0Inds = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam)*(s_circle[only2RealInds] < sepbm))[0]
    xIntersectionsOnly2[type2_0Inds] = np.real(np.asarray([xreal2[only2RealInds[type2_0Inds],0],xreal2[only2RealInds[type2_0Inds],1]]).T)
    yIntersectionsOnly2[type2_0Inds] = np.real(np.asarray([yreal2[only2RealInds[type2_0Inds],0],yreal2[only2RealInds[type2_0Inds],1]]).T)
    type2_1Inds = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam)*(sepbm < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepam))[0]
    xIntersectionsOnly2[type2_1Inds] = np.real(np.asarray([xreal2[only2RealInds[type2_1Inds],0],xreal2[only2RealInds[type2_1Inds],1]]).T)#-x is already in solution
    yIntersectionsOnly2[type2_1Inds] = np.real(np.asarray([yreal2[only2RealInds[type2_1Inds],0],yreal2[only2RealInds[type2_1Inds],1]]).T)
    type2_2Inds = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam)*(sepam < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepbp))[0]
    xIntersectionsOnly2[type2_2Inds] = np.real(np.asarray([xreal2[only2RealInds[type2_2Inds],0],xreal2[only2RealInds[type2_2Inds],1]]).T)#-x is already in solution
    yIntersectionsOnly2[type2_2Inds] = np.real(np.asarray([yreal2[only2RealInds[type2_2Inds],0],-yreal2[only2RealInds[type2_2Inds],1]]).T)
    type2_3Inds = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam)*(sepbp < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepap))[0]
    xIntersectionsOnly2[type2_3Inds] = np.real(np.asarray([xreal2[only2RealInds[type2_3Inds],0],xreal2[only2RealInds[type2_3Inds],1]]).T)#-x is already in solution
    yIntersectionsOnly2[type2_3Inds] = np.real(np.asarray([yreal2[only2RealInds[type2_3Inds],0],-yreal2[only2RealInds[type2_3Inds],1]]).T)
    type2_4Inds = np.where((sepam < sepbp)*(sepbp < sepap)*(sepbm < sepam)*(sepap < s_circle[only2RealInds]))[0]
    xIntersectionsOnly2[type2_4Inds] = np.real(np.asarray([xreal2[only2RealInds[type2_4Inds],0],xreal2[only2RealInds[type2_4Inds],1]]).T)#-x is already in solution
    yIntersectionsOnly2[type2_4Inds] = np.real(np.asarray([-yreal2[only2RealInds[type2_4Inds],0],-yreal2[only2RealInds[type2_4Inds],1]]).T)
    #Type3 sepam, sepbm, sepbp, sepap
    type3_0Inds = np.where((sepam < sepbm)*(s_circle[only2RealInds] < sepam))[0]
    xIntersectionsOnly2[type3_0Inds] = np.real(np.asarray([xreal2[only2RealInds[type3_0Inds],0],xreal2[only2RealInds[type3_0Inds],1]]).T)
    yIntersectionsOnly2[type3_0Inds] = np.real(np.asarray([yreal2[only2RealInds[type3_0Inds],0],yreal2[only2RealInds[type3_0Inds],1]]).T)
    type3_1Inds = np.where((sepam < sepbm)*(sepam < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepbm))[0]
    xIntersectionsOnly2[type3_1Inds] = np.real(np.asarray([xreal2[only2RealInds[type3_1Inds],0],xreal2[only2RealInds[type3_1Inds],1]]).T)
    yIntersectionsOnly2[type3_1Inds] = np.real(np.asarray([yreal2[only2RealInds[type3_1Inds],0],-yreal2[only2RealInds[type3_1Inds],1]]).T)
    type3_2Inds = np.where((sepam < sepbm)*(sepbm < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepbp))[0]
    xIntersectionsOnly2[type3_2Inds] = np.real(np.asarray([xreal2[only2RealInds[type3_2Inds],0],xreal2[only2RealInds[type3_2Inds],1]]).T)#-x is already in solution
    yIntersectionsOnly2[type3_2Inds] = np.real(np.asarray([yreal2[only2RealInds[type3_2Inds],0],-yreal2[only2RealInds[type3_2Inds],1]]).T)
    type3_3Inds = np.where((sepam < sepbm)*(sepbp < s_circle[only2RealInds])*(s_circle[only2RealInds] < sepap))[0]
    xIntersectionsOnly2[type3_3Inds] = np.real(np.asarray([xreal2[only2RealInds[type3_3Inds],0],xreal2[only2RealInds[type3_3Inds],1]]).T)#-x is already in solution
    yIntersectionsOnly2[type3_3Inds] = np.real(np.asarray([yreal2[only2RealInds[type3_3Inds],0],-yreal2[only2RealInds[type3_3Inds],1]]).T)
    type3_4Inds = np.where((sepam < sepbm)*(sepap < s_circle[only2RealInds]))[0]
    xIntersectionsOnly2[type3_4Inds] = np.real(np.asarray([xreal2[only2RealInds[type3_4Inds],0],xreal2[only2RealInds[type3_4Inds],1]]).T)#-x is already in solution
    yIntersectionsOnly2[type3_4Inds] = np.real(np.asarray([-yreal2[only2RealInds[type3_4Inds],0],-yreal2[only2RealInds[type3_4Inds],1]]).T)
    #Adjust for Quadrant Star Belongs to
    xIntersectionsOnly2 = (xIntersectionsOnly2.T*(2*bool1[only2RealInds]-1)).T
    yIntersectionsOnly2 = (yIntersectionsOnly2.T*(2*bool2[only2RealInds]-1)).T
    ################################################
    allIndsUsed = np.concatenate((type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,
            type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds))


    return only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3,\
        fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
        twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, twoIntSameYInds,\
        type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,\
        type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,\
        allIndsUsed

#### Generalized Correct Ellipse Circle Intersection Fixer
def intersectionFixer_pm(x, y, sep_xlocs, sep_ylocs, afflictedIndsxy, rs):
    """ NOTE: where 1e-7 error floor comes from
    Uses x and y intersections from above to calculate intersection star distances,
    calculate error from the desired distance pick out indicies to fix error for (anything greater than 1e-7),
    calculate separations for different x,y scenarios (checking if the quadrant is wrong),
    Pick the lowest error solutions
    Args:
        x (numpy array):
            x components of host star in projected ellipse
        y (numpy array):
            y components of host star in projected ellipse
        sep_xlocs (numpy array):
            x components of proposed locations of ellipse circle intersections
        sep_ylocs (numpy array):
            y components of proposed locations of ellipse circle intersections
        afflictedIndsxy (numpy array):
            the inds to fix errors for
        rs (numpy array):
            the desired circle radius in AU with length n (number of planets)
    Returns:
        sep_xlocs (numpy array):
            quadrant adjusted x components of proposed locations of ellipse circle intersections
        sep_ylocs (numpy array):
            quadrant adjusted y components of proposed locations of ellipse circle intersections
    """
    # seps = np.sqrt((sep_xlocs-x[afflictedIndsxy])**2 + (sep_ylocs-y[afflictedIndsxy])**2) #calculate error for all TwoIntSameY
    # error = np.abs(np.sort(-np.abs(np.ones(len(seps)) - seps))) #calculate error for all TwoIntSameY
    # largeErrorInds = np.where(error > 1e-7)[0] #get inds of large errors
    # indsToFix = np.argsort(-np.abs(np.ones(len(seps)) - seps))[largeErrorInds] #inds of TwoIndSameY
    # seps_deciding = np.sqrt((sep_xlocs[indsToFix]-x[afflictedIndsxy[indsToFix]])**2 + (-sep_ylocs[indsToFix]-y[afflictedIndsxy[indsToFix]])**2) #calculate error for indsToFix
    # error_deciding = -np.abs(np.ones(len(seps_deciding)) - seps_deciding) #calculate errors for swapping y of the candidated to swap y for
    # indsToSwap = np.where(np.abs(error_deciding) < np.abs(error[indsToFix]))[0] #find where the errors produced by swapping y is lowered
    # sep_ylocs[indsToFix[indsToSwap]] = -sep_ylocs[indsToFix[indsToSwap]] #here we fix the y values where they should be fixed by swapping y values
    # seps = np.sqrt((sep_xlocs-x[afflictedIndsxy])**2 + (sep_ylocs-y[afflictedIndsxy])**2)
    # error = np.abs(np.sort(-np.abs(np.ones(len(seps)) - seps)))
    # indsToFix = np.argsort(-np.abs(np.ones(len(seps)) - seps))[np.where(error > 1e-7)[0]]

    seps = np.sqrt((sep_xlocs-x[afflictedIndsxy])**2 + (sep_ylocs-y[afflictedIndsxy])**2) #calculate error for all TwoIntSameY
    error = np.abs(rs - seps) #calculate error for all TwoIntSameY
    indsToFix = np.where(error > 1e-7)[0] #get inds of large errors
    #DELETEerror = np.abs(np.sort(-np.abs(rs[afflictedIndsxy] - seps))) #calculate error for all TwoIntSameY
    #DELETElargeErrorInds = np.where(error > 1e-7)[0] #get inds of large errors
    #DELETEindsToFix = np.argsort(-np.abs(rs[afflictedIndsxy] - seps))[largeErrorInds] #inds of TwoIndSameY
    if len(indsToFix) == 0: #There are no inds to fix
        return sep_xlocs, sep_ylocs

    seps_decidingpm = np.sqrt((sep_xlocs[indsToFix]-x[afflictedIndsxy[indsToFix]])**2 + (-sep_ylocs[indsToFix]-y[afflictedIndsxy[indsToFix]])**2) #calculate error for indsToFix
    seps_decidingmp = np.sqrt((-sep_xlocs[indsToFix]-x[afflictedIndsxy[indsToFix]])**2 + (sep_ylocs[indsToFix]-y[afflictedIndsxy[indsToFix]])**2) #calculate error for indsToFix
    seps_decidingmm = np.sqrt((-sep_xlocs[indsToFix]-x[afflictedIndsxy[indsToFix]])**2 + (-sep_ylocs[indsToFix]-y[afflictedIndsxy[indsToFix]])**2) #calculate error for indsToFix

    #error_decidingpm = -np.abs(np.ones(len(seps_decidingpm)) - seps_decidingpm) #calculate errors for swapping y of the candidated to swap y for
    #error_decidingmp = -np.abs(np.ones(len(seps_decidingmp)) - seps_decidingmp) #calculate errors for swapping y of the candidated to swap y for
    #error_decidingmm = -np.abs(np.ones(len(seps_decidingmm)) - seps_decidingmm) #calculate errors for swapping y of the candidated to swap y for
    #DELETEerror_deciding = np.array([error,-np.abs(np.ones(len(seps_decidingpm)) - seps_decidingpm),-np.abs(np.ones(len(seps_decidingmp)) - seps_decidingmp),-np.abs(np.ones(len(seps_decidingmm)) - seps_decidingmm)])
    error_deciding = np.stack((error[indsToFix],np.abs(rs[indsToFix] - seps_decidingpm),np.abs(rs[indsToFix] - seps_decidingmp),np.abs(rs[indsToFix] - seps_decidingmm)),axis=1)


    minErrorInds = np.argmin(error_deciding,axis=1)

    tmpxlocs = np.asarray([sep_xlocs,sep_xlocs,-sep_xlocs,-sep_xlocs]).T
    sep_xlocs[indsToFix] = tmpxlocs[indsToFix,minErrorInds]
    tmpylocs = np.asarray([sep_ylocs,-sep_ylocs,sep_ylocs,-sep_ylocs]).T
    sep_ylocs[indsToFix] = tmpylocs[indsToFix,minErrorInds]

    #indsToSwap = np.where(np.abs(error_deciding) < np.abs(error[indsToFix]))[0] #find where the errors produced by swapping y is lowered
    #sep_ylocs[indsToFix[indsToSwap]] = -sep_ylocs[indsToFix[indsToSwap]] #here we fix the y values where they should be fixed by swapping y values
    #seps = np.sqrt((sep_xlocs-x[afflictedIndsxy])**2 + (sep_ylocs-y[afflictedIndsxy])**2)
    #error = np.abs(np.sort(-np.abs(np.ones(len(seps)) - seps)))
    #indsToFix = np.argsort(-np.abs(np.ones(len(seps)) - seps))[np.where(error > 1e-7)[0]]

    return sep_xlocs, sep_ylocs

#### Calculate True Anomalies of Points
def trueAnomaliesOfPoints(minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
    fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr,\
    yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, W, w, inc):
    """ Given the location of the points of the intersections between the circle and the derotated ellipse,
    this method calculates the true anomalies of these intersections
    Args:
        minSepPoints_x_dr (numpy array):
            derotated minSepPoints_x
        minSepPoints_y_dr (numpy array):
            derotated minSepPoints_y
        maxSepPoints_x_dr (numpy array):
            derotated maxSepPoints_x
        maxSepPoints_y_dr (numpy array):
            derotated maxSepPoints_y
        lminSepPoints_x_dr (numpy array):
            derotated lminSepPoints_x
        lminSepPoints_y_dr (numpy array):
            derotated lminSepPoints_y
        lmaxSepPoints_x_dr (numpy array):
            derotated lmaxSepPoints_x
        lmaxSepPoints_y_dr (numpy array):
            derotated lmaxSepPoints_y
        fourInt_x_dr (numpy array):
            derotated fourInt_x
        fourInt_y_dr (numpy array):
            derotated fourInt_y
        twoIntSameY_x_dr (numpy array):
            derotated twoIntSameY_x
        twoIntSameY_y_dr (numpy array):
            derotated twoIntSameY_y
        twoIntOppositeX_x_dr (numpy array):
            derotated twoIntOppositeX_x
        twoIntOppositeX_y_dr (numpy array):
            derotated twoIntOppositeX_y
        xIntersectionsOnly2_dr (numpy array):
            derotated xIntersectionsOnly2
        yIntersectionsOnly2_dr (numpy array):
            derotated yIntersectionsOnly2
        yrealAllRealInds (numpy array):
            an array of integers acting as indicies of planets which have min, max, local min, local max        fourIntInds (numpy array):
        twoIntSameYInds (numpy array):
            indicies of yrealAllRealInds which should have 2 intersections on the 
            same Y side of the projected ellipse as the star        twoIntOppositeXInds (numpy array):
        only2RealInds (numpy array):
            indicies where there can only ever by 2 circle-ellipse intersections
        W (numpy array): 
            Longitude of the ascending node of the body
        w (numpy array): 
            argument of periapsis of the body
        inc (numpy array): 
            inclination of the body's orbit
    Returns:
        nu_minSepPoints (numpy array):
            true anomaly of the minimum separation points
        nu_maxSepPoints (numpy array):
            true anomaly of the maximum separation points
        nu_lminSepPoints (numpy array):
            true anomaly of the local minimum separation points
        nu_lmaxSepPoints (numpy array):
            true anomaly of the maximum separation points
        nu_fourInt (numpy array):
            true anomalies of the four intersection point cases
        nu_twoIntSameY (numpy array):
            true anomalies of the two intersection points on same y-side as star cases
        nu_twoIntOppositeX (numpy array):
            true anomalies of the two intersection point on the opposite x-side of the star cases
        nu_IntersectionsOnly2 (numpy array):
            true anomalies of the only two intersection point cases
    """
    nu_minSepPoints = trueAnomalyFromXY(minSepPoints_x_dr, minSepPoints_y_dr,W,w,inc)
    nu_maxSepPoints = trueAnomalyFromXY(maxSepPoints_x_dr, maxSepPoints_y_dr,W,w,inc)
    nu_lminSepPoints = trueAnomalyFromXY(lminSepPoints_x_dr, lminSepPoints_y_dr,W[yrealAllRealInds],w[yrealAllRealInds],inc[yrealAllRealInds])
    nu_lmaxSepPoints = trueAnomalyFromXY(lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,W[yrealAllRealInds],w[yrealAllRealInds],inc[yrealAllRealInds])
    nu_fourInt = np.zeros(fourInt_x_dr.shape)
    nu_fourInt[:,0] = trueAnomalyFromXY(fourInt_x_dr[:,0], fourInt_y_dr[:,0],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_fourInt[:,1] = trueAnomalyFromXY(fourInt_x_dr[:,1], fourInt_y_dr[:,1],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_fourInt[:,2] = trueAnomalyFromXY(fourInt_x_dr[:,2], fourInt_y_dr[:,2],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_fourInt[:,3] = trueAnomalyFromXY(fourInt_x_dr[:,3], fourInt_y_dr[:,3],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_twoIntSameY = np.zeros(twoIntSameY_x_dr.shape)
    nu_twoIntSameY[:,0] = trueAnomalyFromXY(twoIntSameY_x_dr[:,0], twoIntSameY_y_dr[:,0],W[yrealAllRealInds[twoIntSameYInds]],w[yrealAllRealInds[twoIntSameYInds]],inc[yrealAllRealInds[twoIntSameYInds]])
    nu_twoIntSameY[:,1] = trueAnomalyFromXY(twoIntSameY_x_dr[:,1], twoIntSameY_y_dr[:,1],W[yrealAllRealInds[twoIntSameYInds]],w[yrealAllRealInds[twoIntSameYInds]],inc[yrealAllRealInds[twoIntSameYInds]])
    nu_twoIntOppositeX = np.zeros(twoIntOppositeX_x_dr.shape)
    nu_twoIntOppositeX[:,0] = trueAnomalyFromXY(twoIntOppositeX_x_dr[:,0], twoIntOppositeX_y_dr[:,0],W[yrealAllRealInds[twoIntOppositeXInds]],w[yrealAllRealInds[twoIntOppositeXInds]],inc[yrealAllRealInds[twoIntOppositeXInds]])
    nu_twoIntOppositeX[:,1] = trueAnomalyFromXY(twoIntOppositeX_x_dr[:,1], twoIntOppositeX_y_dr[:,1],W[yrealAllRealInds[twoIntOppositeXInds]],w[yrealAllRealInds[twoIntOppositeXInds]],inc[yrealAllRealInds[twoIntOppositeXInds]])
    nu_IntersectionsOnly2 = np.zeros(xIntersectionsOnly2_dr.shape)
    nu_IntersectionsOnly2[:,0] = trueAnomalyFromXY(xIntersectionsOnly2_dr[:,0], yIntersectionsOnly2_dr[:,0],W[only2RealInds],w[only2RealInds],inc[only2RealInds])
    nu_IntersectionsOnly2[:,1] = trueAnomalyFromXY(xIntersectionsOnly2_dr[:,1], yIntersectionsOnly2_dr[:,1],W[only2RealInds],w[only2RealInds],inc[only2RealInds])
    return nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2

def trueAnomaliesOfIntPoints(fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr,\
    yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, W, w, inc):
    """ Given the location of the points of the intersections between the circle and the derotated ellipse,
    this method calculates the true anomalies of these intersections
    Args:
        fourInt_x_dr (numpy array):
            derotated fourInt_x
        fourInt_y_dr (numpy array):
            derotated fourInt_y
        twoIntSameY_x_dr (numpy array):
            derotated twoIntSameY_x
        twoIntSameY_y_dr (numpy array):
            derotated twoIntSameY_y
        twoIntOppositeX_x_dr (numpy array):
            derotated twoIntOppositeX_x
        twoIntOppositeX_y_dr (numpy array):
            derotated twoIntOppositeX_y
        xIntersectionsOnly2_dr (numpy array):
            derotated xIntersectionsOnly2
        yIntersectionsOnly2_dr (numpy array):
            derotated yIntersectionsOnly2
        yrealAllRealInds (numpy array):
            an array of integers acting as indicies of planets which have min, max, local min, local max        fourIntInds (numpy array):
        twoIntSameYInds (numpy array):
            indicies of yrealAllRealInds which should have 2 intersections on the 
            same Y side of the projected ellipse as the star        twoIntOppositeXInds (numpy array):
        only2RealInds (numpy array):
            indicies where there can only ever by 2 circle-ellipse intersections
        W (numpy array): 
            Longitude of the ascending node of the body
        w (numpy array): 
            argument of periapsis of the body
        inc (numpy array): 
            inclination of the body's orbit
    Returns:
        nu_fourInt (numpy array):
            true anomalies of the four intersection point cases
        nu_twoIntSameY (numpy array):
            true anomalies of the two intersection points on same y-side as star cases
        nu_twoIntOppositeX (numpy array):
            true anomalies of the two intersection point on the opposite x-side of the star cases
        nu_IntersectionsOnly2 (numpy array):
            true anomalies of the only two intersection point cases
    """
    nu_fourInt = np.zeros(fourInt_x_dr.shape)
    nu_fourInt[:,0] = trueAnomalyFromXY(fourInt_x_dr[:,0], fourInt_y_dr[:,0],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_fourInt[:,1] = trueAnomalyFromXY(fourInt_x_dr[:,1], fourInt_y_dr[:,1],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_fourInt[:,2] = trueAnomalyFromXY(fourInt_x_dr[:,2], fourInt_y_dr[:,2],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_fourInt[:,3] = trueAnomalyFromXY(fourInt_x_dr[:,3], fourInt_y_dr[:,3],W[yrealAllRealInds[fourIntInds]],w[yrealAllRealInds[fourIntInds]],inc[yrealAllRealInds[fourIntInds]])
    nu_twoIntSameY = np.zeros(twoIntSameY_x_dr.shape)
    nu_twoIntSameY[:,0] = trueAnomalyFromXY(twoIntSameY_x_dr[:,0], twoIntSameY_y_dr[:,0],W[yrealAllRealInds[twoIntSameYInds]],w[yrealAllRealInds[twoIntSameYInds]],inc[yrealAllRealInds[twoIntSameYInds]])
    nu_twoIntSameY[:,1] = trueAnomalyFromXY(twoIntSameY_x_dr[:,1], twoIntSameY_y_dr[:,1],W[yrealAllRealInds[twoIntSameYInds]],w[yrealAllRealInds[twoIntSameYInds]],inc[yrealAllRealInds[twoIntSameYInds]])
    nu_twoIntOppositeX = np.zeros(twoIntOppositeX_x_dr.shape)
    nu_twoIntOppositeX[:,0] = trueAnomalyFromXY(twoIntOppositeX_x_dr[:,0], twoIntOppositeX_y_dr[:,0],W[yrealAllRealInds[twoIntOppositeXInds]],w[yrealAllRealInds[twoIntOppositeXInds]],inc[yrealAllRealInds[twoIntOppositeXInds]])
    nu_twoIntOppositeX[:,1] = trueAnomalyFromXY(twoIntOppositeX_x_dr[:,1], twoIntOppositeX_y_dr[:,1],W[yrealAllRealInds[twoIntOppositeXInds]],w[yrealAllRealInds[twoIntOppositeXInds]],inc[yrealAllRealInds[twoIntOppositeXInds]])
    nu_IntersectionsOnly2 = np.zeros(xIntersectionsOnly2_dr.shape)
    nu_IntersectionsOnly2[:,0] = trueAnomalyFromXY(xIntersectionsOnly2_dr[:,0], yIntersectionsOnly2_dr[:,0],W[only2RealInds],w[only2RealInds],inc[only2RealInds])
    nu_IntersectionsOnly2[:,1] = trueAnomalyFromXY(xIntersectionsOnly2_dr[:,1], yIntersectionsOnly2_dr[:,1],W[only2RealInds],w[only2RealInds],inc[only2RealInds])
    return nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2

def trueAnomaliesOfSepPoints(minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
    yrealAllRealInds, W, w, inc):
    """ Given the location of the points of the separation extrema between the star and the derotated ellipse,
    this method calculates the true anomalies of these extrema
    Args:
        minSepPoints_x_dr (numpy array):
            derotated minSepPoints_x
        minSepPoints_y_dr (numpy array):
            derotated minSepPoints_y
        maxSepPoints_x_dr (numpy array):
            derotated maxSepPoints_x
        maxSepPoints_y_dr (numpy array):
            derotated maxSepPoints_y
        lminSepPoints_x_dr (numpy array):
            derotated lminSepPoints_x
        lminSepPoints_y_dr (numpy array):
            derotated lminSepPoints_y
        lmaxSepPoints_x_dr (numpy array):
            derotated lmaxSepPoints_x
        lmaxSepPoints_y_dr (numpy array):
            derotated lmaxSepPoints_y
        yrealAllRealInds (numpy array):
            an array of integers acting as indicies of planets which have min, max, local min, local max        fourIntInds (numpy array):
        W (numpy array): 
            Longitude of the ascending node of the body
        w (numpy array): 
            argument of periapsis of the body
        inc (numpy array): 
            inclination of the body's orbit
    Returns:
        nu_minSepPoints (numpy array):
            true anomaly of the minimum separation points
        nu_maxSepPoints (numpy array):
            true anomaly of the maximum separation points
        nu_lminSepPoints (numpy array):
            true anomaly of the local minimum separation points
        nu_lmaxSepPoints (numpy array):
            true anomaly of the maximum separation points
    """
    nu_minSepPoints = trueAnomalyFromXY(minSepPoints_x_dr, minSepPoints_y_dr,W,w,inc)
    nu_maxSepPoints = trueAnomalyFromXY(maxSepPoints_x_dr, maxSepPoints_y_dr,W,w,inc)
    nu_lminSepPoints = trueAnomalyFromXY(lminSepPoints_x_dr, lminSepPoints_y_dr,W[yrealAllRealInds],w[yrealAllRealInds],inc[yrealAllRealInds])
    nu_lmaxSepPoints = trueAnomalyFromXY(lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,W[yrealAllRealInds],w[yrealAllRealInds],inc[yrealAllRealInds])
    return nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints

#### Nu corections for extrema
def nuCorrections_extrema(sma,e,W,w,inc,nus,mainInds,seps):
    """ A method for correcting the nus of the extrema points. The input nus are calculated with arctan meaning it could be nu or nu+pi
    Args:
        sma (numpy array):
            semi-major axis
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the ascending nodes
        w (numpy array):
            Argument of periapsis
        inc (numpy array):
            inclination
        nus (numpy array):
            the true anomalies 
        mainInds (numpy array):
            the set of all inds to consider for corrections
        seps (numpy array):
            the expected separations of all these points
    Returns:
        nus (numpy array):
            the corrected true anomalies
    """
    #Calculates the planet-star separations at the input nus
    r_extrema = xyz_3Dellipse(sma[mainInds],e[mainInds],W[mainInds],w[mainInds],inc[mainInds],nus)
    s_extrema = np.sqrt(r_extrema[0,0]**2 + r_extrema[1,0]**2)
    #Calculates the errors for the input nus
    error0 = np.abs(seps - s_extrema)
    #Sets up nus for off by pi error (nu+pi)
    nus_extrema_ppi = nus + np.pi
    #Calculates the planet-star separations for nus+pi
    r_extrema_ppi = xyz_3Dellipse(sma[mainInds],e[mainInds],W[mainInds],w[mainInds],inc[mainInds],nus_extrema_ppi)
    s_extrema_ppi = np.sqrt(r_extrema_ppi[0,0]**2 + r_extrema_ppi[1,0]**2)
    #Calculates the error for the adjusted nus
    error1 = np.abs(seps - s_extrema_ppi)

    #Figure out which inds to keep the adjusted nus for
    error_deciding = np.stack((error0,error1),axis=1)
    minErrorInds = np.argmin(error_deciding,axis=1)

    tmpnus = np.asarray([nus,nus_extrema_ppi]).T
    nus = tmpnus[np.arange(len(nus)),minErrorInds]
    #KEEPerror = error_deciding[np.arange(len(nus)),minErrorInds]
    nus = np.mod(nus,2.*np.pi)
    return nus#KEEP, error

#### Correcting nu for ellipse-circle intersections
def nuCorrections_int(sma,e,W,w,inc,r,nus,mainInds,subInds):
    """ A method for correcting the nus of the intersection points. The input nus are calculated with arctan meaning it could be nu or nu+pi
    Args:
        sma (numpy array):
            semi-major axis
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the ascending nodes
        w (numpy array):
            Argument of periapsis
        inc (numpy array):
            inclination
        nus (numpy array):
            the true anomalies of intersections
        mainInds (numpy array):
            the set of all inds to consider for corrections (the subset of all planet inds i.e. yrealAllRealInds)
        subInds (numpy array):
            the subset of mianInds to consider (i.e. fourIntInds)
    Returns:
        nus (numpy array):
            the corrected true anomalies
    """
    r_fourInt0 = xyz_3Dellipse(sma[mainInds[subInds]],e[mainInds[subInds]],W[mainInds[subInds]],w[mainInds[subInds]],inc[mainInds[subInds]],nus)
    tmp_fourInt0Seps = np.sqrt(r_fourInt0[0,0]**2 + r_fourInt0[1,0]**2)
    wrong_fourInt0Inds = np.where(np.abs(r[mainInds[subInds]] - tmp_fourInt0Seps) > 1e-6)[0]#1e-6)[0]
    if len(wrong_fourInt0Inds) > 0:
        nus[wrong_fourInt0Inds] = nus[wrong_fourInt0Inds] + np.pi
        r_fourInt0 = xyz_3Dellipse(sma[mainInds[subInds]],e[mainInds[subInds]],W[mainInds[subInds]],w[mainInds[subInds]],inc[mainInds[subInds]],nus)
        tmp_fourInt0Seps = np.sqrt(r_fourInt0[0,0]**2 + r_fourInt0[1,0]**2)
        wrong_fourInt0Inds = np.where(np.abs(r[mainInds[subInds]] - tmp_fourInt0Seps) > 1e-6)[0]
    #DELETEprint(len(wrong_fourInt0Inds))
    #print(wrong_fourInt0Inds)
    if len(wrong_fourInt0Inds) > 0: #now choose the smaller error of the two
        r_fourInt0_2 = xyz_3Dellipse(sma[mainInds[subInds[wrong_fourInt0Inds]]],e[mainInds[subInds[wrong_fourInt0Inds]]],W[mainInds[subInds[wrong_fourInt0Inds]]],w[mainInds[subInds[wrong_fourInt0Inds]]],inc[mainInds[subInds[wrong_fourInt0Inds]]],nus[wrong_fourInt0Inds])
        tmp_fourInt0Seps_2 = np.sqrt(r_fourInt0_2[0,0]**2 + r_fourInt0_2[1,0]**2)
        r_fourInt0_3 = xyz_3Dellipse(sma[mainInds[subInds[wrong_fourInt0Inds]]],e[mainInds[subInds[wrong_fourInt0Inds]]],W[mainInds[subInds[wrong_fourInt0Inds]]],w[mainInds[subInds[wrong_fourInt0Inds]]],inc[mainInds[subInds[wrong_fourInt0Inds]]],nus[wrong_fourInt0Inds]+np.pi)
        tmp_fourInt0Seps_3 = np.sqrt(r_fourInt0_3[0,0]**2 + r_fourInt0_3[1,0]**2)
        indsToSwap = np.where(np.abs(r[mainInds[subInds[wrong_fourInt0Inds]]] - tmp_fourInt0Seps_2) > np.abs(r[mainInds[subInds[wrong_fourInt0Inds]]] - tmp_fourInt0Seps_3))[0]
        if len(indsToSwap) > 0:
            nus[wrong_fourInt0Inds[indsToSwap]] = nus[wrong_fourInt0Inds[indsToSwap]] + np.pi
    r_fourInt0 = xyz_3Dellipse(sma[mainInds[subInds]],e[mainInds[subInds]],W[mainInds[subInds]],w[mainInds[subInds]],inc[mainInds[subInds]],nus)
    tmp_fourInt0Seps = np.sqrt(r_fourInt0[0,0]**2 + r_fourInt0[1,0]**2)
    errors = np.abs(r[mainInds[subInds]] - tmp_fourInt0Seps)
    #DELETE?maxError_fourInt0 = np.max(errors) #Was causing an error when subInds is empty, specifically when fourIntInds is empty and passed in
    #print(maxError_fourInt0)
    nus = np.mod(nus,2.*np.pi)
    return nus, errors

def rerotateExtremaAndIntersectionPoints(minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
    fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    Phi, Op, yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds):
    """ DEPRECATED. Replaced piecemeal by rerotateExtremaPoints and rerotateIntersectionPoints. Rotate the intersection points from (the projected ellipse centered at the origin and x-axis aligned with semi-major axis) to the original projected ellipse
    Args:
        minSepPoints_x (numpy array):
            the first quadrant x-coordinates of the minimum separations (with length n)
        minSepPoints_y (numpy array):
            the first quadrant y-coordinates of the minimum separations (with length n)
        maxSepPoints_x (numpy array):
            the first quadrant x-coordinates of the maximum separations (with length n)
        maxSepPoints_y (numpy array):
            the first quadrant y-coordinates of the maximum separations (with length n)
        lminSepPoints_x (numpy array):
            the first quadrant x-coordinates of the local minimum separations (with same length as yrealImagInds)
        lminSepPoints_y (numpy array):
            the first quadrant y-coordinates of the local minimum separations (with same length as yrealImagInds)
        lmaxSepPoints_x (numpy array):
            the first quadrant x-coordinates of the local maximum separations (with same length as yrealImagInds)
        lmaxSepPoints_y (numpy array):
            the first quadrant y-coordinates of the local maximum separations (with same length as yrealImagInds)
        fourInt_x (numpy array):
            x coordinates of fourIntInds
        fourInt_y (numpy array):
            y coordinates of fourIntInds
        twoIntSameY_x (numpy array):
            x components of intersections which must occur on same Y side of the projected ellipse as the star
        twoIntSameY_y (numpy array):
            y components of intersections which must occur on same Y side of the projected ellipse as the star
        twoIntOppositeX_x (numpy array):
            x components of intersections which must occur on opposite X side of the projected ellipse as the star
        twoIntOppositeX_y (numpy array):
            y components of intersections which must occur on opposite X side of the projected ellipse as the star
        xIntersectionsOnly2 (numpy array):
            x components of intersections where there must be only 2 intersections
        yIntersectionsOnly2 (numpy array):
            y components of intersections where there must be only 2 intersections
        phi (numpy array):
            angle from X-axis to semi-minor axis of projected ellipse 
        Op (numpy array):
            the geometric center of the projected ellipse
        yrealAllRealInds (numpy array):
            an array of integers acting as indicies of planets which have min, max, local min, local max
        fourIntInds (numpy array):
            indicies of yrealAllRealInds which should have 4 intersections
        twoIntSameYInds (numpy array):
            indicies of yrealAllRealInds which should have 2 intersections on the 
            same Y side of the projected ellipse as the star
        twoIntOppositeXInds (numpy array):
            indicies of yrealAllRealInds which should have 2 intersections on the 
            opposite X side of the projected ellipse as the star
        only2RealInds (numpy array):
            indicies where there can only ever by 2 circle-ellipse intersections
    Returns:
        minSepPoints_x_dr (numpy array):
            derotated minSepPoints_x
        minSepPoints_y_dr (numpy array):
            derotated minSepPoints_y
        maxSepPoints_x_dr (numpy array):
            derotated maxSepPoints_x
        maxSepPoints_y_dr (numpy array):
            derotated maxSepPoints_y
        lminSepPoints_x_dr (numpy array):
            derotated lminSepPoints_x
        lminSepPoints_y_dr (numpy array):
            derotated lminSepPoints_y
        lmaxSepPoints_x_dr (numpy array):
            derotated lmaxSepPoints_x
        lmaxSepPoints_y_dr (numpy array):
            derotated lmaxSepPoints_y
        fourInt_x_dr (numpy array):
            derotated fourInt_x
        fourInt_y_dr (numpy array):
            derotated fourInt_y
        twoIntSameY_x_dr (numpy array):
            derotated twoIntSameY_x
        twoIntSameY_y_dr (numpy array):
            derotated twoIntSameY_y
        twoIntOppositeX_x_dr (numpy array):
            derotated twoIntOppositeX_x
        twoIntOppositeX_y_dr (numpy array):
            derotated twoIntOppositeX_y
        xIntersectionsOnly2_dr (numpy array):
            derotated xIntersectionsOnly2
        yIntersectionsOnly2_dr (numpy array):
            derotated yIntersectionsOnly2
    """
    minSepPoints_x_dr = np.zeros(len(minSepPoints_x))
    minSepPoints_y_dr = np.zeros(len(minSepPoints_y))
    maxSepPoints_x_dr = np.zeros(len(maxSepPoints_x))
    maxSepPoints_y_dr = np.zeros(len(maxSepPoints_y))
    lminSepPoints_x_dr = np.zeros(len(lminSepPoints_x))
    lminSepPoints_y_dr = np.zeros(len(lminSepPoints_y))
    lmaxSepPoints_x_dr = np.zeros(len(lmaxSepPoints_x))
    lmaxSepPoints_y_dr = np.zeros(len(lmaxSepPoints_y))
    fourInt_x_dr = np.zeros((len(fourInt_x),4))
    fourInt_y_dr = np.zeros((len(fourInt_y),4))
    twoIntSameY_x_dr = np.zeros((len(twoIntSameY_x),2))
    twoIntSameY_y_dr = np.zeros((len(twoIntSameY_y),2))
    twoIntOppositeX_x_dr = np.zeros((len(twoIntOppositeX_x),2))
    twoIntOppositeX_y_dr = np.zeros((len(twoIntOppositeX_y),2))
    xIntersectionsOnly2_dr = np.zeros((len(xIntersectionsOnly2),2))
    yIntersectionsOnly2_dr = np.zeros((len(yIntersectionsOnly2),2))


    minSepPoints_x_dr, minSepPoints_y_dr = rerotateEllipsePoints(minSepPoints_x, minSepPoints_y,Phi,Op[0],Op[1])
    maxSepPoints_x_dr, maxSepPoints_y_dr = rerotateEllipsePoints(maxSepPoints_x, maxSepPoints_y,Phi,Op[0],Op[1])
    lminSepPoints_x_dr, lminSepPoints_y_dr = rerotateEllipsePoints(lminSepPoints_x, lminSepPoints_y,Phi[yrealAllRealInds],Op[0][yrealAllRealInds],Op[1][yrealAllRealInds])
    lmaxSepPoints_x_dr, lmaxSepPoints_y_dr = rerotateEllipsePoints(lmaxSepPoints_x, lmaxSepPoints_y,Phi[yrealAllRealInds],Op[0][yrealAllRealInds],Op[1][yrealAllRealInds])
    fourInt_x_dr[:,0], fourInt_y_dr[:,0] = rerotateEllipsePoints(fourInt_x[:,0], fourInt_y[:,0],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    fourInt_x_dr[:,1], fourInt_y_dr[:,1] = rerotateEllipsePoints(fourInt_x[:,1], fourInt_y[:,1],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    fourInt_x_dr[:,2], fourInt_y_dr[:,2] = rerotateEllipsePoints(fourInt_x[:,2], fourInt_y[:,2],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    fourInt_x_dr[:,3], fourInt_y_dr[:,3] = rerotateEllipsePoints(fourInt_x[:,3], fourInt_y[:,3],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    twoIntSameY_x_dr[:,0], twoIntSameY_y_dr[:,0] = rerotateEllipsePoints(twoIntSameY_x[:,0], twoIntSameY_y[:,0],Phi[yrealAllRealInds[twoIntSameYInds]],Op[0][yrealAllRealInds[twoIntSameYInds]],Op[1][yrealAllRealInds[twoIntSameYInds]])
    twoIntSameY_x_dr[:,1], twoIntSameY_y_dr[:,1] = rerotateEllipsePoints(twoIntSameY_x[:,1], twoIntSameY_y[:,1],Phi[yrealAllRealInds[twoIntSameYInds]],Op[0][yrealAllRealInds[twoIntSameYInds]],Op[1][yrealAllRealInds[twoIntSameYInds]])
    twoIntOppositeX_x_dr[:,0], twoIntOppositeX_y_dr[:,0] = rerotateEllipsePoints(twoIntOppositeX_x[:,0], twoIntOppositeX_y[:,0],Phi[yrealAllRealInds[twoIntOppositeXInds]],Op[0][yrealAllRealInds[twoIntOppositeXInds]],Op[1][yrealAllRealInds[twoIntOppositeXInds]])
    twoIntOppositeX_x_dr[:,1], twoIntOppositeX_y_dr[:,1] = rerotateEllipsePoints(twoIntOppositeX_x[:,1], twoIntOppositeX_y[:,1],Phi[yrealAllRealInds[twoIntOppositeXInds]],Op[0][yrealAllRealInds[twoIntOppositeXInds]],Op[1][yrealAllRealInds[twoIntOppositeXInds]])
    xIntersectionsOnly2_dr[:,0], yIntersectionsOnly2_dr[:,0] = rerotateEllipsePoints(xIntersectionsOnly2[:,0], yIntersectionsOnly2[:,0],Phi[only2RealInds],Op[0][only2RealInds],Op[1][only2RealInds])
    xIntersectionsOnly2_dr[:,1], yIntersectionsOnly2_dr[:,1] = rerotateEllipsePoints(xIntersectionsOnly2[:,1], yIntersectionsOnly2[:,1],Phi[only2RealInds],Op[0][only2RealInds],Op[1][only2RealInds])
    return minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
            fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr

def rerotateExtremaPoints(minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y,\
    lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
    Phi, Op, yrealAllRealInds):
    """ Rotate the extrema points from (the projected ellipse centered at the origin
    and x-axis aligned with semi-major axis) to the original projected ellipse
    Args:
        minSepPoints_x (numpy array):
            the first quadrant x-coordinates of the minimum separations (with length n)
        minSepPoints_y (numpy array):
            the first quadrant y-coordinates of the minimum separations (with length n)
        maxSepPoints_x (numpy array):
            the first quadrant x-coordinates of the maximum separations (with length n)
        maxSepPoints_y (numpy array):
            the first quadrant y-coordinates of the maximum separations (with length n)
        lminSepPoints_x (numpy array):
            the first quadrant x-coordinates of the local minimum separations (with same length as yrealImagInds)
        lminSepPoints_y (numpy array):
            the first quadrant y-coordinates of the local minimum separations (with same length as yrealImagInds)
        lmaxSepPoints_x (numpy array):
            the first quadrant x-coordinates of the local maximum separations (with same length as yrealImagInds)
        lmaxSepPoints_y (numpy array):
            the first quadrant y-coordinates of the local maximum separations (with same length as yrealImagInds)
        phi (numpy array):
            angle from X-axis to semi-minor axis of projected ellipse 
        Op (numpy array):
            the geometric center of the projected ellipse
        yrealAllRealInds (numpy array):
            an array of integers acting as indicies of planets which have min, max, local min, local max
    Returns:
        minSepPoints_x_dr (numpy array):
            derotated minSepPoints_x
        minSepPoints_y_dr (numpy array):
            derotated minSepPoints_y
        maxSepPoints_x_dr (numpy array):
            derotated maxSepPoints_x
        maxSepPoints_y_dr (numpy array):
            derotated maxSepPoints_y
        lminSepPoints_x_dr (numpy array):
            derotated lminSepPoints_x
        lminSepPoints_y_dr (numpy array):
            derotated lminSepPoints_y
        lmaxSepPoints_x_dr (numpy array):
            derotated lmaxSepPoints_x
        lmaxSepPoints_y_dr (numpy array):
            derotated lmaxSepPoints_y
    """
    minSepPoints_x_dr = np.zeros(len(minSepPoints_x))
    minSepPoints_y_dr = np.zeros(len(minSepPoints_y))
    maxSepPoints_x_dr = np.zeros(len(maxSepPoints_x))
    maxSepPoints_y_dr = np.zeros(len(maxSepPoints_y))
    lminSepPoints_x_dr = np.zeros(len(lminSepPoints_x))
    lminSepPoints_y_dr = np.zeros(len(lminSepPoints_y))
    lmaxSepPoints_x_dr = np.zeros(len(lmaxSepPoints_x))
    lmaxSepPoints_y_dr = np.zeros(len(lmaxSepPoints_y))
    
    minSepPoints_x_dr, minSepPoints_y_dr = rerotateEllipsePoints(minSepPoints_x, minSepPoints_y,Phi,Op[0],Op[1])
    maxSepPoints_x_dr, maxSepPoints_y_dr = rerotateEllipsePoints(maxSepPoints_x, maxSepPoints_y,Phi,Op[0],Op[1])
    lminSepPoints_x_dr, lminSepPoints_y_dr = rerotateEllipsePoints(lminSepPoints_x, lminSepPoints_y,Phi[yrealAllRealInds],Op[0][yrealAllRealInds],Op[1][yrealAllRealInds])
    lmaxSepPoints_x_dr, lmaxSepPoints_y_dr = rerotateEllipsePoints(lmaxSepPoints_x, lmaxSepPoints_y,Phi[yrealAllRealInds],Op[0][yrealAllRealInds],Op[1][yrealAllRealInds])
    
    return minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr,\
        lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr

def rerotateIntersectionPoints(minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
    fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
    Phi, Op, yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds):
    """ Rotate the intersection points from (the projected ellipse centered at the origin and x-axis aligned with semi-major axis) to the original projected ellipse
    Args:
        fourInt_x (numpy array):
            x coordinates of fourIntInds
        fourInt_y (numpy array):
            y coordinates of fourIntInds
        twoIntSameY_x (numpy array):
            x components of intersections which must occur on same Y side of the projected ellipse as the star
        twoIntSameY_y (numpy array):
            y components of intersections which must occur on same Y side of the projected ellipse as the star
        twoIntOppositeX_x (numpy array):
            x components of intersections which must occur on opposite X side of the projected ellipse as the star
        twoIntOppositeX_y (numpy array):
            y components of intersections which must occur on opposite X side of the projected ellipse as the star
        xIntersectionsOnly2 (numpy array):
            x components of intersections where there must be only 2 intersections
        yIntersectionsOnly2 (numpy array):
            y components of intersections where there must be only 2 intersections
        phi (numpy array):
            angle from X-axis to semi-minor axis of projected ellipse 
        Op (numpy array):
            the geometric center of the projected ellipse
        yrealAllRealInds (numpy array):
            an array of integers acting as indicies of planets which have min, max, local min, local max
        fourIntInds (numpy array):
            indicies of yrealAllRealInds which should have 4 intersections
        twoIntSameYInds (numpy array):
            indicies of yrealAllRealInds which should have 2 intersections on the 
            same Y side of the projected ellipse as the star
        twoIntOppositeXInds (numpy array):
            indicies of yrealAllRealInds which should have 2 intersections on the 
            opposite X side of the projected ellipse as the star
        only2RealInds (numpy array):
            indicies where there can only ever by 2 circle-ellipse intersections
    Returns:
        fourInt_x_dr (numpy array):
            derotated fourInt_x
        fourInt_y_dr (numpy array):
            derotated fourInt_y
        twoIntSameY_x_dr (numpy array):
            derotated twoIntSameY_x
        twoIntSameY_y_dr (numpy array):
            derotated twoIntSameY_y
        twoIntOppositeX_x_dr (numpy array):
            derotated twoIntOppositeX_x
        twoIntOppositeX_y_dr (numpy array):
            derotated twoIntOppositeX_y
        xIntersectionsOnly2_dr (numpy array):
            derotated xIntersectionsOnly2
        yIntersectionsOnly2_dr (numpy array):
            derotated yIntersectionsOnly2
    """
    fourInt_x_dr = np.zeros((len(fourInt_x),4))
    fourInt_y_dr = np.zeros((len(fourInt_y),4))
    twoIntSameY_x_dr = np.zeros((len(twoIntSameY_x),2))
    twoIntSameY_y_dr = np.zeros((len(twoIntSameY_y),2))
    twoIntOppositeX_x_dr = np.zeros((len(twoIntOppositeX_x),2))
    twoIntOppositeX_y_dr = np.zeros((len(twoIntOppositeX_y),2))
    xIntersectionsOnly2_dr = np.zeros((len(xIntersectionsOnly2),2))
    yIntersectionsOnly2_dr = np.zeros((len(yIntersectionsOnly2),2))

    fourInt_x_dr[:,0], fourInt_y_dr[:,0] = rerotateEllipsePoints(fourInt_x[:,0], fourInt_y[:,0],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    fourInt_x_dr[:,1], fourInt_y_dr[:,1] = rerotateEllipsePoints(fourInt_x[:,1], fourInt_y[:,1],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    fourInt_x_dr[:,2], fourInt_y_dr[:,2] = rerotateEllipsePoints(fourInt_x[:,2], fourInt_y[:,2],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    fourInt_x_dr[:,3], fourInt_y_dr[:,3] = rerotateEllipsePoints(fourInt_x[:,3], fourInt_y[:,3],Phi[yrealAllRealInds[fourIntInds]],Op[0][yrealAllRealInds[fourIntInds]],Op[1][yrealAllRealInds[fourIntInds]])
    twoIntSameY_x_dr[:,0], twoIntSameY_y_dr[:,0] = rerotateEllipsePoints(twoIntSameY_x[:,0], twoIntSameY_y[:,0],Phi[yrealAllRealInds[twoIntSameYInds]],Op[0][yrealAllRealInds[twoIntSameYInds]],Op[1][yrealAllRealInds[twoIntSameYInds]])
    twoIntSameY_x_dr[:,1], twoIntSameY_y_dr[:,1] = rerotateEllipsePoints(twoIntSameY_x[:,1], twoIntSameY_y[:,1],Phi[yrealAllRealInds[twoIntSameYInds]],Op[0][yrealAllRealInds[twoIntSameYInds]],Op[1][yrealAllRealInds[twoIntSameYInds]])
    twoIntOppositeX_x_dr[:,0], twoIntOppositeX_y_dr[:,0] = rerotateEllipsePoints(twoIntOppositeX_x[:,0], twoIntOppositeX_y[:,0],Phi[yrealAllRealInds[twoIntOppositeXInds]],Op[0][yrealAllRealInds[twoIntOppositeXInds]],Op[1][yrealAllRealInds[twoIntOppositeXInds]])
    twoIntOppositeX_x_dr[:,1], twoIntOppositeX_y_dr[:,1] = rerotateEllipsePoints(twoIntOppositeX_x[:,1], twoIntOppositeX_y[:,1],Phi[yrealAllRealInds[twoIntOppositeXInds]],Op[0][yrealAllRealInds[twoIntOppositeXInds]],Op[1][yrealAllRealInds[twoIntOppositeXInds]])
    xIntersectionsOnly2_dr[:,0], yIntersectionsOnly2_dr[:,0] = rerotateEllipsePoints(xIntersectionsOnly2[:,0], yIntersectionsOnly2[:,0],Phi[only2RealInds],Op[0][only2RealInds],Op[1][only2RealInds])
    xIntersectionsOnly2_dr[:,1], yIntersectionsOnly2_dr[:,1] = rerotateEllipsePoints(xIntersectionsOnly2[:,1], yIntersectionsOnly2[:,1],Phi[only2RealInds],Op[0][only2RealInds],Op[1][only2RealInds])
    
    return fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr

def calcMasterIntersections(sma,e,W,w,inc,s_circle,starMass,plotBool):
    """ A method for calculating the nu and times of orbit and circle intersections as well as extrema
    In the memory efficient method, there are approximately 349 Bytes per planet
    When plotting, extra variables are saved resulting in approximately 373 Bytes per planet
    Args:
        sma (numpy array):
            semi-major axis
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the ascending nodes
        w (numpy array):
            Argument of periapsis
        inc (numpy array):
            inclination
        s_circle (numpy array):
            circle radius to calculate intersections for
        starMass (astropy quantity):
            mass of the star
        plotBool (boolean):
            a boolean used to determine whether some outputs are deleted
            for memory efficiency or kept for plotting
    Returns:
        dmajorp (numpy array): 
            the semi-major axis of the projected ellipse
        dminorp (numpy array):
            the semi-minor axis of the projected ellipse
        theta_OpQ_X (numpy array):
            the angle formed between point Q, the geometric center of
            the projected ellipse, and the X-axis
        theta_OpQp_X (numpy array):
            the angle formed between point Q, the geometric center of
            the projected ellipse, and the X-axis
        Op (numpy array):
            the geometric center of the projected ellipse
        x (numpy array):
            the x component of the projected star location
        y (numpy array):
            the y component of the projected star location
        phi (numpy array):
            angle from X-axis to semi-minor axis of projected ellipse 

        xreal (numpy array):
        only2RealInds (numpy array):
        yrealAllRealInds (numpy array):
        fourIntInds (numpy array):
        twoIntOppositeXInds (numpy array):
        twoIntSameYInds (numpy array):
        nu_minSepPoints (numpy array):
        nu_maxSepPoints (numpy array):
        nu_lminSepPoints (numpy array):
        nu_lmaxSepPoints (numpy array):
        nu_fourInt (numpy array):
        nu_twoIntSameY (numpy array):
        nu_twoIntOppositeX (numpy array):
        nu_IntersectionsOnly2 (numpy array):
        yrealImagInds (numpy array):
        t_minSep (numpy array):
        t_maxSep (numpy array):
        t_lminSep (numpy array):
        t_lmaxSep (numpy array):
        t_fourInt0 (numpy array):
        t_fourInt1 (numpy array):
        t_fourInt2 (numpy array):
        t_fourInt3 (numpy array):
        t_twoIntSameY0 (numpy array):
        t_twoIntSameY1 (numpy array):
        t_twoIntOppositeX0 (numpy array):
        t_twoIntOppositeX1 (numpy array):
        t_IntersectionOnly20 (numpy array):
        t_IntersectionOnly21 (numpy array):
        minSepPoints_x (numpy array):
        minSepPoints_y (numpy array):
        maxSepPoints_x (numpy array):
        maxSepPoints_y (numpy array):
        lminSepPoints_x (numpy array):
        lminSepPoints_y (numpy array):
        lmaxSepPoints_x (numpy array):
        lmaxSepPoints_y (numpy array):
        minSep (numpy array):
        maxSep (numpy array):
        lminSep (numpy array):
        lmaxSep (numpy array):
        errors_fourInt0 (numpy array):
        errors_fourInt1 (numpy array):
        errors_fourInt2 (numpy array):
        errors_fourInt3 (numpy array):
        errors_twoIntSameY0 (numpy array):
        errors_twoIntSameY1 (numpy array):
        errors_twoIntOppositeX0 (numpy array):
        errors_twoIntOppositeX1 (numpy array):
        errors_IntersectionsOnly2X0 (numpy array):
        errors_IntersectionsOnly2X1 (numpy array):
        type0_0Inds (numpy array):\
        type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
        type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds
        fourInt_x (numpy array):
        fourInt_y (numpy array):
        twoIntSameY_x (numpy array):
        twoIntSameY_y (numpy array):
        twoIntOppositeX_x (numpy array):
        twoIntOppositeX_y (numpy array):
        xIntersectionsOnly2 (numpy array):
        yIntersectionsOnly2 (numpy array):
        typeInds0 (numpy array):
        typeInds1 (numpy array):
        typeInds2 (numpy array):
        typeInds3 (numpy array):
        periods (numpy array):
            planet period in years
    """
    #### Calculate Projected Ellipse Angles and Minor Axis
    # start0 = time.time()
    dmajorp, dminorp, theta_OpQ_X, theta_OpQp_X = projected_apbpPsipsi(sma,e,W,w,inc)#dmajorp_v2, dminorp_v2, Psi_v2, psi_v2, Psi, psi,
    # stop0 = time.time()
    # print('stop0: ' + str(stop0-start0))
    #3D Ellipse Center
    # start1 = time.time()
    Op = projected_Op(sma,e,W,w,inc)
    # stop1 = time.time()
    # print('stop1: ' + str(stop1-start1))
    # del start1, stop1

    # Checks
    if not np.all(dmajorp <= sma):
        print("Not all Semi-major axis of the projected ellipse are less than the original 3D ellipse, caused by circular orbits required for circular orbits")
        assert np.all(sma - dmajorp >= -1e-12), "Not all Semi-major axis of the projected ellipse are less than the original 3D ellipse" #required for circular orbits
    assert np.all(dminorp <= dmajorp), "All projected Semi-minor axes are less than all projected semi-major axes"

    #### Derotate Ellipse Calculations
    # start5 = time.time()
    x, y, Phi = derotatedEllipse(theta_OpQ_X, theta_OpQp_X, Op)
    #x- x coordinates of host star relative to projected ellipse center
    #y- y coordinates of host star relative to projected ellipse center
    #Phi- Angle of projected ellipse semi-major axis from x-axis
    if plotBool == False: #deletes these angles because they are no longer necessary
        del theta_OpQ_X, theta_OpQp_X
    # stop5 = time.time()
    # print('stop5: ' + str(stop5-start5))
    # del start5, stop5
    ####

    #### Calculate X,Y Position of Minimum and Maximums with Quartic
    # start7 = time.time()
    A, B, C, D = quarticCoefficients_smin_smax_lmin_lmax(dmajorp.astype('complex128'), dminorp, np.abs(x), np.abs(y)) #calculate the quartic solutions to the min-max separation problem
    #xreal, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
    xreal, _, _, _, _, _ = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
    del A, B, C, D #delting for memory efficiency
    if not np.max(np.nanmin(np.abs(np.imag(xreal)),axis=1)) < 1e-5:
        myInd = np.where(np.nanmin(np.abs(np.imag(xreal)),axis=1) >= 1e-5)[0][0]
        print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
        print('badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)')
        with open("/home/dean/Documents/exosims/twoDetMC/badKOE.py","a") as f:
            f.write('#calcMasterIntersections' + '\n')
            f.write('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]) + '\n')
            f.write('badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)' + '\n')
    assert np.max(np.nanmin(np.abs(np.imag(xreal)),axis=1)) < 1e-5, 'At least one row has min > 1e-5' #this ensures each row has a solution
    #myInd = np.where(np.nanmin(np.abs(np.imag(xreal)),axis=1) > 1e-5)
    #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
    #print(w[np.argmax(np.nanmin(np.abs(np.imag(xreal)),axis=1))]) #prints the argument of perigee (assert above fails on 1.57 or 1.5*pi)
    #Failure of the above occured where w=4.712 which is approx 1.5pi
    #NOTE: originally 1e-15 but there were some with x=1e-7 and w=pi/2, 5e-6 from 
    #DELETEtind = np.argmax(np.nanmin(np.abs(np.imag(xreal)),axis=1)) #DELETE
    #DELETEtinds = np.argsort(np.nanmin(np.abs(np.imag(xreal)),axis=1)) #DELETE
    #DELETEdel tind, tinds #DELETE
    xreal.real = np.abs(xreal) #all solutions should be positive
    # stop7 = time.time()
    # print('stop7: ' + str(stop7-start7))
    # del stop7, start7
    #DELETEprintKOE(ind,a,e,W,w,inc)

    #### Technically, each row must have at least 2 solutions, but whatever
    # start8 = time.time()
    yreal = ellipseYFromX(xreal.astype('complex128'), dmajorp, dminorp) #Calculates the y values corresponding to the x values in the first quadrant of an ellipse
    # stop8 = time.time()
    # print('stop8: ' + str(stop8-start8))
    # del start8, stop8
    ####

    #### Calculate Minimum, Maximum, Local Minimum, Local Maximum Separations
    # start9 = time.time()
    minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds = smin_smax_slmin_slmax(len(x), xreal, yreal, np.abs(x), np.abs(y), x, y)
    lminSepPoints_x = np.real(lminSepPoints_x)
    lminSepPoints_y = np.real(lminSepPoints_y)
    lmaxSepPoints_x = np.real(lmaxSepPoints_x)
    lmaxSepPoints_y = np.real(lmaxSepPoints_y)
    # stop9 = time.time()
    # print('stop9: ' + str(stop9-start9))
    # del start9, stop9
    ####

    #### Ellipse Circle Intersection #######################################################################
    # start11 = time.time()
    only2RealInds, typeInds0, typeInds1, typeInds2, typeInds3,\
            fourIntInds, fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y,\
            twoIntOppositeXInds, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2, twoIntSameYInds,\
            type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,\
            type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,\
            _ = ellipseCircleIntersections(s_circle, dmajorp, dminorp, np.abs(x), np.abs(y), x, y, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds,sma, e, W, w, inc)
    if plotBool == False:
        del typeInds0, typeInds1, typeInds2, typeInds3
        del type0_0Inds,type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds
        del type2_0Inds,type2_1Inds,type2_2Inds,type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds
    # stop11 = time.time()
    # print('stop11: ' + str(stop11-start11))
    # del start11, stop11
    ####


    #### Correct Ellipse Circle Intersections fourInt1 ####################################
    fourInt_x[:,0], fourInt_y[:,0] = intersectionFixer_pm(x, y, fourInt_x[:,0], fourInt_y[:,0], yrealAllRealInds[fourIntInds], s_circle[fourIntInds]) #necessary because a minority of cases occur in quadrant 3
    fourInt_x[:,1], fourInt_y[:,1] = intersectionFixer_pm(x, y, fourInt_x[:,1], fourInt_y[:,1], yrealAllRealInds[fourIntInds], s_circle[fourIntInds]) #necessary because a minority of cases occur in quadrant 4
    #fourInt_x[:,2], fourInt_y[:,2] #unnecessary because these solution quadrants are well known
    #fourInt_x[:,3], fourInt_y[:,3] #unnecessary becasue these solution quadrants are well known
    #### Correct Ellipse Circle Intersections twoIntSameY0
    twoIntSameY_x[:,0], twoIntSameY_y[:,0] = intersectionFixer_pm(x, y, twoIntSameY_x[:,0], twoIntSameY_y[:,0], yrealAllRealInds[twoIntSameYInds], s_circle[twoIntSameYInds])
    #### Correct Ellipse Circle Intersections twoIntSameY1 
    twoIntSameY_x[:,1], twoIntSameY_y[:,1] = intersectionFixer_pm(x, y, twoIntSameY_x[:,1], twoIntSameY_y[:,1], yrealAllRealInds[twoIntSameYInds], s_circle[twoIntSameYInds])
    #### Correct Ellipse Circle Intersections twoIntOppositeX0
    twoIntOppositeX_x[:,0], twoIntOppositeX_y[:,0] = intersectionFixer_pm(x, y, twoIntOppositeX_x[:,0], twoIntOppositeX_y[:,0], yrealAllRealInds[twoIntOppositeXInds], s_circle[twoIntOppositeXInds])
    #### Correct Ellipse Circle Intersections twoIntOppositeX1 
    twoIntOppositeX_x[:,1], twoIntOppositeX_y[:,1] = intersectionFixer_pm(x, y, twoIntOppositeX_x[:,1], twoIntOppositeX_y[:,1], yrealAllRealInds[twoIntOppositeXInds], s_circle[twoIntOppositeXInds])
    #### COULD RUN ON OTHER CASES #########################################################

    #### Rerotate Extrema and Intersection Points
    # start13 = time.time()
    minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
        fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr = \
        rerotateExtremaAndIntersectionPoints(minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
        fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2,\
        Phi, Op, yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds)
    if plotBool == False:
        del minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y
        del fourInt_x, fourInt_y, twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2
    # stop13 = time.time()
    # print('stop13: ' + str(stop13-start13))
    # del start13, stop13
    ####

    #### Calculate True Anomalies of Points
    # start14 = time.time()

    nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2\
         = trueAnomaliesOfPoints(minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
        fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr,\
        yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, W, w, inc)
    del minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr
    del fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr
    # stop14 = time.time()
    # print('stop14: ' + str(stop14-start14))
    # del start14, stop14
    #Now can I delete the x,y points?
    #del minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, fourInt_x, fourInt_y
    #del twoIntSameY_x, twoIntSameY_y, twoIntOppositeX_x, twoIntOppositeX_y, xIntersectionsOnly2, yIntersectionsOnly2
    ####

    #### Fix minSep True Anomalies
    nu_minSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_minSepPoints,np.arange(len(sma)),minSep)
    ####
    #### Fix maxSep True Anomalies
    nu_maxSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_maxSepPoints,np.arange(len(sma)),maxSep)
    ####
    #### Fix lminSep True Anomalies
    nu_lminSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_lminSepPoints,yrealAllRealInds,lminSep)
    ####
    #### Fix lmaxSep True Anomalies
    nu_lmaxSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_lmaxSepPoints,yrealAllRealInds,lmaxSep)
    ####

    #### Correcting nu for ellipse-circle intersections
    #### yrealAllRealInds[fourIntInds]
    nu_fourInt[:,0], errors_fourInt0 = nuCorrections_int(sma,e,W,w,inc,s_circle,nu_fourInt[:,0],yrealAllRealInds,fourIntInds)
    nu_fourInt[:,1], errors_fourInt1 = nuCorrections_int(sma,e,W,w,inc,s_circle,nu_fourInt[:,1],yrealAllRealInds,fourIntInds)
    nu_fourInt[:,2], errors_fourInt2 = nuCorrections_int(sma,e,W,w,inc,s_circle,nu_fourInt[:,2],yrealAllRealInds,fourIntInds)
    nu_fourInt[:,3], errors_fourInt3 = nuCorrections_int(sma,e,W,w,inc,s_circle,nu_fourInt[:,3],yrealAllRealInds,fourIntInds)
    if plotBool == False:
        del errors_fourInt0, errors_fourInt1, errors_fourInt2, errors_fourInt3
    ####
    #### yrealAllRealInds[twoIntSameYInds]
    if len(twoIntSameYInds) != 0:
        nu_twoIntSameY[:,0], errors_twoIntSameY0 = nuCorrections_int(sma,e,W,w,inc,s_circle,nu_twoIntSameY[:,0],yrealAllRealInds,twoIntSameYInds)
        nu_twoIntSameY[:,1], errors_twoIntSameY1 = nuCorrections_int(sma,e,W,w,inc,s_circle,nu_twoIntSameY[:,1],yrealAllRealInds,twoIntSameYInds)
        if plotBool == False:
            del errors_twoIntSameY0, errors_twoIntSameY1
    ####
    #### yrealAllRealInds[twoIntOppositeXInds]
    nu_twoIntOppositeX[:,0], errors_twoIntOppositeX0 = nuCorrections_int(sma,e,W,w,inc,s_circle,nu_twoIntOppositeX[:,0],yrealAllRealInds,twoIntOppositeXInds)
    nu_twoIntOppositeX[:,1], errors_twoIntOppositeX1 = nuCorrections_int(sma,e,W,w,inc,s_circle,nu_twoIntOppositeX[:,1],yrealAllRealInds,twoIntOppositeXInds)
    if plotBool == False:
        del errors_twoIntOppositeX0, errors_twoIntOppositeX1
    ####
    #### only2RealInds
    nu_IntersectionsOnly2[:,0], errors_IntersectionsOnly2X0 = nuCorrections_int(sma,e,W,w,inc,s_circle,nu_IntersectionsOnly2[:,0],np.arange(len(sma)),only2RealInds)
    nu_IntersectionsOnly2[:,1], errors_IntersectionsOnly2X1 = nuCorrections_int(sma,e,W,w,inc,s_circle,nu_IntersectionsOnly2[:,1],np.arange(len(sma)),only2RealInds)
    if plotBool == False:
        del errors_IntersectionsOnly2X0, errors_IntersectionsOnly2X1
    ####

    #### Calculate time from nu
    periods = (2.*np.pi*np.sqrt((sma*u.AU)**3./(const.G.to('AU3 / (kg s2)')*starMass))).to('year').value
    t_minSep = timeFromTrueAnomaly(nu_minSepPoints,periods,e)
    t_maxSep = timeFromTrueAnomaly(nu_maxSepPoints,periods,e)
    t_lminSep = timeFromTrueAnomaly(nu_lminSepPoints,periods[yrealAllRealInds],e[yrealAllRealInds])
    t_lmaxSep = timeFromTrueAnomaly(nu_lmaxSepPoints,periods[yrealAllRealInds],e[yrealAllRealInds])
    t_fourInt0 = timeFromTrueAnomaly(nu_fourInt[:,0],periods[yrealAllRealInds[fourIntInds]],e[yrealAllRealInds[fourIntInds]])
    t_fourInt1 = timeFromTrueAnomaly(nu_fourInt[:,1],periods[yrealAllRealInds[fourIntInds]],e[yrealAllRealInds[fourIntInds]])
    t_fourInt2 = timeFromTrueAnomaly(nu_fourInt[:,2],periods[yrealAllRealInds[fourIntInds]],e[yrealAllRealInds[fourIntInds]])
    t_fourInt3 = timeFromTrueAnomaly(nu_fourInt[:,3],periods[yrealAllRealInds[fourIntInds]],e[yrealAllRealInds[fourIntInds]])
    if not len(twoIntSameYInds) == 0:
        t_twoIntSameY0 = timeFromTrueAnomaly(nu_twoIntSameY[:,0],periods[yrealAllRealInds[twoIntSameYInds]],e[yrealAllRealInds[twoIntSameYInds]])
        t_twoIntSameY1 = timeFromTrueAnomaly(nu_twoIntSameY[:,1],periods[yrealAllRealInds[twoIntSameYInds]],e[yrealAllRealInds[twoIntSameYInds]])
    else:
        t_twoIntSameY0 = np.asarray([])
        t_twoIntSameY1 = np.asarray([])
    t_twoIntOppositeX0 = timeFromTrueAnomaly(nu_twoIntOppositeX[:,0],periods[yrealAllRealInds[twoIntOppositeXInds]],e[yrealAllRealInds[twoIntOppositeXInds]])
    t_twoIntOppositeX1 = timeFromTrueAnomaly(nu_twoIntOppositeX[:,1],periods[yrealAllRealInds[twoIntOppositeXInds]],e[yrealAllRealInds[twoIntOppositeXInds]])
    t_IntersectionOnly20 = timeFromTrueAnomaly(nu_IntersectionsOnly2[:,0],periods[only2RealInds],e[only2RealInds])
    t_IntersectionOnly21 = timeFromTrueAnomaly(nu_IntersectionsOnly2[:,1],periods[only2RealInds],e[only2RealInds])

    #### Memory Calculations
    #Necessary Variables
    if plotBool == True:
        memory_necessary = [inc.nbytes,w.nbytes,W.nbytes,sma.nbytes,e.nbytes,dmajorp.nbytes,dminorp.nbytes,\
            Op.nbytes,x.nbytes,y.nbytes,Phi.nbytes,xreal.nbytes,only2RealInds.nbytes,yrealAllRealInds.nbytes,fourIntInds.nbytes,twoIntOppositeXInds.nbytes,twoIntSameYInds.nbytes,\
            nu_minSepPoints.nbytes,nu_maxSepPoints.nbytes,nu_lminSepPoints.nbytes,nu_lmaxSepPoints.nbytes,nu_fourInt.nbytes,nu_twoIntSameY.nbytes,nu_twoIntOppositeX.nbytes,nu_IntersectionsOnly2.nbytes,\
            minSepPoints_x.nbytes, minSepPoints_y.nbytes, maxSepPoints_x.nbytes, maxSepPoints_y.nbytes, lminSepPoints_x.nbytes, lminSepPoints_y.nbytes, lmaxSepPoints_x.nbytes,\
            lmaxSepPoints_y.nbytes, minSep.nbytes, maxSep.nbytes, lminSep.nbytes, lmaxSep.nbytes, yrealImagInds.nbytes,\
            t_minSep.nbytes,t_maxSep.nbytes,t_lminSep.nbytes,t_lmaxSep.nbytes,t_fourInt0.nbytes,t_fourInt1.nbytes,t_fourInt2.nbytes,t_fourInt3.nbytes,\
            t_twoIntSameY0.nbytes,t_twoIntSameY1.nbytes,t_twoIntOppositeX0.nbytes,t_twoIntOppositeX1.nbytes,t_IntersectionOnly20.nbytes,t_IntersectionOnly21.nbytes]
        print('memory_necessary Used: ' + str(np.sum(memory_necessary)/10**9) + ' GB')


    #Things currently calculated, used, and later deleted
    #A, B, C, D
    #minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr
    #fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr

    # Vestigal Variables
    #TODO a and b are duplicates of dmajorp and dminorp
    # memory_vestigal = [0]
    #theta_OpQ_X.nbytes,theta_OpQp_X.nbytes
    #a.nbytes,b.nbytes,
    #error_numinSep.nbytes,error_numaxSep.nbytes,error_nulminSep.nbytes,error_nulmaxSep.nbytes,
    #dmajorp_v2.nbytes,dminorp_v2.nbytes,Psi_v2.nbytes,psi_v2.nbytes,Psi.nbytes,psi.nbytes,
    #delta.nbytes,delta_0.nbytes,P.nbytes, #not 100% sureD2.nbytes,R.nbytes,
    #allIndsUsed.nbytes
    # print('memory_vestigal Used: ' + str(np.sum(memory_vestigal)/10**9) + ' GB')

    # Variables Only For Plotting
    if plotBool == True:
        memory_plotting = [errors_fourInt0.nbytes,
        errors_fourInt1.nbytes,
        errors_fourInt2.nbytes,
        errors_fourInt3.nbytes,
        #errors_twoIntSameY0.nbytes,
        #errors_twoIntSameY1.nbytes,
        errors_twoIntOppositeX0.nbytes,
        errors_twoIntOppositeX1.nbytes,
        errors_IntersectionsOnly2X0.nbytes,
        errors_IntersectionsOnly2X1.nbytes,
        type0_0Inds.nbytes,
        type0_1Inds.nbytes,
        type0_2Inds.nbytes,
        type0_3Inds.nbytes,
        type0_4Inds.nbytes,
        type1_0Inds.nbytes,
        type1_1Inds.nbytes,
        type1_2Inds.nbytes,
        type1_3Inds.nbytes,
        type1_4Inds.nbytes,
        type2_0Inds.nbytes,
        type2_1Inds.nbytes,
        type2_2Inds.nbytes,
        type2_3Inds.nbytes,
        type2_4Inds.nbytes,
        type3_0Inds.nbytes,
        type3_1Inds.nbytes,
        type3_2Inds.nbytes,
        type3_3Inds.nbytes,
        type3_4Inds.nbytes,
        fourInt_x.nbytes,
        fourInt_y.nbytes,
        twoIntSameY_x.nbytes,
        twoIntSameY_y.nbytes,
        twoIntOppositeX_x.nbytes,
        twoIntOppositeX_y.nbytes,
        xIntersectionsOnly2.nbytes,
        yIntersectionsOnly2.nbytes,
        typeInds0.nbytes,
        typeInds1.nbytes,
        typeInds2.nbytes,
        typeInds3.nbytes]
        print('memory_plotting Used: ' + str(np.sum(memory_plotting)/10**9) + ' GB')

    if plotBool == False:
        return dmajorp,dminorp,_,_,Op,x,y,Phi,xreal,only2RealInds,\
            yrealAllRealInds,fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,\
            nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt,nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
            t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,t_twoIntSameY0,\
            t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
            _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_, periods
    else:
        return dmajorp,dminorp,theta_OpQ_X,theta_OpQp_X,Op,x,y,Phi,xreal,only2RealInds,yrealAllRealInds,\
            fourIntInds,twoIntOppositeXInds,twoIntSameYInds,nu_minSepPoints,nu_maxSepPoints,nu_lminSepPoints,nu_lmaxSepPoints,nu_fourInt,\
            nu_twoIntSameY,nu_twoIntOppositeX,nu_IntersectionsOnly2, yrealImagInds,\
            t_minSep,t_maxSep,t_lminSep,t_lmaxSep,t_fourInt0,t_fourInt1,t_fourInt2,t_fourInt3,t_twoIntSameY0,\
            t_twoIntSameY1,t_twoIntOppositeX0,t_twoIntOppositeX1,t_IntersectionOnly20,t_IntersectionOnly21,\
            minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y, minSep, maxSep, lminSep, lmaxSep,\
            errors_fourInt0,errors_fourInt1,errors_fourInt2,errors_fourInt3,errors_twoIntSameY0,\
            errors_twoIntSameY1,errors_twoIntOppositeX0,errors_twoIntOppositeX1,errors_IntersectionsOnly2X0,errors_IntersectionsOnly2X1,type0_0Inds,\
            type0_1Inds,type0_2Inds,type0_3Inds,type0_4Inds,type1_0Inds,type1_1Inds,type1_2Inds,type1_3Inds,type1_4Inds,type2_0Inds,type2_1Inds,type2_2Inds,\
            type2_3Inds,type2_4Inds,type3_0Inds,type3_1Inds,type3_2Inds,type3_3Inds,type3_4Inds,fourInt_x,fourInt_y,twoIntSameY_x,twoIntSameY_y,twoIntOppositeX_x,\
            twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods

def calc_planet_dmagSecondDerivative(e,inc,w,v):
    """ Calculates the second derivative of the dmag function assuming the quasi-lambert phase function
    From AnalyticalNuFromDmag3.ipynb under Second Derivative of RHS
    Args:
        e (numpy array):
        inc (numpy array):
        w (numpy array):
        v (numpy array):
    Returns:
        secondDerivatives (numpy array):
    """
    e = np.tile(e,(16,1)).T
    w = np.tile(w,(16,1)).T
    inc = np.tile(inc,(16,1)).T

    secondDerivatives = -8.*e**2.*np.sin(inc)**2.*np.sin(v)**4.*np.sin(w)**2. + 4.*e**2.*np.sin(inc)**2.*np.sin(v)**4. + 3.*e**2.*np.sin(inc)**2.*np.sin(v)**3.*np.sin(w)*np.cos(v)*np.cos(w) +\
        9.*e**2.*np.sin(inc)**2.*np.sin(v)**2.*np.sin(w)**2. - 4.*e**2.*np.sin(inc)**2.*np.sin(v)**2. - 5.*e**2.*np.sin(inc)**2.*np.sin(v)*np.sin(w)*np.cos(v)**3.*np.cos(w) - 3.*e**2.*np.sin(inc)**2.*np.sin(w)**2./2. +\
        e**2.*np.sin(inc)**2./2. + 9.*e**2.*np.sin(inc)*np.sin(v)**3.*np.cos(w)/2. + 3.*e**2.*np.sin(inc)*np.sin(v)**2.*np.sin(w)*np.cos(v) - 7.*e**2.*np.sin(inc)*np.sin(v)*np.cos(w)/2. -\
        3.*e**2.*np.sin(inc)*np.sin(w)*np.cos(v)**3./2. + e**2.*np.sin(v)**2. - e**2./2. + 9.*e*np.sin(inc)**2.*np.sin(v)**3.*np.sin(w)*np.cos(w) + 13.*e*np.sin(inc)**2.*np.sin(v)**2.*np.sin(w)**2.*np.cos(v)/2. -\
        7.*e*np.sin(inc)**2.*np.sin(v)**2.*np.cos(v)/2. - 7.*e*np.sin(inc)**2.*np.sin(v)*np.sin(w)*np.cos(w) - 5.*e*np.sin(inc)**2.*np.sin(w)**2.*np.cos(v)**3./2. + e*np.sin(inc)**2.*np.cos(v)**3. -\
        4.*e*np.sin(inc)*np.sin(v)*np.cos(v + w) - 2.*e*np.sin(inc)*np.sin(w) - e*np.cos(v)/2. - np.sin(inc)**2.*np.sin(v)**2. - 2.*np.sin(inc)**2.*np.sin(v)*np.sin(w)*np.cos(v + w) -\
        np.sin(inc)**2.*np.sin(w)**2. + np.sin(inc)**2./2. - np.sin(inc)*np.sin(v + w)/2.
    return secondDerivatives

#### nu from dmag functions ##############################################################################
def calc_planet_dmagmin_dmagmax(e,inc,w,a,p,Rp):
    """ TODO: rename to calc_planet_dmag_extrema
    A method for calculating the minimum and maximum dmag of any given planet
    Assumes the planet has a quasi-lambert phase function (a poor approximation).
    Args:
        e (numpy array):
        inc (numpy array):
        w (numpy array):
        a (numpy array):
        p (numpy array):
        Rp (numpy array):
    Returns:
        mindmag (numpy array):
            an array containing the minimum dmags 
        maxdmag (numpy array):
        dmaglminAll (numpy array):
        dmaglmaxAll (numpy array):
        indsWith2 (numpy array):
            planet indicies where there are only 2 solutions
        indsWith4 (numpy array):
            planet indicies where there are 4 solutions
        nuMinDmag (numpy array):
        nuMaxDmag (numpy array):
        nulminAll (numpy array):
        nulmaxAll (numpy array):
    """
    A = e**4.*np.sin(inc)**4.*np.sin(w)**4. + 2.*e**4.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2. + e**4.*np.sin(inc)**4.*np.cos(w)**4.
    B = 3.*e**4.*np.sin(inc)**3.*np.sin(w)**3. + 3.*e**4.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + 3.*e**3.*np.sin(inc)**4.*np.sin(w)**4. + 6.*e**3.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2. + 3.*e**3.*np.sin(inc)**4.*np.cos(w)**4.
    C = -e**4.*np.sin(inc)**4.*np.sin(w)**4. - 3.*e**4.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2. - 2.*e**4.*np.sin(inc)**4.*np.cos(w)**4. + 13.*e**4.*np.sin(inc)**2.*np.sin(w)**2./4. + 5.*e**4.*np.sin(inc)**2.*np.cos(w)**2./4. + 17.*e**3.*np.sin(inc)**3.*np.sin(w)**3./2. + 17.*e**3.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. + 13.*e**2.*np.sin(inc)**4.*np.sin(w)**4./4. + 13.*e**2.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + 13.*e**2.*np.sin(inc)**4.*np.cos(w)**4./4.
    D = -3.*e**4.*np.sin(inc)**3.*np.sin(w)**3. - 4.*e**4.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + 3.*e**4.*np.sin(inc)*np.sin(w)/2. - 3.*e**3.*np.sin(inc)**4.*np.sin(w)**4. - 17.*e**3.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. - 11.*e**3.*np.sin(inc)**4.*np.cos(w)**4./2. + 17.*e**3.*np.sin(inc)**2.*np.sin(w)**2./2. + 7.*e**3.*np.sin(inc)**2.*np.cos(w)**2./2. + 17.*e**2.*np.sin(inc)**3.*np.sin(w)**3./2. + 17.*e**2.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. + 3.*e*np.sin(inc)**4.*np.sin(w)**4./2. + 3.*e*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2. + 3.*e*np.sin(inc)**4.*np.cos(w)**4./2.
    E = 5.*e**4.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./4. + 5.*e**4.*np.sin(inc)**4.*np.cos(w)**4./4. - 13.*e**4.*np.sin(inc)**2.*np.sin(w)**2./4. - 3.*e**4.*np.sin(inc)**2.*np.cos(w)**2./2. + e**4./4. - 17.*e**3.*np.sin(inc)**3.*np.sin(w)**3./2. - 10.*e**3.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + 7.*e**3.*np.sin(inc)*np.sin(w)/2. - 13.*e**2.*np.sin(inc)**4.*np.sin(w)**4./4. - 17.*e**2.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. - 21.*e**2.*np.sin(inc)**4.*np.cos(w)**4./4. + 15.*e**2.*np.sin(inc)**2.*np.sin(w)**2./2. + 7.*e**2.*np.sin(inc)**2.*np.cos(w)**2./2. + 7.*e*np.sin(inc)**3.*np.sin(w)**3./2. + 7.*e*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. + np.sin(inc)**4.*np.sin(w)**4./4. + np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + np.sin(inc)**4.*np.cos(w)**4./4.
    F = 3.*e**4.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. - 3.*e**4.*np.sin(inc)*np.sin(w)/2. + 7.*e**3.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + 3.*e**3.*np.sin(inc)**4.*np.cos(w)**4. - 17.*e**3.*np.sin(inc)**2.*np.sin(w)**2./2. - 7.*e**3.*np.sin(inc)**2.*np.cos(w)**2./2. + e**3./2. - 17.*e**2.*np.sin(inc)**3.*np.sin(w)**3./2. - 8.*e**2.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + 5.*e**2.*np.sin(inc)*np.sin(w)/2. - 3.*e*np.sin(inc)**4.*np.sin(w)**4./2. - 7.*e*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. - 2.*e*np.sin(inc)**4.*np.cos(w)**4. + 5.*e*np.sin(inc)**2.*np.sin(w)**2./2. + 3.*e*np.sin(inc)**2.*np.cos(w)**2./2. + np.sin(inc)**3.*np.sin(w)**3./2. + np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2.
    G = -e**4.*np.sin(inc)**4.*np.cos(w)**4./4. + e**4.*np.sin(inc)**2.*np.cos(w)**2./2. - e**4./4. + 7.*e**3.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. - 7.*e**3.*np.sin(inc)*np.sin(w)/2. + 7.*e**2.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + 9.*e**2.*np.sin(inc)**4.*np.cos(w)**4./4. - 15.*e**2.*np.sin(inc)**2.*np.sin(w)**2./2. - 5.*e**2.*np.sin(inc)**2.*np.cos(w)**2./2. + e**2./4. - 7.*e*np.sin(inc)**3.*np.sin(w)**3./2. - 2.*e*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + e*np.sin(inc)*np.sin(w)/2. - np.sin(inc)**4.*np.sin(w)**4./4. - np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. - np.sin(inc)**4.*np.cos(w)**4./4. + np.sin(inc)**2.*np.sin(w)**2./4. + np.sin(inc)**2.*np.cos(w)**2./4.
    H = -e**3.*np.sin(inc)**4.*np.cos(w)**4./2. + e**3.*np.sin(inc)**2.*np.cos(w)**2. - e**3./2. + 5.*e**2.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. - 5.*e**2.*np.sin(inc)*np.sin(w)/2. + 3.*e*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + e*np.sin(inc)**4.*np.cos(w)**4./2. - 5.*e*np.sin(inc)**2.*np.sin(w)**2./2. - e*np.sin(inc)**2.*np.cos(w)**2./2. - np.sin(inc)**3.*np.sin(w)**3./2.
    I = -e**2.*np.sin(inc)**4.*np.cos(w)**4./4. + e**2.*np.sin(inc)**2.*np.cos(w)**2./2. - e**2./4. + e*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. - e*np.sin(inc)*np.sin(w)/2. + np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./4. - np.sin(inc)**2.*np.sin(w)**2./4.
    coeffs = np.asarray([A,B,C,D,E,F,G,H,I]) #compile the coefficients into a single array
    del A, B, C, D, E, F, G, H, I #delete the
    #solve for x in the polynomial (where x=cos(nu))
    out = list()
    for i in np.arange(coeffs.shape[1]):
        tmp = np.roots(coeffs[:,i])
        if not tmp.shape[0] == 8:
            tmp = np.append(tmp[0:2],np.append(np.asarray([1.j,1.j]),np.append(tmp[2:4],np.asarray([1.j,1.j])))) #done because I think order matters
            #tmp = np.append(tmp,np.asarray([1.j,1.j,1.j,1.j]))
        out.append(tmp) # this is x
    out = np.asarray(out,dtype=np.complex128)
    del coeffs #delete the coefficients. They are no longer needed

    # Saving out (necessary in case all sols in a row are filtered out)
    outSaved = out.copy()

    #### Minor correction for inds marginally outside the valid bounds (the error is nominally 1e-6 so if a solution is greater than 1 by 1e-6, then correct it to 1.
    out[(np.abs(out.imag) <= 1e-7)*(out.real < -1.)*(out.real >= -1.-1e-4)] = -np.ones(out.shape)[(np.abs(out.imag) <= 1e-7)*(out.real < -1.)*(out.real >= -1.-1e-4)] #move those solutions that might be oh so slightly out of bounds back in bounds
    out[(np.abs(out.imag) <= 1e-7)*(out.real > 1.)*(out.real <= 1.+1e-4)] = np.ones(out.shape)[(np.abs(out.imag) <= 1e-7)*(out.real > 1.)*(out.real <= 1.+1e-4)] #move those solutions that might be oh so slightly out of bounds back in bounds

    #### Throw out roots not in correct bounds
    inBoundsBools = (np.abs(out.imag) <= 1e-7)*(out.real >= -1.)*(out.real <= 1.) #the out2 solutions that are inside of the desired bounds
    outBoundsBools = np.logical_not(inBoundsBools) # the out2 solutions that are inside the desired bounds
    outReal = np.real(out) #filling in all terms with numbers with the correct shape
    outReal[outBoundsBools] = out[outBoundsBools]*np.nan #make all zeros out of legal bounds nan

    #If there are any all nan rows
    allNanInds = np.where(np.all(np.isnan(out),axis=1))[0]
    outReal[allNanInds] = outSaved[allNanInds].real
    outReal[allNanInds][np.logical_not((np.abs(outSaved[allNanInds].imag) <= 1e-3)*(outSaved[allNanInds].real >= -1.)*(outSaved[allNanInds].real <= 1.))] = np.nan

    #For arccos in 0-pi
    nuReal = np.ones(outReal.shape)*np.nan
    nuReal[inBoundsBools] = np.arccos(outReal[inBoundsBools]) #calculate arccos, there are 2 potential solutions... need to calculate both
    if np.any(np.all(np.isnan(nuReal),axis=1)): #one of the planets has no solutions... this is a problem
        myInd = np.where(np.all(np.isnan(nuReal),axis=1))[0]
        #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
    gPhi = (1.+np.sin(np.tile(inc,(8,1)).T)*np.sin(nuReal+np.tile(w,(8,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
    gd = np.tile(a.to('AU'),(8,1)).T*(1.-np.tile(e,(8,1)).T**2.)/(np.tile(e,(8,1)).T*np.cos(nuReal)+1.)
    gdmags = deltaMag(np.tile(p,(8,1)).T,np.tile(Rp.to('AU'),(8,1)).T,gd,gPhi) #calculate dmag of the specified x-value
    del gPhi, gd

    #For arccos in pi-2pi
    nuReal2 = np.ones(outReal.shape)*np.nan
    nuReal2[inBoundsBools] = 2.*np.pi - nuReal[inBoundsBools]
    gPhi2 = (1.+np.sin(np.tile(inc,(8,1)).T)*np.sin(nuReal2+np.tile(w,(8,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
    gd2 = np.tile(a.to('AU'),(8,1)).T*(1.-np.tile(e,(8,1)).T**2.)/(np.tile(e,(8,1)).T*np.cos(nuReal2)+1.)
    gdmags2 = deltaMag(np.tile(p,(8,1)).T,np.tile(Rp.to('AU'),(8,1)).T,gd2,gPhi2) #calculate dmag of the specified x-value
    #del gPhi2, gd2, inBoundsBools, outBoundsBools, out

    # Combining two solution arrays
    nuRealComb = np.concatenate((nuReal,nuReal2),axis=1) #combines the two arrays
    dmagsComb = np.concatenate((gdmags,gdmags2),axis=1) #combines the two arrays

    #Saving some stuff for debugging
    nuRealCombSaved = np.concatenate((nuReal,nuReal2),axis=1) #combines the two arrays
    dmagsCombSaved = np.concatenate((gdmags,gdmags2),axis=1) #combines the two arrays

    #### Extracting All Extrema ####################################################
    #This section belongs here before we being checking for duplicates for the very simple reason that all the solutions we have come from the function provided.
    #By this logic, the largest value must be the maximum and the smallest value must be the minimum
    #Step 1: Find the mindmag (it must be the smallest dmag producing solution of all possible solutions)
    indsOfMin = np.nanargmin(dmagsComb,axis=1) #Find the index of the 16 that is the absolute minimum
    #If error, type myInd and 
    #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
    mindmag = dmagsComb[np.arange(len(a)),indsOfMin].copy() #Assign the minimum to mindmag
    nuMinDmag = nuRealComb[np.arange(len(a)),indsOfMin].copy() #Assign nu of the minimum to nuMinDmag
    indsOfMin = np.mod(indsOfMin,8) #Find the indicies of the minimum and complements in 0-7
    indsOfMin2 = indsOfMin+8 #Find the indicies of the minimum and the complements in 8-15
    indsWhereMinCloseTo0 = np.where(np.abs(nuMinDmag) < 0.1)[0]
    indsWhereMinCloseTo2pi = np.where(np.abs(nuMinDmag - 2.*np.pi) < 0.1)[0]
    #Step 2: Find the maxdmag (it must be the largest dmag producing solution of all possible solutions)
    indsOfMax = np.nanargmax(dmagsComb,axis=1) #Find the index of the 16 that is the absolute maximum
    maxdmag = dmagsComb[np.arange(len(a)),indsOfMax].copy() #Assign the maximum to maxdmag
    nuMaxDmag = nuRealComb[np.arange(len(a)),indsOfMax].copy() #Assign nu of the maximum to nuMaxDmag
    indsOfMax = np.mod(indsOfMax,8) #Find the indicies of the minimum and complements in 0-7
    indsOfMax2 = indsOfMax+8 #Find the indicies of the minimum and the complements in 8-15
    indsWhereMaxCloseTo0 = np.where(np.abs(nuMaxDmag) < 0.1)[0]
    indsWhereMaxCloseTo2pi = np.where(np.abs(nuMaxDmag - 2.*np.pi) < 0.1)[0]
    #Step 3: Nan minimum and maximum extrema values
    dmagsComb[np.arange(len(a)),indsOfMin] = np.nan
    dmagsComb[np.arange(len(a)),indsOfMin2] = np.nan
    nuRealComb[np.arange(len(a)),indsOfMin] = np.nan
    nuRealComb[np.arange(len(a)),indsOfMin2] = np.nan
    dmagsComb[np.arange(len(a)),indsOfMax] = np.nan
    dmagsComb[np.arange(len(a)),indsOfMax2] = np.nan
    nuRealComb[np.arange(len(a)),indsOfMax] = np.nan
    nuRealComb[np.arange(len(a)),indsOfMax2] = np.nan
    ###############################################################################

    #### Number of Solutions NEW METHOD ###########################################
    numSols = np.sum((~np.isnan(dmagsComb)).astype('int'),axis=1) # calculate the total number of solutions available
    #1 numSols must be even
    assert np.all(np.logical_not(np.mod(numSols,1).astype('bool'))), 'Uneven number of solutions grrr'
    #2 numSols 
    indsWith2 = np.where(np.logical_or(numSols == 0,numSols == 2))[0]
    indsWith4 = np.where(numSols == 4)[0]
    #assert len(indsWith2) + len(indsWith4) == len(a), 'number of inds does not sum to number of planets'
    if not len(indsWith2) + len(indsWith4) == len(a): #Need to Cleanse the solutions with too many potential solutions
        offendingInds = np.where(np.logical_not(np.logical_or(np.logical_or(numSols == 0,numSols == 2),numSols==4)))[0]
        if np.all(numSols[offendingInds] > 2): #All of these planets have more than 2 offending inds
            #add to indsWith2
            indsWith2 = np.append(indsWith2,offendingInds)
            indsWith2 = np.sort(indsWith2)
    indsWith4Saved = indsWith4
    ###############################################################################

    #### Remove All Duplicate min or max
    #nuMinError = np.abs(nuRealComb - np.tile(nuMinDmag,16))
    #nuMaxError = np.abs(nuRealComb - np.tile(nuMaxDmag,16))
    nuMinDeltaBools = np.abs(nuRealComb - np.tile(nuMinDmag,(16,1)).T) < 0.1 #This is the allowable difference between local minimum and minimum
    nuMaxDeltaBools = np.abs(nuRealComb - np.tile(nuMaxDmag,(16,1)).T) < 0.1 #This is the allowable difference between local maximum and maximum 
    #0.03 should be fine but I really want to nuke this thing so I am using 0.1
    nuRealComb[nuMinDeltaBools] = np.nan
    dmagsComb[nuMinDeltaBools] = np.nan
    nuRealComb[nuMaxDeltaBools] = np.nan
    dmagsComb[nuMaxDeltaBools] = np.nan
    # Remove Duplicate Min or Max close to 0 or 2pi
    dmagsComb[indsWhereMinCloseTo0][nuRealComb[indsWhereMinCloseTo0] > 2.*np.pi - 0.1] = np.nan
    dmagsComb[indsWhereMinCloseTo2pi][nuRealComb[indsWhereMinCloseTo2pi] < 0.1] = np.nan
    dmagsComb[indsWhereMaxCloseTo0][nuRealComb[indsWhereMaxCloseTo0] > 2.*np.pi - 0.1] = np.nan
    dmagsComb[indsWhereMaxCloseTo2pi][nuRealComb[indsWhereMaxCloseTo2pi] < 0.1] = np.nan
    nuRealComb[indsWhereMinCloseTo0][nuRealComb[indsWhereMinCloseTo0] > 2.*np.pi - 0.1] = np.nan
    nuRealComb[indsWhereMinCloseTo2pi][nuRealComb[indsWhereMinCloseTo2pi] < 0.1] = np.nan
    nuRealComb[indsWhereMaxCloseTo0][nuRealComb[indsWhereMaxCloseTo0] > 2.*np.pi - 0.1] = np.nan
    nuRealComb[indsWhereMaxCloseTo2pi][nuRealComb[indsWhereMaxCloseTo2pi] < 0.1] = np.nan
    ####

    #### Remove All Duplicates
    # nuRealComb = np.concatenate((nuReal,nuReal2),axis=1) #combines the two arrays
    # dmagsComb = np.concatenate((gdmags,gdmags2),axis=1) #combines the two arrays
    #Yes we need to do it this way
    for i in np.arange(nuRealComb.shape[0]):
        rounded = np.round(nuRealComb[i],4) #Note: the highest accuracy is +/-0.0014
        tmpdmags = dmagsComb[i]
        theSet = set() #Create an empty set
        theSet.add(np.round(nuMaxDmag[i],4)) #will nan all max values
        if np.abs(nuMaxDmag[i]) <= 0.015: #Add complement at 2pi
            theSet.add(np.round(2.*np.pi,4))
            theSet.add(np.round(2.*np.pi-nuMaxDmag[i],4))
        elif np.abs(nuMaxDmag[i] - 2.*np.pi) <= 0.015: #Add complement at 0
            theSet.add(0.)
            theSet.add(np.round(np.abs(nuMaxDmag[i]),4))
        theSet.add(np.round(nuMinDmag[i],4)) #will nan all min values
        if np.abs(nuMinDmag[i]) <= 0.015: #Add complement at 0
            theSet.add(np.round(2.*np.pi,4))
            theSet.add(np.round(2.*np.pi-nuMinDmag[i],4))
        elif np.abs(2.*np.pi - nuMinDmag[i]) <= 0.015: #Add complement at 2pi
            theSet.add(0.)
            theSet.add(np.round(np.abs(nuMinDmag[i]),4))
            
        for j in np.arange(len(rounded)): #Iterate through each value in the array
            # if np.abs(nuRealComb[i,j] - nuMinDmag[j]) <= 0.03 or np.abs(nuRealComb[i,j] - nuMaxDmag[j]) <= 0.03:
            #     rounded[j] = np.nan #Replace the value with nan
            #     tmpdmags[j] = np.nan
            #     #print('Naning ind: ' + str(i) + ' nuRealComb:' + str(nuRealComb[i]))
            if not (rounded[j] in theSet) and not np.isnan(rounded[j]): #Check if it does not occur in the set of occurrences
                theSet.add(rounded[j]) #If it doesn't exist, add it
                #Keep the value the same
            else:
                rounded[j] = np.nan #Replace the value with nan
                tmpdmags[j] = np.nan
        nuRealComb[i] = rounded
        dmagsComb[i] = tmpdmags
    nuReal = nuRealComb[:,0:8]
    nuReal2 = nuRealComb[:,8:16]
    gdmags = dmagsComb[:,0:8]
    gdmags2 = dmagsComb[:,8:16]

    # #Just checking if second derivatives can be used (result they can't)
    # secondDerivatives = calc_planet_dmagSecondDerivative(e[indsWith4],inc[indsWith4],w[indsWith4],nuRealComb[indsWith4])
    # realLocalMinBool5 = (secondDerivatives > 0)*np.logical_not(np.isnan(secondDerivatives))
    # realLocalMaxBool5 = (secondDerivatives < 0)*np.logical_not(np.isnan(secondDerivatives))
    # numLocalMin = np.sum(realLocalMinBool5.astype('int'),axis=1)
    # numLocalMax = np.sum(realLocalMaxBool5.astype('int'),axis=1)

    #MIGHT BE ABLE TO SPEED THIS UP BY USING THE 2ND DERIVATIVE OF DMAG(NU) BUT THERE IS INHERENT RISK IN DOING SO BECAUSE THE SOLUTIONS THEMSELVES MIGHT NEED CODDLING
    #### Check if the solution is a local extrema a potential saddle-point
    #Of the remaining 4 possible values that local min and local max could take, we will need to manually verify whether each point is a local minimum or local maximum by 
    #taking the original solution and adding/subtracting dnu (nu+dnu)
    nuRealpABIT = nuReal + 5e-2 #1e-10 is too small to make a difference, so is 1e-6, 1e-4 is too small, 1e-3 does pretty well
    nuRealmABIT = nuReal - 5e-2
    nuReal2pABIT = nuReal2 + 5e-2
    nuReal2mABIT = nuReal2 - 5e-2
    #### Calculate the associated dmags
    phi = (1.+np.sin(np.tile(inc,(8,1)).T)*np.sin(nuRealpABIT+np.tile(w,(8,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
    d = np.tile(a.to('AU'),(8,1)).T*(1.-np.tile(e,(8,1)).T**2.)/(np.tile(e,(8,1)).T*np.cos(nuRealpABIT)+1.)
    dmagpABIT = deltaMag(np.tile(p,(8,1)).T,np.tile(Rp.to('AU'),(8,1)).T,d,phi) #calculate dmag of the specified x-value
    phi = (1.+np.sin(np.tile(inc,(8,1)).T)*np.sin(nuRealmABIT+np.tile(w,(8,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
    d = np.tile(a.to('AU'),(8,1)).T*(1.-np.tile(e,(8,1)).T**2.)/(np.tile(e,(8,1)).T*np.cos(nuRealmABIT)+1.)
    dmagmABIT = deltaMag(np.tile(p,(8,1)).T,np.tile(Rp.to('AU'),(8,1)).T,d,phi) #calculate dmag of the specified x-value
    phi = (1.+np.sin(np.tile(inc,(8,1)).T)*np.sin(nuReal2pABIT+np.tile(w,(8,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
    d = np.tile(a.to('AU'),(8,1)).T*(1.-np.tile(e,(8,1)).T**2.)/(np.tile(e,(8,1)).T*np.cos(nuReal2pABIT)+1.)
    dmag2pABIT = deltaMag(np.tile(p,(8,1)).T,np.tile(Rp.to('AU'),(8,1)).T,d,phi) #calculate dmag of the specified x-value
    phi = (1.+np.sin(np.tile(inc,(8,1)).T)*np.sin(nuReal2mABIT+np.tile(w,(8,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
    d = np.tile(a.to('AU'),(8,1)).T*(1.-np.tile(e,(8,1)).T**2.)/(np.tile(e,(8,1)).T*np.cos(nuReal2mABIT)+1.)
    dmag2mABIT = deltaMag(np.tile(p,(8,1)).T,np.tile(Rp.to('AU'),(8,1)).T,d,phi) #calculate dmag of the specified x-value
    #### Now do the boolean Comparisons. Can this point be a local min or local max
    realLocalMinBool = (dmagpABIT > gdmags)*(dmagmABIT > gdmags)#*~np.isnan(gdmags[indsWith4])
    realLocalMaxBool = (dmagpABIT < gdmags)*(dmagmABIT < gdmags)#*~np.isnan(gdmags[indsWith4])
    real2LocalMinBool = (dmag2pABIT > gdmags2)*(dmag2mABIT > gdmags2)#*~np.isnan(gdmags[indsWith4])
    real2LocalMaxBool = (dmag2pABIT < gdmags2)*(dmag2mABIT < gdmags2)#*~np.isnan(gdmags[indsWith4])

    ####NAN solutions that are not a min or max extrema (there are many solutions that are not extrema)
    isNotExtrema = np.logical_not(np.logical_or(realLocalMinBool,realLocalMaxBool)) #value is either local minimum or local maximum
    isNotExtrema2 = np.logical_not(np.logical_or(real2LocalMinBool,real2LocalMaxBool)) #value is either local minimum or local maximum
    nuReal[isNotExtrema] = np.nan
    gdmags[isNotExtrema] = np.nan
    nuReal2[isNotExtrema2] = np.nan
    gdmags2[isNotExtrema2] = np.nan

    # Why second derivatives can't be used
    # #Need to do a quick check: Is does the naning of isNotExtrema do anything?
    # assert np.isnan(nuRealComb) == np.logical_not(np.logical_or(realLocalMinBool5,realLocalMaxBool5)), 'not equivalent methods, extrema sorting necessary'

    #Create arrays containing all local min and max nus and dmags
    nulminAll = np.zeros(len(indsWith4))
    nulmaxAll = np.zeros(len(indsWith4))
    dmaglminAll = np.zeros(len(indsWith4))
    dmaglmaxAll = np.zeros(len(indsWith4))

    #Calculate the number of local minimum and ensure it sums properly
    numLocalMin = np.sum(realLocalMinBool[indsWith4].astype('int'),axis=1)
    numLocalMin2 = np.sum(real2LocalMinBool[indsWith4].astype('int'),axis=1)
    indsWhere1LocalMinInnuReal = np.where(np.sum(realLocalMinBool[indsWith4].astype('int'),axis=1)==1)[0]
    indsWhere1LocalMinInnuReal2 = np.where(np.sum(real2LocalMinBool[indsWith4].astype('int'),axis=1)==1)[0]
    nulminAll[indsWhere1LocalMinInnuReal] = nuReal[indsWith4[indsWhere1LocalMinInnuReal]][realLocalMinBool[indsWith4[indsWhere1LocalMinInnuReal]]] #THIS SEEMS TO WORK???
    nulminAll[indsWhere1LocalMinInnuReal2] = nuReal2[indsWith4[indsWhere1LocalMinInnuReal2]][real2LocalMinBool[indsWith4[indsWhere1LocalMinInnuReal2]]] #THIS SEEMS TO WORK???
    dmaglminAll[indsWhere1LocalMinInnuReal] = gdmags[indsWith4[indsWhere1LocalMinInnuReal]][realLocalMinBool[indsWith4[indsWhere1LocalMinInnuReal]]] #THIS SEEMS TO WORK???
    dmaglminAll[indsWhere1LocalMinInnuReal2] = gdmags2[indsWith4[indsWhere1LocalMinInnuReal2]][real2LocalMinBool[indsWith4[indsWhere1LocalMinInnuReal2]]] #THIS SEEMS TO WORK???
    nuReal[indsWith4[indsWhere1LocalMinInnuReal]][realLocalMinBool[indsWith4[indsWhere1LocalMinInnuReal]]] = np.nan #Set all the nuReal values to nan
    nuReal2[indsWith4[indsWhere1LocalMinInnuReal2]][real2LocalMinBool[indsWith4[indsWhere1LocalMinInnuReal2]]] = np.nan #Set all the nuReal2 values to nan
    gdmags[indsWith4[indsWhere1LocalMinInnuReal]][realLocalMinBool[indsWith4[indsWhere1LocalMinInnuReal]]] = np.nan #Set all the nuReal values to nan
    gdmags2[indsWith4[indsWhere1LocalMinInnuReal2]][real2LocalMinBool[indsWith4[indsWhere1LocalMinInnuReal2]]] = np.nan #Set all the nuReal2 values to nan

    indsWhere1LocalMin = np.where(numLocalMin+numLocalMin2==1)[0]
    indsWhere2LocalMin = np.where(numLocalMin+numLocalMin2==2)[0]
    indsWhere3LocalMin = np.where(numLocalMin+numLocalMin2==3)[0]
    assert np.all(np.isnan(nuReal2[indsWith4[indsWhere1LocalMin]][realLocalMinBool[indsWith4[indsWhere1LocalMin]]])) #solong as this is true, the immediately above works

    numLocalMax = np.sum(realLocalMaxBool[indsWith4].astype('int'),axis=1)
    numLocalMax2 = np.sum(real2LocalMaxBool[indsWith4].astype('int'),axis=1)
    indsWhere1LocalMaxInnuReal = np.where(np.sum(realLocalMaxBool[indsWith4].astype('int'),axis=1)==1)[0]
    indsWhere1LocalMaxInnuReal2 = np.where(np.sum(real2LocalMaxBool[indsWith4].astype('int'),axis=1)==1)[0]
    assert np.all(np.sum(realLocalMaxBool[indsWith4[indsWhere1LocalMaxInnuReal]],axis=1)==1), 'whoops it looks like at least 1 planet has 2 local max'
    nulmaxAll[indsWhere1LocalMaxInnuReal] = nuReal[indsWith4[indsWhere1LocalMaxInnuReal]][realLocalMaxBool[indsWith4[indsWhere1LocalMaxInnuReal]]] #THIS SEEMS TO WORK???
    nulmaxAll[indsWhere1LocalMaxInnuReal2] = nuReal2[indsWith4[indsWhere1LocalMaxInnuReal2]][real2LocalMaxBool[indsWith4[indsWhere1LocalMaxInnuReal2]]] #THIS SEEMS TO WORK???
    dmaglmaxAll[indsWhere1LocalMaxInnuReal] = gdmags[indsWith4[indsWhere1LocalMaxInnuReal]][realLocalMaxBool[indsWith4[indsWhere1LocalMaxInnuReal]]] #THIS SEEMS TO WORK???
    dmaglmaxAll[indsWhere1LocalMaxInnuReal2] = gdmags2[indsWith4[indsWhere1LocalMaxInnuReal2]][real2LocalMaxBool[indsWith4[indsWhere1LocalMaxInnuReal2]]] #THIS SEEMS TO WORK???
    nuReal[indsWith4[indsWhere1LocalMaxInnuReal]][realLocalMaxBool[indsWith4[indsWhere1LocalMaxInnuReal]]] = np.nan
    nuReal2[indsWith4[indsWhere1LocalMaxInnuReal2]][real2LocalMaxBool[indsWith4[indsWhere1LocalMaxInnuReal2]]] = np.nan
    gdmags[indsWith4[indsWhere1LocalMaxInnuReal]][realLocalMaxBool[indsWith4[indsWhere1LocalMaxInnuReal]]] = np.nan
    gdmags2[indsWith4[indsWhere1LocalMaxInnuReal2]][real2LocalMaxBool[indsWith4[indsWhere1LocalMaxInnuReal2]]] = np.nan

    indsWhere1LocalMax = np.where(numLocalMax+numLocalMax2==1)[0]
    indsWhere2LocalMax = np.where(numLocalMax+numLocalMax2==2)[0]
    indsWhere3LocalMax = np.where(numLocalMax+numLocalMax2==3)[0]

    # #Here, nuReal[indsWith4[indswhere2LocalMin]] needs to be iterated over and checked against nuMinDmag if its solutions are within idk 1e-2?
    # #Nuke the ones that are close and their complements
    # nuRealComb = np.concatenate((nuReal,nuReal2),axis=1) #combines the two arrays
    # dmagsComb = np.concatenate((gdmags,gdmags2),axis=1) #combines the two arrays
    # for i in np.arange(len(indsWhere2LocalMin)):
    #     offendingInd = np.where(np.abs(nuRealComb[indsWith4[indsWhere2LocalMin[i]]] - nuMinDmag[indsWith4[indsWhere2LocalMin[i]]]) < 2e-3)[0]
    #     for j in np.arange(len(offendingInd)):
    #         nuRealComb[indsWith4[indsWhere2LocalMin[i]],offendingInd[j]] = np.nan
    #         dmagsComb[indsWith4[indsWhere2LocalMin[i]],offendingInd[j]] = np.nan

    #     if np.abs(nuMinDmag[indsWith4[indsWhere2LocalMin[i]]]) <= 0.015: #Add complement at 2pi
    #         offendingInd3 = np.where(np.abs(2.*np.pi - nuRealComb[indsWith4[indsWhere2LocalMin[i]]]) < 0.015)[0]
    #         for j in np.arange(len(offendingInd3)):
    #             nuRealComb[indsWith4[indsWhere2LocalMin[i]],offendingInd3[j]] = np.nan
    #             dmagsComb[indsWith4[indsWhere2LocalMin[i]],offendingInd3[j]] = np.nan
    #     if np.abs(2.*np.pi - nuMinDmag[indsWith4[indsWhere2LocalMin[i]]]) <= 0.015: #Add complement at 2pi
    #         offendingInd4 = np.where(np.abs(nuRealComb[indsWith4[indsWhere2LocalMin[i]]]) < 0.015)[0]
    #         for j in np.arange(len(offendingInd4)):
    #             nuRealComb[indsWith4[indsWhere2LocalMin[i]],offendingInd4[j]] = np.nan
    #             dmagsComb[indsWith4[indsWhere2LocalMin[i]],offendingInd4[j]] = np.nan
    #     #offendingInd2 = np.where(np.abs(nuReal2[indsWith4[indsWhere2LocalMin[i]]] - nuMinDmag[indsWith4[indsWhere2LocalMin[i]]]) < 2e-3)[0]
    #     # for j in np.arange(len(offendingInd)):
    #     #     nuReal[indsWith4[indsWhere2LocalMin[i]],offendingInd[j]] = np.nan
    #     # for j in np.arange(len(offendingInd2)):
    #     #     nuReal2[indsWith4[indsWhere2LocalMin[i]],offendingInd2[j]] = np.nan
    #     #DELETE
    #     # if len(offendingInd) == 1:
    #     #     nuReal[indsWith4[indsWhere2LocalMin[i]],offendingInd] = np.nan
    #     # if len(offendingInd2) == 1:
    #     #     nuReal2[indsWith4[indsWhere2LocalMin[i]],offendingInd2] = np.nan

    # #Here, nuReal[indsWith4[indswhere2LocalMax]] needs to be iterated over and checked against nuMaxDmag if its solutions are within idk 1e-2?
    # #Nuke the ones that are close and their complements
    # for i in np.arange(len(indsWhere2LocalMax)):
    #     offendingInd = np.where(np.abs(nuRealComb[indsWith4[indsWhere2LocalMax[i]]] - nuMaxDmag[indsWith4[indsWhere2LocalMax[i]]]) < 2e-3)[0]
    #     #offendingInd2 = np.where(np.abs(nuReal2[indsWith4[indsWhere2LocalMax[i]]] - nuMaxDmag[indsWith4[indsWhere2LocalMax[i]]]) < 2e-3)[0]
    #     for j in np.arange(len(offendingInd)):
    #         nuRealComb[indsWith4[indsWhere2LocalMax[i]],offendingInd[j]] = np.nan
    #         dmagsComb[indsWith4[indsWhere2LocalMax[i]],offendingInd[j]] = np.nan
    #     # for j in np.arange(len(offendingInd2)):
    #     #     nuReal2[indsWith4[indsWhere2LocalMax[i]],offendingInd2[j]] = np.nan

    #     if np.abs(nuMaxDmag[indsWith4[indsWhere2LocalMax[i]]]) <= 0.015: #Add complement at 2pi
    #         offendingInd3 = np.where(np.abs(2.*np.pi - nuRealComb[indsWith4[indsWhere2LocalMax[i]]]) < 0.015)[0]
    #         for j in np.arange(len(offendingInd3)):
    #             nuRealComb[indsWith4[indsWhere2LocalMax[i]],offendingInd3[j]] = np.nan
    #             dmagsComb[indsWith4[indsWhere2LocalMax[i]],offendingInd3[j]] = np.nan
    #     if np.abs(2.*np.pi - nuMaxDmag[indsWith4[indsWhere2LocalMax[i]]]) <= 0.015: #Add complement at 2pi
    #         offendingInd4 = np.where(np.abs(nuRealComb[indsWith4[indsWhere2LocalMax[i]]]) < 0.015)[0]
    #         for j in np.arange(len(offendingInd4)):
    #             nuRealComb[indsWith4[indsWhere2LocalMax[i]],offendingInd4[j]] = np.nan
    #             dmagsComb[indsWith4[indsWhere2LocalMax[i]],offendingInd4[j]] = np.nan
    #     #DELETE
    #     # if len(offendingInd) == 1:
    #     #     nuReal[indsWith4[indsWhere2LocalMax[i]],offendingInd] = np.nan
    #     # if len(offendingInd2) == 1:
    #     #     nuReal2[indsWith4[indsWhere2LocalMax[i]],offendingInd2] = np.nan
    # nuReal = nuRealComb[:,0:8]
    # nuReal2 = nuRealComb[:,8:16]
    # gdmags = dmagsComb[:,0:8]
    # gdmags2 = dmagsComb[:,8:16]



    #### Remove Residual Duplicates from indsWhere3LocalMin
    nuRealComb = np.concatenate((nuReal,nuReal2),axis=1) #combines the two arrays
    dmagsComb = np.concatenate((gdmags,gdmags2),axis=1) #combines the two arrays
    #Yes we need to do it this way
    for i in np.arange(indsWhere3LocalMin.shape[0]):
        rounded = np.round(nuRealComb[indsWith4[indsWhere3LocalMin[i]]],2) #Note: the highest accuracy is +/-0.0014
        tmpdmags = dmagsComb[indsWith4[indsWhere3LocalMin[i]]]
        theSet = set() #Create an empty set
        theSet.add(np.round(nuMaxDmag[indsWith4[indsWhere3LocalMin[i]]],2)) #will nan all max values
        if np.abs(nuMaxDmag[indsWith4[indsWhere3LocalMin[i]]]) <= 0.015: #Add complement at 2pi
            theSet.add(np.round(2.*np.pi,2))
            theSet.add(np.round(2.*np.pi-nuMaxDmag[indsWith4[indsWhere3LocalMin[i]]],2))
        elif np.abs(2.*np.pi - nuMaxDmag[indsWith4[indsWhere3LocalMin[i]]]) <= 0.015: #Add complement at 0
            theSet.add(0.)
            theSet.add(np.round(nuMaxDmag[indsWith4[indsWhere3LocalMin[i]]],2))
        theSet.add(np.round(nuMinDmag[indsWith4[indsWhere3LocalMin[i]]],2)) #will nan all min values
        if np.abs(nuMinDmag[indsWith4[indsWhere3LocalMin[i]]]) <= 0.015: #Add complement at 0
            theSet.add(np.round(2.*np.pi,2))
            theSet.add(np.round(2.*np.pi-nuMinDmag[indsWith4[indsWhere3LocalMin[i]]],2))
        elif np.abs(2.*np.pi - nuMinDmag[indsWith4[indsWhere3LocalMin[i]]]) <= 0.015: #Add complement at 2pi
            theSet.add(0.)
            theSet.add(np.round(nuMinDmag[indsWith4[indsWhere3LocalMin[i]]],2))
        for j in np.arange(len(rounded)): #Iterate through each value in the array
            if not (rounded[j] in theSet) and not np.isnan(rounded[j]): #Check if it does not occur in the set of occurrences
                theSet.add(rounded[j]) #If it doesn't exist, add it
            else:
                rounded[j] = np.nan #Replace the value with nan
                tmpdmags[j] = np.nan
        nuRealComb[indsWith4[indsWhere3LocalMin[i]]] = rounded
        dmagsComb[indsWith4[indsWhere3LocalMin[i]]] = tmpdmags

    #### Remove Residual Duplicates from indsWhere3LocalMax
    # nuRealComb = np.concatenate((nuReal,nuReal2),axis=1) #combines the two arrays
    # dmagsComb = np.concatenate((gdmags,gdmags2),axis=1) #combines the two arrays
    #Yes we need to do it this way
    for i in np.arange(indsWhere3LocalMax.shape[0]):
        rounded = np.round(nuRealComb[indsWith4[indsWhere3LocalMax[i]]],2) #Note: the highest accuracy is +/-0.0014
        tmpdmags = dmagsComb[indsWith4[indsWhere3LocalMax[i]]]
        theSet = set() #Create an empty set
        theSet.add(np.round(nuMaxDmag[indsWith4[indsWhere3LocalMax[i]]],2)) #will nan all max values
        if np.abs(nuMaxDmag[indsWith4[indsWhere3LocalMax[i]]]) <= 0.015: #Add complement at 2pi
            theSet.add(np.round(2.*np.pi,2))
            theSet.add(np.round(2.*np.pi-nuMaxDmag[indsWith4[indsWhere3LocalMax[i]]],2))
        elif np.abs(2.*np.pi - nuMaxDmag[indsWith4[indsWhere3LocalMax[i]]]) <= 0.015: #Add complement at 0
            theSet.add(0.)
            theSet.add(np.round(nuMaxDmag[indsWith4[indsWhere3LocalMax[i]]],2))
        theSet.add(np.round(nuMinDmag[indsWith4[indsWhere3LocalMax[i]]],2)) #will nan all min values
        if np.abs(nuMinDmag[indsWith4[indsWhere3LocalMax[i]]]) <= 0.015: #Add complement at 0
            theSet.add(np.round(2.*np.pi,2))
            theSet.add(np.round(2.*np.pi-nuMinDmag[indsWith4[indsWhere3LocalMax[i]]],2))
        elif np.abs(2.*np.pi - nuMinDmag[indsWith4[indsWhere3LocalMax[i]]]) <= 0.015: #Add complement at 2pi
            theSet.add(0.)
            theSet.add(np.round(nuMinDmag[indsWith4[indsWhere3LocalMax[i]]],2))
        for j in np.arange(len(rounded)): #Iterate through each value in the array
            if not (rounded[j] in theSet) and not np.isnan(rounded[j]): #Check if it does not occur in the set of occurrences
                theSet.add(rounded[j]) #If it doesn't exist, add it
            else:
                rounded[j] = np.nan #Replace the value with nan
                tmpdmags[j] = np.nan
        nuRealComb[indsWith4[indsWhere3LocalMax[i]]] = rounded
        dmagsComb[indsWith4[indsWhere3LocalMax[i]]] = tmpdmags
    nuReal = nuRealComb[:,0:8]
    nuReal2 = nuRealComb[:,8:16]
    gdmags = dmagsComb[:,0:8]
    gdmags2 = dmagsComb[:,8:16]


    #DELETE THE FOLLOWING CODE. IT DOES NOT MAKE SENSE
    #If all the nuReal of a solution are 0, then we just pick the smallest
    indsWhereAllNan0 = np.where(~np.all(np.isnan(nuReal[indsWith4[indsWhere2LocalMin]]),axis=1))[0] #Finds all the inds of indsWhere2LocalMin where nuReal contains at least 1 solution
    indsWhereAllNan1 = np.where(np.all(np.isnan(nuReal2[indsWith4[indsWhere2LocalMin]]),axis=1))[0] #Finds all the inds of indsWhere2LocalMin where nuReal2 contains no solutions
    indsWhereAllNan2 = np.where(~np.all(np.isnan(nuReal[indsWith4[indsWhere2LocalMax]]),axis=1))[0] #Finds all the inds of indsWhere2LocalMax where nuReal contains at least 1 solution
    indsWhereAllNan3 = np.where(np.all(np.isnan(nuReal2[indsWith4[indsWhere2LocalMax]]),axis=1))[0] #Finds all the inds of indsWhere2LocalMax where nuReal2 contains no solutions
    #Skip if 0 (Note in testing these were all 0. Not sure if this code is actually being executed currently)
    if len(indsWhereAllNan0) > 0:
        nulminAll[indsWhere2LocalMin[indsWhereAllNan0]] = nuReal[indsWith4[indsWhere2LocalMin[indsWhereAllNan0]],np.nanargmin(gdmags[indsWith4[indsWhere2LocalMin[indsWhereAllNan0]]],axis=1)]
        dmaglminAll[indsWhere2LocalMin[indsWhereAllNan0]] = gdmags[indsWith4[indsWhere2LocalMin[indsWhereAllNan0]],np.nanargmin(gdmags[indsWith4[indsWhere2LocalMin[indsWhereAllNan0]]],axis=1)]
    if len(indsWhereAllNan1) > 0:
        nulminAll[indsWhere2LocalMin[indsWhereAllNan1]] = np.nan #nuReal2[indsWith4[indsWhere2LocalMin[indsWhereAllNan1]],np.nanargmin(gdmags2[indsWith4[indsWhere2LocalMax[indsWhereAllNan1]]],axis=1)]
        dmaglminAll[indsWhere2LocalMin[indsWhereAllNan1]] = np.nan #gdmags2[indsWith4[indsWhere2LocalMin[indsWhereAllNan1]],np.nanargmin(gdmags2[indsWith4[indsWhere2LocalMax[indsWhereAllNan1]]],axis=1)]
    if len(indsWhereAllNan2) > 0:
        nulmaxAll[indsWhere2LocalMax[indsWhereAllNan2]] = nuReal[indsWith4[indsWhere2LocalMax[indsWhereAllNan2]],np.nanargmax(gdmags[indsWith4[indsWhere2LocalMax[indsWhereAllNan2]]],axis=1)]
        dmaglmaxAll[indsWhere2LocalMax[indsWhereAllNan2]] = gdmags[indsWith4[indsWhere2LocalMax[indsWhereAllNan2]],np.nanargmax(gdmags[indsWith4[indsWhere2LocalMax[indsWhereAllNan2]]],axis=1)]
    if len(indsWhereAllNan3) > 0:
        nulmaxAll[indsWhere2LocalMax[indsWhereAllNan3]] = np.nan #nuReal2[indsWith4[indsWhere2LocalMax[indsWhereAllNan3]],np.nanargmax(gdmags2[indsWith4[indsWhere2LocalMax[indsWhereAllNan3]]],axis=1)]
        dmaglmaxAll[indsWhere2LocalMax[indsWhereAllNan3]] = np.nan #nuReal2[indsWith4[indsWhere2LocalMax[indsWhereAllNan3]],np.nanargmax(gdmags2[indsWith4[indsWhere2LocalMax[indsWhereAllNan3]]],axis=1)]

    #CAN SIMPLIFY THIS BIT HERE
    #### Remove any planets from indsWith4 if they have 0 at this point, also add the to indsWith2. Screw it
    indsWhere0dmaglmin = np.where(dmaglminAll==0.)[0]
    for ind in -np.sort(-indsWhere0dmaglmin):
        dmaglminAll = np.delete(dmaglminAll, ind)
        dmaglmaxAll = np.delete(dmaglmaxAll, ind)
        nulminAll = np.delete(nulminAll, ind)
        nulmaxAll = np.delete(nulmaxAll, ind)
        indsWith2 = np.append(indsWith2, ind)
        indsWith4 = np.delete(indsWith4, ind)
    indsWhere0dmaglmax = np.where(dmaglmaxAll==0.)[0]
    for ind in -np.sort(-indsWhere0dmaglmax):
        dmaglminAll = np.delete(dmaglminAll, ind)
        dmaglmaxAll = np.delete(dmaglmaxAll, ind)
        nulminAll = np.delete(nulminAll, ind)
        nulmaxAll = np.delete(nulmaxAll, ind)
        indsWith2 = np.append(indsWith2, ind)
        indsWith4 = np.delete(indsWith4, ind)
    #### Remove any planets from indsWith4 if they are nan at this point, also add the to indsWith2. Screw it
    indsWhere0dmaglmin = np.where(np.isnan(dmaglminAll))[0]
    for ind in -np.sort(-indsWhere0dmaglmin):
        dmaglminAll = np.delete(dmaglminAll, ind)
        dmaglmaxAll = np.delete(dmaglmaxAll, ind)
        nulminAll = np.delete(nulminAll, ind)
        nulmaxAll = np.delete(nulmaxAll, ind)
        indsWith2 = np.append(indsWith2, ind)
        indsWith4 = np.delete(indsWith4, ind)
    indsWhere0dmaglmax = np.where(np.isnan(dmaglmaxAll))[0]
    for ind in -np.sort(-indsWhere0dmaglmax):
        dmaglminAll = np.delete(dmaglminAll, ind)
        dmaglmaxAll = np.delete(dmaglmaxAll, ind)
        nulminAll = np.delete(nulminAll, ind)
        nulmaxAll = np.delete(nulmaxAll, ind)
        indsWith2 = np.append(indsWith2, ind)
        indsWith4 = np.delete(indsWith4, ind)

    #Checking number of solutions is correct
    assert len(indsWith4) == len(dmaglmaxAll), 'number of solutions in dmaglmaxAll does not match number of indicies in indsWith4'
    assert len(indsWith4) == len(dmaglminAll), 'number of solutions in dmaglminAll does not match number of indicies in indsWith4'
    assert len(indsWith4) == len(nulmaxAll), 'number of solutions in dmaglmaxAll does not match number of indicies in indsWith4'
    assert len(indsWith4) == len(nulminAll), 'number of solutions in dmaglminAll does not match number of indicies in indsWith4'
    assert len(indsWith2)+len(indsWith4) == e.shape[0], 'length of indicies does not properly match'

    # #hmmmm.. we must test all scenarios to determine what localmindmag and localmaxdmag are.
    # #It must be true that localmindmag < localmaxdmag

    # count = 0
    # if np.any((dmaglmaxAll > 29.)*(dmaglminAll < 29.)):
    #     inds = np.where((dmaglmaxAll > 29.)*(dmaglminAll < 29.))[0]
    #     import matplotlib.pyplot as plt
    #     for ind in indsWith4[inds]:
    #         plt.figure(num=ind)
    #         nus = np.linspace(start=0,stop=2.*np.pi,num=300)
    #         phis = (1.+np.sin(inc[ind])*np.sin(nus+w[ind]))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
    #         ds = a[ind]*(1.-e[ind]**2.)/(e[ind]*np.cos(nus)+1.)
    #         dmags = deltaMag(p[ind],Rp[ind].to('AU'),ds,phis) #calculate dmag of the specified x-value
    #         plt.plot(nus,dmags,color='black',zorder=10)
    #         plt.scatter(nuMinDmag[ind],mindmag[ind],color='cyan',marker='d',zorder=20)
    #         plt.scatter(nuMaxDmag[ind],maxdmag[ind],color='red',marker='d',zorder=20)
    #         lind = np.where(ind == indsWith4)[0]
    #         plt.scatter(nulminAll[lind],dmaglminAll[lind],color='magenta',marker='d',zorder=20)
    #         plt.scatter(nulmaxAll[lind],dmaglmaxAll[lind],color='gold',marker='d',zorder=20)
    #         plt.show(block=False)
    #         plt.gcf().canvas.draw()
    #         count = count+1
    #         if count > 60:
    #             print(saltyburrito)

    #Simple Quality Checks
    assert np.all(dmaglmaxAll >= dmaglminAll)
    assert np.all(dmaglmaxAll < maxdmag[indsWith4]) #this must be true
    assert np.all(dmaglminAll > mindmag[indsWith4]) #this must be true    

    # print(saltyburrito)

    return mindmag, maxdmag, dmaglminAll, dmaglmaxAll, indsWith2, indsWith4, nuMinDmag, nuMaxDmag, nulminAll, nulmaxAll
#################################################################################################################

def calc_planet_sep_extrema(sma,e,W,w,inc):
    """ A method for calculating the separation extrema and nu where these extrema occur
    Args:
        sma (numpy array):
            semi-major axis
        e (numpy array):
            eccentricity
        W (numpy array):
            Longitude of the ascending nodes
        w (numpy array):
            Argument of periapsis
        inc (numpy array):
            inclination
    Returns:
        nu_minSepPoints (numpy array):
        nu_maxSepPoints (numpy array):
        nu_lminSepPoints (numpy array):
        nu_lmaxSepPoints (numpy array):
        minSep (numpy array):
        maxSep (numpy array):
        lminSep (numpy array):
        lmaxSep (numpy array):
        yrealAllRealInds (numpy array):
        yrealImagInds (numpy array):
    """
    #### Calculate Projected Ellipse Angles and Minor Axis
    dmajorp, dminorp, theta_OpQ_X, theta_OpQp_X = projected_apbpPsipsi(sma,e,W,w,inc)#dmajorp_v2, dminorp_v2, Psi_v2, psi_v2, Psi, psi,
    #3D Ellipse Center
    Op = projected_Op(sma,e,W,w,inc)

    # Checks
    if not np.all(dmajorp <= sma):
        print("Not all Semi-major axis of the projected ellipse are less than the original 3D ellipse, caused by circular orbits required for circular orbits")
        assert np.all(sma - dmajorp >= -1e-12), "Not all Semi-major axis of the projected ellipse are less than the original 3D ellipse" #required for circular orbits
    assert np.all(dminorp <= dmajorp), "All projected Semi-minor axes are less than all projected semi-major axes"

    #### Derotate Ellipse Calculations
    x, y, Phi = derotatedEllipse(theta_OpQ_X, theta_OpQp_X, Op)
    #x- x coordinates of host star relative to projected ellipse center
    #y- y coordinates of host star relative to projected ellipse center
    #Phi- Angle of projected ellipse semi-major axis from x-axis
    #if plotBool == False: #deletes these angles because they are no longer necessary
    del theta_OpQ_X, theta_OpQp_X
    ####

    #### Calculate X,Y Position of Minimum and Maximums with Quartic
    A, B, C, D = quarticCoefficients_smin_smax_lmin_lmax(dmajorp.astype('complex128'), dminorp, np.abs(x), np.abs(y)) #calculate the quartic solutions to the min-max separation problem
    #xreal, delta, P, D2, R, delta_0 = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
    xreal, _, _, _, _, _ = quarticSolutions_ellipse_to_Quarticipynb(A.astype('complex128'), B, C, D)
    del A, B, C, D #delting for memory efficiency
    assert np.max(np.nanmin(np.abs(np.imag(xreal)),axis=1)) < 1e-5, 'At least one row has min > 1e-5' #this ensures each row has a solution
    #myInd = np.where(np.nanmin(np.abs(np.imag(xreal)),axis=1) > 1e-5)
    #print('ar = ' + str(sma[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + str(W[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
    #print(w[np.argmax(np.nanmin(np.abs(np.imag(xreal)),axis=1))]) #prints the argument of perigee (assert above fails on 1.57 or 1.5*pi)
    #Failure of the above occured where w=4.712 which is approx 1.5pi
    #NOTE: originally 1e-15 but there were some with x=1e-7 and w=pi/2, 5e-6 from 
    xreal.real = np.abs(xreal) #all solutions should be positive

    #### Technically, each row must have at least 2 solutions, but whatever
    yreal = ellipseYFromX(xreal.astype('complex128'), dmajorp, dminorp) #Calculates the y values corresponding to the x values in the first quadrant of an ellipse
    ####

    #### Calculate Minimum, Maximum, Local Minimum, Local Maximum Separations
    minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y, lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
        minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds = smin_smax_slmin_slmax(len(x), xreal, yreal, np.abs(x), np.abs(y), x, y)

    #### Rerotate Extrema Points
    minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr,\
        lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr\
        = rerotateExtremaPoints(minSepPoints_x, minSepPoints_y, maxSepPoints_x, maxSepPoints_y,\
        lminSepPoints_x, lminSepPoints_y, lmaxSepPoints_x, lmaxSepPoints_y,\
        Phi, Op, yrealAllRealInds)

    #### Calculate True Anomalies of Points
    nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints = trueAnomaliesOfSepPoints(minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
    yrealAllRealInds, W, w, inc)


    # nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, nu_fourInt, nu_twoIntSameY, nu_twoIntOppositeX, nu_IntersectionsOnly2\
    #      = trueAnomaliesOfPoints(minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr,\
    #     fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr,\
    #     yrealAllRealInds, fourIntInds, twoIntSameYInds, twoIntOppositeXInds, only2RealInds, W, w, inc)
    del minSepPoints_x_dr, minSepPoints_y_dr, maxSepPoints_x_dr, maxSepPoints_y_dr, lminSepPoints_x_dr, lminSepPoints_y_dr, lmaxSepPoints_x_dr, lmaxSepPoints_y_dr
    #del fourInt_x_dr, fourInt_y_dr, twoIntSameY_x_dr, twoIntSameY_y_dr, twoIntOppositeX_x_dr, twoIntOppositeX_y_dr, xIntersectionsOnly2_dr, yIntersectionsOnly2_dr
    ####

    #### Fix minSep True Anomalies
    nu_minSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_minSepPoints,np.arange(len(sma)),minSep)
    ####
    #### Fix maxSep True Anomalies
    nu_maxSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_maxSepPoints,np.arange(len(sma)),maxSep)
    ####
    #### Fix lminSep True Anomalies
    nu_lminSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_lminSepPoints,yrealAllRealInds,lminSep)
    ####
    #### Fix lmaxSep True Anomalies
    nu_lmaxSepPoints = nuCorrections_extrema(sma,e,W,w,inc,nu_lmaxSepPoints,yrealAllRealInds,lmaxSep)
    ####

    return nu_minSepPoints, nu_maxSepPoints, nu_lminSepPoints, nu_lmaxSepPoints, minSep, maxSep, lminSep, lmaxSep, yrealAllRealInds, yrealImagInds

def solve_dmag_Poly(dmag,e,inc,w,a,p,Rp):
    """
    Args:
        dmag,e,inc,w,a,p,Rp,pInds
    Returns:
        out (numpy array):
            a len(e) by 8 array of cos(x) values
    """
    #Calculate the left hand side (all the things that are separable and constant)
    lhs = 10.**(-0.4*dmag)*(1.-e**2.)**2.*(a.to('AU')/Rp.to('AU')).decompose().value**2./p

    #These are the coefficients starting at cos(nu)^8 
    A = e**4.*np.sin(inc)**4.*np.sin(w)**4./16. + e**4.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./8. + e**4.*np.sin(inc)**4.*np.cos(w)**4./16.
    B = e**4.*np.sin(inc)**3.*np.sin(w)**3./4. + e**4.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4. + e**3.*np.sin(inc)**4.*np.sin(w)**4./4. + e**3.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + e**3.*np.sin(inc)**4.*np.cos(w)**4./4.
    C = -e**4.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./8. - e**4.*np.sin(inc)**4.*np.cos(w)**4./8. + 3.*e**4.*np.sin(inc)**2.*np.sin(w)**2./8. + e**4.*np.sin(inc)**2.*np.cos(w)**2./8. + e**3.*np.sin(inc)**3.*np.sin(w)**3. + e**3.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + 3.*e**2.*np.sin(inc)**4.*np.sin(w)**4./8. + 3.*e**2.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./4. + 3.*e**2.*np.sin(inc)**4.*np.cos(w)**4./8.
    D = -e**4.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4. + e**4.*np.sin(inc)*np.sin(w)/4. - e**3.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. - e**3.*np.sin(inc)**4.*np.cos(w)**4./2. + 3.*e**3.*np.sin(inc)**2.*np.sin(w)**2./2. + e**3.*np.sin(inc)**2.*np.cos(w)**2./2. + 3.*e**2.*np.sin(inc)**3.*np.sin(w)**3./2. + 3.*e**2.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. + e*np.sin(inc)**4.*np.sin(w)**4./4. + e*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. + e*np.sin(inc)**4.*np.cos(w)**4./4.
    E = e**4.*np.sin(inc)**4.*np.cos(w)**4./16. - e**4.*np.sin(inc)**2.*np.cos(w)**2./8. + e**4./16. - e**3.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + e**3.*np.sin(inc)*np.sin(w) - e**2.*lhs*np.sin(inc)**2.*np.sin(w)**2./2. + e**2.*lhs*np.sin(inc)**2.*np.cos(w)**2./2. - 3.*e**2.*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./4. - 3.*e**2.*np.sin(inc)**4.*np.cos(w)**4./4. + 9*e**2.*np.sin(inc)**2.*np.sin(w)**2./4. + 3.*e**2.*np.sin(inc)**2.*np.cos(w)**2./4. + e*np.sin(inc)**3.*np.sin(w)**3. + e*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + np.sin(inc)**4.*np.sin(w)**4./16. + np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./8. + np.sin(inc)**4.*np.cos(w)**4./16.
    F = e**3.*np.sin(inc)**4.*np.cos(w)**4./4. - e**3.*np.sin(inc)**2.*np.cos(w)**2./2. + e**3./4. - e**2.*lhs*np.sin(inc)*np.sin(w) - 3.*e**2.*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./2. + 3.*e**2.*np.sin(inc)*np.sin(w)/2. - e*lhs*np.sin(inc)**2.*np.sin(w)**2. + e*lhs*np.sin(inc)**2.*np.cos(w)**2. - e*np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./2. - e*np.sin(inc)**4.*np.cos(w)**4./2. + 3.*e*np.sin(inc)**2.*np.sin(w)**2./2. + e*np.sin(inc)**2.*np.cos(w)**2./2. + np.sin(inc)**3.*np.sin(w)**3./4. + np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4.
    G = -e**2.*lhs*np.sin(inc)**2.*np.cos(w)**2./2. - e**2.*lhs/2. + 3.*e**2.*np.sin(inc)**4.*np.cos(w)**4./8. - 3.*e**2.*np.sin(inc)**2.*np.cos(w)**2./4. + 3.*e**2./8. - 2*e*lhs*np.sin(inc)*np.sin(w) - e*np.sin(inc)**3.*np.sin(w)*np.cos(w)**2. + e*np.sin(inc)*np.sin(w) - lhs*np.sin(inc)**2.*np.sin(w)**2./2. + lhs*np.sin(inc)**2.*np.cos(w)**2./2. - np.sin(inc)**4.*np.sin(w)**2.*np.cos(w)**2./8. - np.sin(inc)**4.*np.cos(w)**4./8. + 3.*np.sin(inc)**2.*np.sin(w)**2./8. + np.sin(inc)**2.*np.cos(w)**2./8.
    H = -e*lhs*np.sin(inc)**2.*np.cos(w)**2. - e*lhs + e*np.sin(inc)**4.*np.cos(w)**4./4. - e*np.sin(inc)**2.*np.cos(w)**2./2. + e/4. - lhs*np.sin(inc)*np.sin(w) - np.sin(inc)**3.*np.sin(w)*np.cos(w)**2./4. + np.sin(inc)*np.sin(w)/4.
    I = lhs**2. - lhs*np.sin(inc)**2.*np.cos(w)**2./2. - lhs/2. + np.sin(inc)**4.*np.cos(w)**4./16. - np.sin(inc)**2.*np.cos(w)**2./8. + 1/16.
    coeffs = np.asarray([A,B,C,D,E,F,G,H,I])

    #solve for x in the polynomial (where x=cos(nu))
    out = list()
    for i in np.arange(coeffs.shape[1]):
        out.append(np.roots(coeffs[:,i])) # this is x)
    out = np.asarray(out)

    return out

def calc_planetnu_from_dmag(dmag,e,inc,w,a,p,Rp,mindmag, maxdmag, indsWith2Int, indsWith4Int):
    """ This method calculates nu of a planet which have the provided dmag assuming a quasi-lambert phase function fullEqnX from AnalyticalNuFromDmag3.ipynb
    Args:
        dmag (ndarray):
        e (ndarray):
        inc (ndarray):
        w (ndarray):
        a (ndarray):
        p (ndarray):
        Rp (ndarray):
        pInds (ndarray):
    Returns:
        nus2Int, nus4Int, dmag2Int, dmag4Int
    """
    tstart_cos = time.time()
    out2Int = solve_dmag_Poly(dmag,e[indsWith2Int],inc[indsWith2Int],w[indsWith2Int],a[indsWith2Int],p[indsWith2Int],Rp[indsWith2Int])

    #Move the vvalues that are slightly above or below -1 back into -1 to 1 range
    out2Int[(out2Int.real < -1.)*(out2Int.real >= -1.-1e-7)] = -np.ones(out2Int.shape)[(out2Int.real < -1.)*(out2Int.real >= -1.-1e-7)] #move those solutions that might be oh so slightly out of bounds back in bounds
    out2Int[(out2Int.real > 1.)*(out2Int.real <= 1.+1e-7)] = np.ones(out2Int.shape)[(out2Int.real > 1.)*(out2Int.real <= 1.+1e-7)] #move those solutions that might be oh so slightly out of bounds back in bounds
    
    #Throw out roots not in correct bounds
    inBoundsBools2Int = (np.abs(out2Int.imag) <= 1e-7)*(out2Int.real >= -1.)*(out2Int.real <= 1.) #the out2 solutions that are inside of the desired bounds #adding 1e-9 bc a solution that was 1. was filtered out once
    outBoundsBools2Int = np.logical_not(inBoundsBools2Int) # the out2 solutions that are inside the desired bounds
    outReal2Int = np.zeros(out2Int.shape) #just getting something with the right shape
    outReal2Int[outBoundsBools2Int] = np.real(out2Int[outBoundsBools2Int])*np.nan
    #del outBoundsBools2Int
    outReal2Int[inBoundsBools2Int] = np.real(out2Int[inBoundsBools2Int])
    #del out2Int
    outReal2Int = np.real(outReal2Int)

    #Calculate nus
    nuReal2Int = np.ones(outReal2Int.shape)*np.nan
    nuReal2Int[inBoundsBools2Int] = np.arccos(outReal2Int[inBoundsBools2Int]) #calculate arccos, there are 2 potential solutions... need to calculate both
    nuReal22Int = np.ones(outReal2Int.shape)*np.nan
    nuReal22Int[inBoundsBools2Int] = 2.*np.pi - np.arccos(outReal2Int[inBoundsBools2Int])
    nuRealComb2Int = np.concatenate((nuReal2Int,nuReal22Int),axis=1) #combines the two arrays

    #Remove duplicates for inds that have more than 4 real solutions
    #DELETEindsWithMoreThan4RealSols = np.where(np.sum(np.isnan(nuRealComb).astype('int'),axis=1) >= 4)[0]
    #Yes we need to do it this way
    for i in np.arange(nuRealComb2Int.shape[0]):
        rounded = np.round(nuRealComb2Int[i],4) #Note: the highest accuracy is +/-0.0014
        # tmpdmags = dmagsComb[i]
        theSet = set() #Create an empty set
        for j in np.arange(len(rounded)): #Iterate through each value in the array
            if not (rounded[j] in theSet) and not np.isnan(rounded[j]): #Check if it does not occur in the set of occurrences
                theSet.add(rounded[j]) #If it doesn't exist, add it
                if rounded[j] == 0.:
                    theSet.add(np.round(2.*np.pi,4))
                elif rounded[j] == np.round(2.*np.pi,4):
                    theSet.add(0.)
                #Keep the value the same
            else:
                rounded[j] = np.nan #Replace the value with nan
                # tmpdmags[j] = np.nan
        nuRealComb2Int[i] = rounded
        # dmagsComb[i] = tmpdmags
    #nuReal2Int = nuRealComb2Int[:,0:8]
    #nuReal22Int = nuRealComb2Int[:,8:16]
    # gdmags4Int = dmagsComb[:,0:8]
    # gdmags24Int = dmagsComb[:,8:16]
    #Differentiate between inds with 4 non-nans (in combined) and inds with 6 non-nans in combined

    #### Commenting to see if this really matters
    #HEY I DON'T THINK THIS REALLY MATTERS
    # indsWith5NonNan = np.where(np.sum(np.isnan(nuRealComb2Int).astype('int'),axis=1) == 11)[0] #a very rare case
    # indsWith4NonNan = np.where(np.sum(np.isnan(nuRealComb2Int).astype('int'),axis=1) == 12)[0]
    # indsWith3NonNan = np.where(np.sum(np.isnan(nuRealComb2Int).astype('int'),axis=1) == 13)[0]
    # indsWith2NonNan = np.where(np.sum(np.isnan(nuRealComb2Int).astype('int'),axis=1) == 14)[0] # a very rare case
    # if not nuRealComb2Int.shape[0] == len(indsWith4NonNan) + len(indsWith3NonNan) + len(indsWith5NonNan) + len(indsWith2NonNan):
    #     myInd2 = list(set(np.arange(len(nuRealComb2Int)))^set(indsWith4NonNan).union(set(indsWith3NonNan).union(set(indsWith5NonNan).union(set(indsWith2NonNan)))))
    #     for i in np.arange(len(myInd2)):
    #         myInd = indsWith2Int[myInd2[i]]
    #         #print(myInd)
    #         print('ar = ' + str(a[myInd].value) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + 'np.nan'  + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
    #         print('badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)')
    #         with open("/home/dean/Documents/exosims/twoDetMC/badKOE.py","a") as f:
    #             f.write('#calc_planetnu_from_dmag' + '\n')
    #             f.write('ar = ' + str(a[myInd].value) + '*u.AU\ner = ' + str(e[myInd]) + '\nWr = ' + 'np.nan' + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]) + '\n')
    #             f.write('badsma.append(ar.value),bader.append(er),badWr.append(Wr),badwr.append(wr),badinc.append(incr)' + '\n')
    # assert nuRealComb2Int.shape[0] == len(indsWith4NonNan) + len(indsWith3NonNan) + len(indsWith5NonNan) + len(indsWith2NonNan), 'The number of 3 real and 4 real does not sum to the number of nuRealCombs with 2 Intersections'
    #myInd = list(set(np.arange(len(nuRealComb2Int)))^set(indsWith4NonNan).union(set(indsWith3NonNan).union(set(indsWith5NonNan).union(set(indsWith2NonNan)))))
    #myInd2 = indsWith2Int[myInd]      print('ar = ' + str(a[myInd]) + '*u.AU\ner = ' + str(e[myInd]) + '\nwr = ' + str(w[myInd]) + '\nincr = ' + str(inc[myInd]))
    ####

    #Note this if else may cause errors. For some reason, nuRealComb2Int was producing 8 and not 16.. causing me to implement this
    if nuRealComb2Int.shape[1] == 8:
            #For arccos in 0-pi (indicies 0-7) and pi-2pi (indicies 8-15)
        gPhi2Int = (1.+np.sin(np.tile(inc[indsWith2Int],(8,1)).T)*np.sin(nuRealComb2Int+np.tile(w[indsWith2Int],(8,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
        gd2Int = np.tile(a[indsWith2Int].to('AU'),(8,1)).T*(1.-np.tile(e[indsWith2Int],(8,1)).T**2.)/(np.tile(e[indsWith2Int],(8,1)).T*np.cos(nuRealComb2Int)+1.)
        gdmags2IntComb = deltaMag(np.tile(p[indsWith2Int],(8,1)).T,np.tile(Rp[indsWith2Int].to('AU'),(8,1)).T,gd2Int,gPhi2Int) #calculate dmag of the specified x-value

        dmags2IntErrorComb = np.abs(gdmags2IntComb - dmag)
        mindmag2IntErrorInds = np.argsort(dmags2IntErrorComb,axis=1)
        sortedNuReal2IntComb = nuRealComb2Int[np.tile(np.arange(dmags2IntErrorComb.shape[0]),(mindmag2IntErrorInds.shape[1],1)).T,mindmag2IntErrorInds[np.arange(dmags2IntErrorComb.shape[0])]]
        sortedgdmags2IntComb = gdmags2IntComb[np.tile(np.arange(dmags2IntErrorComb.shape[0]),(mindmag2IntErrorInds.shape[1],1)).T,mindmag2IntErrorInds[np.arange(dmags2IntErrorComb.shape[0])]]

        #Create Output Arrays
        nus2Int = np.asarray([sortedNuReal2IntComb[:,0],sortedNuReal2IntComb[:,1]]).T
        dmag2Int = np.asarray([sortedgdmags2IntComb[:,0],sortedgdmags2IntComb[:,1]]).T
    elif nuRealComb2Int.shape[1] == 16:
        #For arccos in 0-pi (indicies 0-7) and pi-2pi (indicies 8-15)
        gPhi2Int = (1.+np.sin(np.tile(inc[indsWith2Int],(16,1)).T)*np.sin(nuRealComb2Int+np.tile(w[indsWith2Int],(16,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
        gd2Int = np.tile(a[indsWith2Int].to('AU'),(16,1)).T*(1.-np.tile(e[indsWith2Int],(16,1)).T**2.)/(np.tile(e[indsWith2Int],(16,1)).T*np.cos(nuRealComb2Int)+1.)
        gdmags2IntComb = deltaMag(np.tile(p[indsWith2Int],(16,1)).T,np.tile(Rp[indsWith2Int].to('AU'),(16,1)).T,gd2Int,gPhi2Int) #calculate dmag of the specified x-value
        #del gPhi2Int, gd2Int
        #For arccos in pi-2pi

        # #For arccos in 0-pi
        # gPhi2Int = (1.+np.sin(np.tile(inc[indsWith2Int],(8,1)).T)*np.sin(nuReal2Int+np.tile(w[indsWith2Int],(8,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
        # gd2Int = np.tile(a[indsWith2Int].to('AU'),(8,1)).T*(1.-np.tile(e[indsWith2Int],(8,1)).T**2.)/(np.tile(e[indsWith2Int],(8,1)).T*np.cos(nuReal2Int)+1.)
        # gdmags2Int = deltaMag(np.tile(p[indsWith2Int],(8,1)).T,np.tile(Rp[indsWith2Int].to('AU'),(8,1)).T,gd2Int,gPhi2Int) #calculate dmag of the specified x-value
        # #del gPhi2Int, gd2Int
        # #For arccos in pi-2pi

        # #del inBoundsBools2Int, outReal2Int
        # gPhi22Int = (1.+np.sin(np.tile(inc[indsWith2Int],(8,1)).T)*np.sin(nuReal22Int+np.tile(w[indsWith2Int],(8,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
        # gd22Int = np.tile(a[indsWith2Int].to('AU'),(8,1)).T*(1.-np.tile(e[indsWith2Int],(8,1)).T**2.)/(np.tile(e[indsWith2Int],(8,1)).T*np.cos(nuReal22Int)+1.)
        # gdmags22Int = deltaMag(np.tile(p[indsWith2Int],(8,1)).T,np.tile(Rp[indsWith2Int].to('AU'),(8,1)).T,gd22Int,gPhi22Int) #calculate dmag of the specified x-value
        # #del gPhi22Int, gd22Int

        #
        dmags2IntErrorComb = np.abs(gdmags2IntComb - dmag)
        mindmag2IntErrorInds = np.argsort(dmags2IntErrorComb,axis=1)
        sortedNuReal2IntComb = nuRealComb2Int[np.tile(np.arange(dmags2IntErrorComb.shape[0]),(mindmag2IntErrorInds.shape[1],1)).T,mindmag2IntErrorInds[np.arange(dmags2IntErrorComb.shape[0])]]
        sortedgdmags2IntComb = gdmags2IntComb[np.tile(np.arange(dmags2IntErrorComb.shape[0]),(mindmag2IntErrorInds.shape[1],1)).T,mindmag2IntErrorInds[np.arange(dmags2IntErrorComb.shape[0])]]

        #Create Output Arrays
        nus2Int = np.asarray([sortedNuReal2IntComb[:,0],sortedNuReal2IntComb[:,1]]).T
        dmag2Int = np.asarray([sortedgdmags2IntComb[:,0],sortedgdmags2IntComb[:,1]]).T

        # #Evaluate which solutions are good and which aren't
        # #1 What is the smallest in 1 Int and what is the smallest in 2 Int
        # assert np.all(np.sum(~np.isnan(gdmags2Int),axis=1)==2), 'There are more or less than 2 solutions for 2 Int for gdmags2Int'
        # assert np.all(np.sum(~np.isnan(gdmags22Int),axis=1)==2), 'There are more or less than 2 solutions for 2 Int for gdmags22Int'
        # error1Int = np.abs(gdmags2Int - dmag)
        # error2Int = np.abs(gdmags22Int - dmag)
        # minerror1IntInds = np.nanargmin(error1Int,axis=1) #Finds the minimum error of 1
        # minerror2IntInds = np.nanargmin(error2Int,axis=1) #Finds the minimum error of 2
        # maxerror1IntInds = np.nanargmax(error1Int,axis=1) #Finds the maximum error of 1
        # maxerror2IntInds = np.nanargmax(error2Int,axis=1) #Finds the maximum error of 2
        # minerror1Int = np.nanmin(error1Int,axis=1) #Finds the minimum error of 1
        # minerror2Int = np.nanmin(error2Int,axis=1) #Finds the minimum error of 2
        # maxerror1Int = np.nanmax(error1Int,axis=1) #Finds the maximum error of 1
        # maxerror2Int = np.nanmax(error2Int,axis=1) #Finds the maximum error of 2
        # #Case 1, the maximum error in 1 are all less than the minimum error in 2, so gdmags2Int contains all valid solutions
        # indsCase1 = np.where(maxerror1Int <= minerror2Int)[0]
        # nus2Int[indsCase1,0] = nuReal2Int[indsCase1,minerror1IntInds[indsCase1]]
        # nus2Int[indsCase1,1] = nuReal2Int[indsCase1,maxerror1IntInds[indsCase1]]
        # dmag2Int[indsCase1,0] = gdmags2Int[indsCase1,minerror1IntInds[indsCase1]]
        # dmag2Int[indsCase1,1] = gdmags2Int[indsCase1,maxerror1IntInds[indsCase1]]
        # #Do a check if both nu values are identical
        # #Case 2, the maximum error in 2 are all less than the minimum error in 1, so gdmags22Int contains all valid solutions
        # indsCase2 = np.where(maxerror2Int < minerror1Int)[0]
        # nus2Int[indsCase2,0] = nuReal22Int[indsCase2,minerror2IntInds[indsCase2]]
        # nus2Int[indsCase2,1] = nuReal22Int[indsCase2,maxerror2IntInds[indsCase2]]
        # dmag2Int[indsCase2,0] = gdmags22Int[indsCase2,minerror2IntInds[indsCase2]]
        # dmag2Int[indsCase2,1] = gdmags22Int[indsCase2,maxerror2IntInds[indsCase2]]
        # #Case 3, the smallest exists in gdmags2Int, the second smallest exists in gdmags22Int
        # indsCase3 = np.where((minerror1Int <= minerror2Int)*(maxerror1Int > minerror2Int))[0]
        # nus2Int[indsCase3,0] = nuReal2Int[indsCase3,minerror1IntInds[indsCase3]]
        # nus2Int[indsCase3,1] = nuReal22Int[indsCase3,minerror2IntInds[indsCase3]]
        # dmag2Int[indsCase3,0] = gdmags2Int[indsCase3,minerror1IntInds[indsCase3]]
        # dmag2Int[indsCase3,1] = gdmags22Int[indsCase3,minerror2IntInds[indsCase3]]
        # #Case 4, the smallest exists in gdmags22Int, the second smallest exists in gdmags2Int
        # indsCase4 = np.where((minerror2Int < minerror1Int)*(maxerror2Int > minerror1Int))[0]
        # nus2Int[indsCase4,0] = nuReal22Int[indsCase4,minerror2IntInds[indsCase4]]
        # nus2Int[indsCase4,1] = nuReal2Int[indsCase4,minerror1IntInds[indsCase4]]
        # dmag2Int[indsCase4,0] = gdmags22Int[indsCase4,minerror2IntInds[indsCase4]]
        # dmag2Int[indsCase4,1] = gdmags2Int[indsCase4,minerror1IntInds[indsCase4]]
        # #Check indsCase1+indsCase2+indsCase3+indsCase3 encompases all inds
        # assert indsCase1.shape[0]+indsCase2.shape[0]+indsCase3.shape[0]+indsCase4.shape[0] == gdmags2Int.shape[0], 'solutions number dont fit'


        # #Checking nus2Int identical-ness
        # errornus2Int = np.abs(nus2Int[:,0]-nus2Int[:,1])


        #DELETE THIS CHUNK
        # print(saltyburrito)
        # #Between the minimum of 1 and the minimum of 2, which is the smallest
        # minErrorOf1or2 = np.argmin(np.stack((error1Int[np.arange(gdmags2Int.shape[0]),minerror1IntInds],error2Int[np.arange(gdmags2Int.shape[0]),minerror2IntInds])).T,axis=1) #Determines whether 1 or 2 has the smaller error
        # #Find the inds that have the smallest error in gdmags2Int or gdmags22Int
        # indsWithMinError1 = np.where(minErrorOf1or2 == 0)[0]
        # indsWithMinError2 = np.where(minErrorOf1or2 == 1)[0]
        
        # gdmagsCombined2Int = np.zeros((gdmags2Int.shape[0],2))
        # gdmagsCombined2Int[indsWithMinError1,0] = gdmags2Int[indsWithMinError1,minerror1IntInds[indsWithMinError1]]
        # gdmagsCombined2Int[indsWithMinError2,0] = gdmags22Int[indsWithMinError2,minerror2IntInds[indsWithMinError2]]


        # #Evaluate which solutions are good and which aren't
        # correctValBoolean12Int = np.abs(gdmags2Int - dmag) < 1e-4 #Values of nuReal which yield the desired dmag
        # correctValBoolean22Int = np.abs(gdmags22Int - dmag) < 1e-4 #values of nuReal2 which yield the desired dmag
        # bothBools2Int = correctValBoolean12Int*correctValBoolean22Int #values of nuReal 
        # #Combine the two sets of solutions
        # nusCombined2Int = np.zeros(nuReal2Int.shape)
        # nusCombined2Int = nuReal2Int*np.logical_xor(correctValBoolean12Int,bothBools2Int) + nuReal22Int*np.logical_xor(correctValBoolean22Int,bothBools2Int) + nuReal2Int*bothBools2Int #these are the nus where intersections occur
        # #del nuReal2Int, nuReal22Int
        # #Combine and verify the two sets of dmags resulting from the solutions
        # gdmagsCombined2Int = np.zeros(gdmags2Int.shape)
        # gdmagsCombined2Int = gdmags2Int*np.logical_xor(correctValBoolean12Int,bothBools2Int) + gdmags22Int*np.logical_xor(correctValBoolean22Int,bothBools2Int) + gdmags2Int*bothBools2Int
        # #del gdmags2Int, gdmags22Int, correctValBoolean12Int, correctValBoolean22Int
        # numSolsPer2Int = np.sum((~np.isnan(gdmagsCombined2Int)).astype('int'),axis=1)
        # #DELETEnumSolsHist2Int = np.histogram(numSolsPer2Int,bins=[-0.1,0.9,1.9,2.9,3.9,4.9,5.9,6.9,7.9,8.9,9.9])
        # #print(saltyburrito)
        # assert np.all(numSolsPer2Int == 2) #All 2 int must have 2 solutions
        # del numSolsPer2Int
        # #Now that all 2Int only have 2 solutions
        # # nus2IntSol0 = np.nanargmin(nusCombined2Int,axis=1)
        # # nus2IntSol1 = np.nanargmax(nusCombined2Int,axis=1)
        # nus2IntSol0 = np.nanargmin(np.abs(gdmagsCombined2Int-dmag),axis=1)
        # nus2IntSol1 = np.nanargmax(np.absnusCombined2Int,axis=1)
        # badInds = np.where(nus2IntSol0 == nus2IntSol1)[0] #inds where same nus inds are trying to be assigned
        # assert np.all(~(nus2IntSol0 == nus2IntSol1)) #Checks that all solutions use different inds
        # #Combine the 2 individual intersection solutions
        # nus2Int = np.stack((nusCombined2Int[np.arange(nusCombined2Int.shape[0]),nus2IntSol0],nusCombined2Int[np.arange(nusCombined2Int.shape[0]),nus2IntSol1])).T
        # dmag2Int = np.stack((gdmagsCombined2Int[np.arange(nusCombined2Int.shape[0]),nus2IntSol0],gdmagsCombined2Int[np.arange(nusCombined2Int.shape[0]),nus2IntSol1])).T
        # del gdmagsCombined2Int, nus2IntSol0, nus2IntSol1, nusCombined2Int

        # assert ~np.any(np.equal(nus2Int[:,0],nus2Int[:,1])), 'one of the 2 extrema nus are identical'
    else:
        print(saltyburrito) #this elseif structure this is a part of need to be undone

    if np.logical_not(indsWith4Int.size == 0):
        #4 Int
        out4Int = solve_dmag_Poly(dmag,e[indsWith4Int],inc[indsWith4Int],w[indsWith4Int],a[indsWith4Int],p[indsWith4Int],Rp[indsWith4Int])
        #Throw out roots not in correct bounds
        inBoundsBools4Int = (np.abs(out4Int.imag) <= 1e-7)*(out4Int.real >= -1.-1e-9)*(out4Int.real <= 1.+1e-9) #the out2 solutions that are inside of the desired bounds
        outBoundsBools4Int = np.logical_not(inBoundsBools4Int) # the out2 solutions that are inside the desired bounds
        outReal4Int = np.zeros(out4Int.shape) #just getting something with the right shape
        outReal4Int[outBoundsBools4Int] = np.real(out4Int[outBoundsBools4Int])*np.nan
        outReal4Int[inBoundsBools4Int] = np.real(out4Int[inBoundsBools4Int])
        #del out4Int
        outReal4Int = np.real(outReal4Int)

        #### Calculate nu from out (both 0-pi and pi-2pi)
        nuReal4Int = np.ones(outReal4Int.shape)*np.nan
        nuReal4Int[inBoundsBools4Int] = np.arccos(outReal4Int[inBoundsBools4Int]) #calculate arccos, there are 2 potential solutions... need to calculate both
        nuReal24Int = np.ones(outReal4Int.shape)*np.nan
        nuReal24Int[inBoundsBools4Int] = 2.*np.pi - np.arccos(outReal4Int[inBoundsBools4Int])
        nuRealComb = np.concatenate((nuReal4Int,nuReal24Int),axis=1) #combines the two arrays

        #Remove duplicates for inds that have more than 4 real solutions
        #DELETEindsWithMoreThan4RealSols = np.where(np.sum(np.isnan(nuRealComb).astype('int'),axis=1) >= 4)[0]
        #Yes we need to do it this way
        for i in np.arange(nuRealComb.shape[0]):
            rounded = np.round(nuRealComb[i],4) #Note: the highest accuracy is +/-0.0014
            theSet = set() #Create an empty set
            for j in np.arange(len(rounded)): #Iterate through each value in the array
                if not (rounded[j] in theSet) and not np.isnan(rounded[j]): #Check if it does not occur in the set of occurrences
                    theSet.add(rounded[j]) #If it doesn't exist, add it
                    #Keep the value the same
                else:
                    rounded[j] = np.nan #Replace the value with nan
                    # tmpdmags[j] = np.nan
            nuRealComb[i] = rounded
            # dmagsComb[i] = tmpdmags
        nuReal4Int = nuRealComb[:,0:8]
        nuReal24Int = nuRealComb[:,8:16]
        # gdmags4Int = dmagsComb[:,0:8]
        # gdmags24Int = dmagsComb[:,8:16]

        #For arccos in 0-pi
        gPhi4Int = (1.+np.sin(np.tile(inc[indsWith4Int],(8,1)).T)*np.sin(nuReal4Int+np.tile(w[indsWith4Int],(8,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
        gd4Int = np.tile(a[indsWith4Int].to('AU'),(8,1)).T*(1.-np.tile(e[indsWith4Int],(8,1)).T**2.)/(np.tile(e[indsWith4Int],(8,1)).T*np.cos(nuReal4Int)+1.)
        gdmags4Int = deltaMag(np.tile(p[indsWith4Int],(8,1)).T,np.tile(Rp[indsWith4Int].to('AU'),(8,1)).T,gd4Int,gPhi4Int) #calculate dmag of the specified x-value
        #For arccos in pi-2pi 
        gPhi24Int = (1.+np.sin(np.tile(inc[indsWith4Int],(8,1)).T)*np.sin(nuReal24Int+np.tile(w[indsWith4Int],(8,1)).T))**2./4. #TRYING THIS TO CIRCUMVENT POTENTIAL ARCCOS
        gd24Int = np.tile(a[indsWith4Int].to('AU'),(8,1)).T*(1.-np.tile(e[indsWith4Int],(8,1)).T**2.)/(np.tile(e[indsWith4Int],(8,1)).T*np.cos(nuReal24Int)+1.)
        gdmags24Int = deltaMag(np.tile(p[indsWith4Int],(8,1)).T,np.tile(Rp[indsWith4Int].to('AU'),(8,1)).T,gd24Int,gPhi24Int) #calculate dmag of the specified x-value
    
        #HERE REPLACE THIS TMPDMAGS WITH THE NX16 ARRAY LIKE IN THE PREVIOUS. THEN DO A NU IDENTICALNESS SWEEP
        dmagsComb = np.concatenate((gdmags4Int,gdmags24Int),axis=1) #combines the two arrays
        dmagsErrorComb = np.abs(dmagsComb-dmag) #KEEP, OLD WORKS np.concatenate((np.abs(gdmags4Int-dmag),np.abs(gdmags24Int-dmag)),axis=1) #combines the two arrays
        mindmagIndsAxis2 = np.nanargmin(dmagsErrorComb,axis=1)
        #ARGSORT INSTEAD
        mindmagErrorInds = np.argsort(dmagsErrorComb,axis=1)
        sortedNuRealComb = nuRealComb[np.tile(np.arange(dmagsErrorComb.shape[0]),(mindmagErrorInds.shape[1],1)).T,mindmagErrorInds[np.arange(dmagsErrorComb.shape[0])]]
        sorteddmagsComb = dmagsComb[np.tile(np.arange(dmagsErrorComb.shape[0]),(mindmagErrorInds.shape[1],1)).T,mindmagErrorInds[np.arange(dmagsErrorComb.shape[0])]]

        nus4Int = np.asarray([sortedNuRealComb[:,0],sortedNuRealComb[:,1],sortedNuRealComb[:,2],sortedNuRealComb[:,3]]).T
        dmag4Int = np.asarray([sorteddmagsComb[:,0],sorteddmagsComb[:,1],sorteddmagsComb[:,2],sorteddmagsComb[:,3]]).T
        
        #DELETEprint(saltyburrito)

        #DELETEtmpdmags = np.dstack((gdmags4Int,gdmags24Int))
        #DELETEtmpdmagError = np.dstack((np.abs(gdmags4Int-dmag),np.abs(gdmags24Int-dmag)))
        #DELETEmindmagIndsAxis2 = np.nanargmin(tmpdmagError,axis=2)
        #DELETEdmagMinErrorArray = tmpdmags[np.arange(tmpdmagError.shape[0]),:,mindmagIndsAxis2[np.arange(tmpdmagError.shape[0])]]

        # #Create Output Arrays
        # nus4Int = np.zeros((nuReal4Int.shape[0],4)) #The output Arrays for 4 Intersections
        # dmag4Int = np.zeros((nuReal4Int.shape[0],4)) #The output Arrays for 4 Intersections
        # #Evaluate which solutions are good and which aren't
        # #1 What is the smallest in 1 Int and what is the smallest in 2 Int
        # assert np.all(np.sum(~np.isnan(gdmags4Int),axis=1)==4), 'There are more or less than 4 solutions for 4 Int for gdmags4Int'
        # assert np.all(np.sum(~np.isnan(gdmags24Int),axis=1)==4), 'There are more or less than 4 solutions for 4 Int for gdmags24Int'
        # error1Int = np.abs(gdmags4Int - dmag)
        # error2Int = np.abs(gdmags24Int - dmag)
        # minerror1IntInds = np.nanargmin(error1Int,axis=1) #Finds the minimum error of 1
        # minerror2IntInds = np.nanargmin(error2Int,axis=1) #Finds the minimum error of 2
        # maxerror1IntInds = np.nanargmax(error1Int,axis=1) #Finds the maximum error of 1
        # maxerror2IntInds = np.nanargmax(error2Int,axis=1) #Finds the maximum error of 2
        # minerror1Int = np.nanmin(error1Int,axis=1) #Finds the minimum error of 1
        # minerror2Int = np.nanmin(error2Int,axis=1) #Finds the minimum error of 2
        # maxerror1Int = np.nanmax(error1Int,axis=1) #Finds the maximum error of 1
        # maxerror2Int = np.nanmax(error2Int,axis=1) #Finds the maximum error of 2
        # #Case 1, the maximum error in 1 are all less than the minimum error in 2, so gdmags2Int contains all valid solutions
        # indsCase1 = np.where(maxerror1Int <= minerror2Int)[0]
        # nus2Int[indsCase1,0] = nuReal4Int[indsCase1,minerror1IntInds[indsCase1]]
        # nus2Int[indsCase1,1] = nuReal4Int[indsCase1,maxerror1IntInds[indsCase1]]
        # dmag2Int[indsCase1,0] = gdmags4Int[indsCase1,minerror1IntInds[indsCase1]]
        # dmag2Int[indsCase1,1] = gdmags4Int[indsCase1,maxerror1IntInds[indsCase1]]
        # #Do a check if both nu values are identical
        # #Case 2, the maximum error in 2 are all less than the minimum error in 1, so gdmags22Int contains all valid solutions
        # indsCase2 = np.where(maxerror2Int < minerror1Int)[0]
        # nus2Int[indsCase2,0] = nuReal24Int[indsCase2,minerror2IntInds[indsCase2]]
        # nus2Int[indsCase2,1] = nuReal24Int[indsCase2,maxerror2IntInds[indsCase2]]
        # dmag2Int[indsCase2,0] = gdmags24Int[indsCase2,minerror2IntInds[indsCase2]]
        # dmag2Int[indsCase2,1] = gdmags24Int[indsCase2,maxerror2IntInds[indsCase2]]
        # #Case 3, the smallest exists in gdmags2Int, the second smallest exists in gdmags22Int
        # indsCase3 = np.where((minerror1Int <= minerror2Int)*(maxerror1Int > minerror2Int))[0]
        # nus2Int[indsCase3,0] = nuReal4Int[indsCase3,minerror1IntInds[indsCase3]]
        # nus2Int[indsCase3,1] = nuReal24Int[indsCase3,minerror2IntInds[indsCase3]]
        # dmag2Int[indsCase3,0] = gdmags4Int[indsCase3,minerror1IntInds[indsCase3]]
        # dmag2Int[indsCase3,1] = gdmags24Int[indsCase3,minerror2IntInds[indsCase3]]
        # #Case 4, the smallest exists in gdmags22Int, the second smallest exists in gdmags2Int
        # indsCase4 = np.where((minerror2Int < minerror1Int)*(maxerror2Int > minerror1Int))[0]
        # nus2Int[indsCase4,0] = nuReal24Int[indsCase4,minerror2IntInds[indsCase4]]
        # nus2Int[indsCase4,1] = nuReal4Int[indsCase4,minerror1IntInds[indsCase4]]
        # dmag2Int[indsCase4,0] = gdmags24Int[indsCase4,minerror2IntInds[indsCase4]]
        # dmag2Int[indsCase4,1] = gdmags4Int[indsCase4,minerror1IntInds[indsCase4]]
        # #Check indsCase1+indsCase2+indsCase3+indsCase3 encompases all inds
        # assert indsCase1.shape[0]+indsCase2.shape[0]+indsCase3.shape[0]+indsCase4.shape[0] == gdmags4Int.shape[0], 'solutions number dont fit'
        # # Turn Used Solutions To Nan
        # nuReal4Int[indsCase1,minerror1IntInds[indsCase1]] = np.ones(len(indsCase1))*np.nan
        # nuReal4Int[indsCase1,maxerror1IntInds[indsCase1]] = np.ones(len(indsCase1))*np.nan
        # gdmags4Int[indsCase1,minerror1IntInds[indsCase1]] = np.ones(len(indsCase1))*np.nan
        # gdmags4Int[indsCase1,maxerror1IntInds[indsCase1]] = np.ones(len(indsCase1))*np.nan
        # nuReal24Int[indsCase2,minerror2IntInds[indsCase2]] = np.ones(len(indsCase2))*np.nan
        # nuReal24Int[indsCase2,maxerror2IntInds[indsCase2]] = np.ones(len(indsCase2))*np.nan
        # gdmags24Int[indsCase2,minerror2IntInds[indsCase2]] = np.ones(len(indsCase2))*np.nan
        # gdmags24Int[indsCase2,maxerror2IntInds[indsCase2]] = np.ones(len(indsCase2))*np.nan
        # nuReal4Int[indsCase3,minerror1IntInds[indsCase3]] = np.ones(len(indsCase3))*np.nan
        # nuReal24Int[indsCase3,minerror2IntInds[indsCase3]] = np.ones(len(indsCase3))*np.nan
        # gdmags4Int[indsCase3,minerror1IntInds[indsCase3]] = np.ones(len(indsCase3))*np.nan
        # gdmags24Int[indsCase3,minerror2IntInds[indsCase3]] = np.ones(len(indsCase3))*np.nan
        # nuReal24Int[indsCase4,minerror2IntInds[indsCase4]] = np.ones(len(indsCase4))*np.nan
        # nuReal4Int[indsCase4,minerror1IntInds[indsCase4]] = np.ones(len(indsCase4))*np.nan
        # gdmags24Int[indsCase4,minerror2IntInds[indsCase4]] = np.ones(len(indsCase4))*np.nan
        # gdmags4Int[indsCase4,minerror1IntInds[indsCase4]] = np.ones(len(indsCase4))*np.nan

        # #DO IT ALL AGAIN
        # error1Int = np.abs(gdmags4Int - dmag)
        # error2Int = np.abs(gdmags24Int - dmag)
        # minerror1IntInds = np.nanargmin(error1Int,axis=1) #Finds the minimum error of 1
        # minerror2IntInds = np.nanargmin(error2Int,axis=1) #Finds the minimum error of 2
        # maxerror1IntInds = np.nanargmax(error1Int,axis=1) #Finds the maximum error of 1
        # maxerror2IntInds = np.nanargmax(error2Int,axis=1) #Finds the maximum error of 2
        # minerror1Int = np.nanmin(error1Int,axis=1) #Finds the minimum error of 1
        # minerror2Int = np.nanmin(error2Int,axis=1) #Finds the minimum error of 2
        # maxerror1Int = np.nanmax(error1Int,axis=1) #Finds the maximum error of 1
        # maxerror2Int = np.nanmax(error2Int,axis=1) #Finds the maximum error of 2
        # #Case 1, the maximum error in 1 are all less than the minimum error in 2, so gdmags2Int contains all valid solutions
        # indsCase1 = np.where(maxerror1Int <= minerror2Int)[0]
        # nus2Int[indsCase1,2] = nuReal4Int[indsCase1,minerror1IntInds[indsCase1]]
        # nus2Int[indsCase1,3] = nuReal4Int[indsCase1,maxerror1IntInds[indsCase1]]
        # dmag2Int[indsCase1,2] = gdmags4Int[indsCase1,minerror1IntInds[indsCase1]]
        # dmag2Int[indsCase1,3] = gdmags4Int[indsCase1,maxerror1IntInds[indsCase1]]
        # #Do a check if both nu values are identical
        # #Case 2, the maximum error in 2 are all less than the minimum error in 1, so gdmags22Int contains all valid solutions
        # indsCase2 = np.where(maxerror2Int < minerror1Int)[0]
        # nus2Int[indsCase2,2] = nuReal24Int[indsCase2,minerror2IntInds[indsCase2]]
        # nus2Int[indsCase2,3] = nuReal24Int[indsCase2,maxerror2IntInds[indsCase2]]
        # dmag2Int[indsCase2,2] = gdmags24Int[indsCase2,minerror2IntInds[indsCase2]]
        # dmag2Int[indsCase2,3] = gdmags24Int[indsCase2,maxerror2IntInds[indsCase2]]
        # #Case 3, the smallest exists in gdmags2Int, the second smallest exists in gdmags22Int
        # indsCase3 = np.where((minerror1Int <= minerror2Int)*(maxerror1Int > minerror2Int))[0]
        # nus2Int[indsCase3,2] = nuReal4Int[indsCase3,minerror1IntInds[indsCase3]]
        # nus2Int[indsCase3,3] = nuReal24Int[indsCase3,minerror2IntInds[indsCase3]]
        # dmag2Int[indsCase3,2] = gdmags4Int[indsCase3,minerror1IntInds[indsCase3]]
        # dmag2Int[indsCase3,3] = gdmags24Int[indsCase3,minerror2IntInds[indsCase3]]
        # #Case 4, the smallest exists in gdmags22Int, the second smallest exists in gdmags2Int
        # indsCase4 = np.where((minerror2Int < minerror1Int)*(maxerror2Int > minerror1Int))[0]
        # nus2Int[indsCase4,2] = nuReal24Int[indsCase4,minerror2IntInds[indsCase4]]
        # nus2Int[indsCase4,3] = nuReal4Int[indsCase4,minerror1IntInds[indsCase4]]
        # dmag2Int[indsCase4,2] = gdmags24Int[indsCase4,minerror2IntInds[indsCase4]]
        # dmag2Int[indsCase4,3] = gdmags4Int[indsCase4,minerror1IntInds[indsCase4]]
        # #Check indsCase1+indsCase2+indsCase3+indsCase3 encompases all inds
        # assert indsCase1.shape[0]+indsCase2.shape[0]+indsCase3.shape[0]+indsCase4.shape[0] == gdmags4Int.shape[0], 'solutions number dont fit'

        # print(saltyburrito)

        # #Evaluate which solutions are good and which aren't
        # correctValBoolean14Int = np.abs(gdmags4Int - dmag) < 1e-7 #Values of nuReal which yield the desired dmag
        # correctValBoolean24Int = np.abs(gdmags24Int - dmag) < 1e-7 #values of nuReal2 which yield the desired dmag
        # bothBools4Int = correctValBoolean14Int*correctValBoolean24Int #values of nuReal 
        # #Combine the two sets of solutions
        # nusCombined4Int = np.zeros(nuReal4Int.shape)
        # nusCombined4Int = nuReal4Int*np.logical_xor(correctValBoolean14Int,bothBools4Int) + nuReal24Int*np.logical_xor(correctValBoolean24Int,bothBools4Int) + nuReal4Int*bothBools4Int #these are the nus where intersections occur
        # #Combine and verify the two sets of dmags resulting from the solutions
        # gdmagsCombined4Int = np.zeros(gdmags4Int.shape)
        # gdmagsCombined4Int = gdmags4Int*np.logical_xor(correctValBoolean14Int,bothBools4Int) + gdmags24Int*np.logical_xor(correctValBoolean24Int,bothBools4Int) + gdmags4Int*bothBools4Int
        # numSolsPer4Int = np.sum((~np.isnan(gdmagsCombined4Int)).astype('int'),axis=1)
        # assert np.all(numSolsPer4Int == 4) #All 4 int must have 4 solutions
        # #Now that all 4Int only have 4 solutions, remove the nans from the combined array
        # nus4IntSol0Ind = np.nanargmin(nusCombined4Int,axis=1)
        # nus4IntSol3Ind = np.nanargmax(nusCombined4Int,axis=1)
        # assert np.all(~(nus4IntSol0Ind == nus4IntSol3Ind))
        # nus4IntSol0 = nusCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol0Ind]
        # nus4IntSol3 = nusCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol3Ind]
        # gdmags4IntSol0 = gdmagsCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol0Ind]
        # gdmags4IntSol3 = gdmagsCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol3Ind]
        # nusCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol0Ind] = np.nan
        # nusCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol3Ind] = np.nan
        # nus4IntSol1Ind = np.nanargmin(nusCombined4Int,axis=1)
        # nus4IntSol2Ind = np.nanargmax(nusCombined4Int,axis=1)
        # assert np.all(~(nus4IntSol1Ind == nus4IntSol2Ind))
        # nus4IntSol1 = nusCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol1Ind]
        # nus4IntSol2 = nusCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol2Ind]
        # gdmags4IntSol1 = gdmagsCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol1Ind]
        # gdmags4IntSol2 = gdmagsCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol2Ind]
        # nusCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol1Ind] = np.nan
        # nusCombined4Int[np.arange(nusCombined4Int.shape[0]),nus4IntSol2Ind] = np.nan
        # #Combine the 4 individual intersection solutions
        # nus4Int = np.stack((nus4IntSol0,nus4IntSol1,nus4IntSol2,nus4IntSol3)).T
        # dmag4Int = np.stack((gdmags4IntSol0,gdmags4IntSol1,gdmags4IntSol2,gdmags4IntSol3)).T
        # assert ~np.any(np.equal(nus4Int[:,0],nus4Int[:,1])), 'one of the 4 extrema nus are identical'
        # assert ~np.any(np.equal(nus4Int[:,0],nus4Int[:,2])), 'one of the 4 extrema nus are identical'
        # assert ~np.any(np.equal(nus4Int[:,0],nus4Int[:,3])), 'one of the 4 extrema nus are identical'
        # assert ~np.any(np.equal(nus4Int[:,1],nus4Int[:,2])), 'one of the 4 extrema nus are identical'
        # assert ~np.any(np.equal(nus4Int[:,1],nus4Int[:,3])), 'one of the 4 extrema nus are identical'
        # assert ~np.any(np.equal(nus4Int[:,2],nus4Int[:,3])), 'one of the 4 extrema nus are identical'
    else:
        nus4Int = None
        dmag4Int = None

    #histogram check for number of solutions. We should see either 0, 2, or 4
    #DELETEvals1Hist = np.histogram(np.sum(correctValBoolean1,axis=1),bins=[-0.1,0.9,1.9,2.9,3.9,4.9,5.9])
    #DELETEvals2Hist = np.histogram(np.sum(correctValBoolean2,axis=1),bins=[-0.1,0.9,1.9,2.9,3.9,4.9,5.9])
    #Take nuReal1, and nuReal2 where not in both Bools

    # sumNumSol = np.sum(np.logical_xor(correctValBoolean1,bothBools)) + np.sum(np.logical_xor(correctValBoolean2,bothBools)) + np.sum(bothBools)
    # numSols = np.sum(np.logical_xor(correctValBoolean1,bothBools) + np.logical_xor(correctValBoolean2,bothBools) + bothBools,axis=1)
    # numSolHist = np.histogram(numSols,bins=[-0.1,0.9,1.9,2.9,3.9,4.9])
    # np.sum(np.histogram(numSols,bins=[-0.1,0.9,1.9,2.9,3.9,4.9])[0])
    tstop_cos = time.time()
    print('nu From dmag execution time: ' + str(tstop_cos-tstart_cos))

    # #Verification Histogram
    # plt.figure(num=9000)
    # if ~(indsWith4Int.size == 0):
    #     plt.hist(dmag4Int.flatten(),alpha=0.3,color='red',bins=50)
    # plt.hist(dmag2Int.flatten(),alpha=0.3,color='blue',bins=50)
    # plt.yscale('log')
    # plt.show(block=False)

    #Quality Checks
    # assert np.all(numSols[indsWith2Int]==2)
    # assert np.all(numSols[indsWith4Int]==4)

    return nus2Int, nus4Int, dmag2Int, dmag4Int #nusCombined, gdmagsCombined, sumNumSol, numSols, numSolHist 

def calc_t_sInnersOuter(sma,e,W,w,inc,s_inner,s_outer,starMass,plotBool):
    """ Collates the times where each planet crosses s_inner and s_outer
    Args:
        sma (ndarray):
        e (ndarray):
        W (ndarray):
        w (ndarray):
        inc (ndarray):
        s_inner (ndarray):
        s_outer (ndarray):
        starMass (ndarray):
        plotBool (bool)
    Returns:
        times (numpy array):
            the collective array of times when the planet crosses the separation circle size (n x 8)
    """
    times_o = np.zeros((sma.shape[0],4))*np.nan
    times_i = np.zeros((sma.shape[0],4))*np.nan

    _,_,_,_,_,_,_,_,_,only2RealInds_o,yrealAllRealInds_o,\
        fourIntInds_o,twoIntOppositeXInds_o,twoIntSameYInds_o,_,_,_,_,_,\
        _,_,_, yrealImagInds_o,\
        _,_,_,_,t_fourInt0_o,t_fourInt1_o,t_fourInt2_o,t_fourInt3_o,t_twoIntSameY0_o,\
        t_twoIntSameY1_o,t_twoIntOppositeX0_o,t_twoIntOppositeX1_o,t_IntersectionOnly20_o,t_IntersectionOnly21_o,\
        _, _, _, _, _, _, _, _, _, _, _, _,\
        _,_,_,_,_,\
        _,_,_,_,_,_,\
        _,_,_,_,_,_,_,_,_,_,_,_,\
        _,_,_,_,_,_,_,_,_,_,_,_,\
        _,_,_,_,_,_,_, _ = calcMasterIntersections(sma,e,W,w,inc,s_inner*np.ones(len(sma)),starMass,False)

    #Combine them all into one storage array
    times_o[yrealAllRealInds_o[fourIntInds_o],0] = t_fourInt0_o
    times_o[yrealAllRealInds_o[fourIntInds_o],1] = t_fourInt1_o
    times_o[yrealAllRealInds_o[fourIntInds_o],2] = t_fourInt2_o
    times_o[yrealAllRealInds_o[fourIntInds_o],3] = t_fourInt3_o
    times_o[yrealAllRealInds_o[twoIntSameYInds_o],0] = t_twoIntSameY0_o
    times_o[yrealAllRealInds_o[twoIntSameYInds_o],1] = t_twoIntSameY1_o
    times_o[yrealAllRealInds_o[twoIntOppositeXInds_o],0] = t_twoIntOppositeX0_o
    times_o[yrealAllRealInds_o[twoIntOppositeXInds_o],1] = t_twoIntOppositeX1_o
    times_o[only2RealInds_o,0] = t_IntersectionOnly20_o
    times_o[only2RealInds_o,1] = t_IntersectionOnly21_o

    _,_,_,_,_,_,_,_,_,only2RealInds_i,yrealAllRealInds_i,\
        fourIntInds_i,twoIntOppositeXInds_i,twoIntSameYInds_i,_,_,_,_,_,\
        _,_,_, yrealImagInds_i,\
        _,_,_,_,t_fourInt0_i,t_fourInt1_i,t_fourInt2_i,t_fourInt3_i,t_twoIntSameY0_i,\
        t_twoIntSameY1_i,t_twoIntOppositeX0_i,t_twoIntOppositeX1_i,t_IntersectionOnly20_i,t_IntersectionOnly21_i,\
        _, _, _, _, _, _, _, _, _, _, _, _,\
        _,_,_,_,_,\
        _,_,_,_,_,_,\
        _,_,_,_,_,_,_,_,_,_,_,_,\
        _,_,_,_,_,_,_,_,_,_,_,_,\
        _,_,_,_,_,_,_, _ = calcMasterIntersections(sma,e,W,w,inc,s_outer*np.ones(len(sma)),starMass,False)

    #Combine them all into one storage array
    times_i[yrealAllRealInds_i[fourIntInds_i],0] = t_fourInt0_i
    times_i[yrealAllRealInds_i[fourIntInds_i],1] = t_fourInt1_i
    times_i[yrealAllRealInds_i[fourIntInds_i],2] = t_fourInt2_i
    times_i[yrealAllRealInds_i[fourIntInds_i],3] = t_fourInt3_i
    times_i[yrealAllRealInds_i[twoIntSameYInds_i],0] = t_twoIntSameY0_i
    times_i[yrealAllRealInds_i[twoIntSameYInds_i],1] = t_twoIntSameY1_i
    times_i[yrealAllRealInds_i[twoIntOppositeXInds_i],0] = t_twoIntOppositeX0_i
    times_i[yrealAllRealInds_i[twoIntOppositeXInds_i],1] = t_twoIntOppositeX1_i
    times_i[only2RealInds_i,0] = t_IntersectionOnly20_i
    times_i[only2RealInds_i,1] = t_IntersectionOnly21_i

    times = np.concatenate((times_o,times_i),axis=1)
    return times


def planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, s_inner, s_outer, dmag_upper, dmag_lower=None):
    """ Finds the nu values where the planet intersects the separations or dmags, subsequently checks whether the planet is visible in the specified time ranges
    Args:
        sma (ndarray):
        e (ndarray):
        W (ndarray):
        w (ndarray):
        inc (ndarray):
        p (ndarray):
        Rp (ndarray):
        starMass (ndarray):
        plotBool (ndarray):
        s_inner (ndarray):
        s_outer (ndarray):
        dmag_upper (ndarray):
        dmag_lower (ndarray):
    Returns:
        nus (ndarray):
        planetIsVisibleBool (ndarray)
    """
    nus = np.zeros((len(sma),18))*np.nan #4 from s_inner, 4 from s_outer, 4 from dmag_upper, 4 from dmag_lower, 2 for previous orbit intersection and next orbit intersection
    #### nu from s_inner
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
        twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods = calcMasterIntersections(sma,e,W,w,inc,s_inner*np.ones(len(sma)),starMass,plotBool)
    nus[only2RealInds,0:2] = nu_IntersectionsOnly2
    nus[yrealAllRealInds[fourIntInds],0:4] = nu_fourInt
    nus[yrealAllRealInds[twoIntOppositeXInds],0:2] = nu_twoIntOppositeX
    nus[yrealAllRealInds[twoIntSameYInds],0:2] = nu_twoIntSameY
    #### nu from s_outer
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
        twoIntOppositeX_y,xIntersectionsOnly2,yIntersectionsOnly2,typeInds0,typeInds1,typeInds2,typeInds3, periods = calcMasterIntersections(sma,e,W,w,inc,s_outer*np.ones(len(sma)),starMass,plotBool)
    nus[only2RealInds,4:6] = nu_IntersectionsOnly2
    nus[yrealAllRealInds[fourIntInds],4:8] = nu_fourInt
    nus[yrealAllRealInds[twoIntOppositeXInds],4:6] = nu_twoIntOppositeX
    nus[yrealAllRealInds[twoIntSameYInds],4:6] = nu_twoIntSameY
    #### Solving for dmag_min and dmag_max for each planet ################
    mindmag, maxdmag, dmaglminAll, dmaglmaxAll, indsWith2, indsWith4, nuMinDmag, nuMaxDmag, nulminAll, nulmaxAll = calc_planet_dmagmin_dmagmax(e,inc,w,sma*u.AU,p,Rp)
    #### nu From dmag_upper
    print('Num Planets with At Least 2 Int given dmag: ' + str(np.sum((mindmag < dmag_upper)*(maxdmag > dmag_upper))))
    print('Num Planets with dmag local extrema: ' + str(len(indsWith4)))
    print('Num Planets with given 4 Int given dmag: ' + str(np.sum((dmaglminAll < dmag_upper)*(dmaglmaxAll > dmag_upper))))
    indsWith4Int = indsWith4[np.where((dmaglminAll < dmag_upper)*(dmaglmaxAll > dmag_upper))[0]]
    indsWith2Int = list(set(np.where((mindmag < dmag_upper)*(maxdmag > dmag_upper))[0]) - set(indsWith4Int))
    nus2Int, nus4Int, dmag2Int, dmag4Int = calc_planetnu_from_dmag(dmag_upper,e,inc,w,sma*u.AU,p,Rp,mindmag, maxdmag, indsWith2Int, indsWith4Int)
    nus[indsWith2Int,8:10] = nus2Int
    nus[indsWith4Int,8:12] = nus4Int
    #### nu From dmag_lower
    if dmag_lower == None:
        #default case? 0s maybe idk empty stuff
        dmag_lower = 0.
    else:
        print('Num Planets with At Least 2 Int given dmag: ' + str(np.sum((mindmag < dmag_lower)*(maxdmag > dmag_lower))))
        print('Num Planets with dmag local extrema: ' + str(len(indsWith4)))
        print('Num Planets with given 4 Int given dmag: ' + str(np.sum((dmaglminAll < dmag_lower)*(dmaglmaxAll > dmag_lower))))
        indsWith4Int = indsWith4[np.where((dmaglminAll < dmag_lower)*(dmaglmaxAll > dmag_lower))[0]]
        indsWith2Int = list(set(np.where((mindmag < dmag_lower)*(maxdmag > dmag_lower))[0]) - set(indsWith4Int))
        nus2Int, nus4Int, dmag2Int, dmag4Int = calc_planetnu_from_dmag(dmag_lower,e,inc,w,sma*u.AU,p,Rp,mindmag, maxdmag, indsWith2Int, indsWith4Int)
        nus[indsWith2Int,12:14] = nus2Int
        nus[indsWith4Int,12:16] = nus4Int
    ########################################################################
    
    #Finding which planets are all nan for efficiency
    nanbool = np.isnan(nus)
    indsNotAllNan = np.where(np.logical_not(np.all(nanbool,axis=1)))[0] #Get the inds of planets where not all nus are nan

    #Aded ranges above or below each nan (so I can simply do a midpoint evaluation with no fancy indexing)
    nus_min = np.nanmin(nus[indsNotAllNan],axis=1)
    nus_max = np.nanmax(nus[indsNotAllNan],axis=1)
    nus[:,16] = 2.*np.pi #2.*np.pi + nus_min #append the next orbit to this bit
    nus[:,17] = 0. #nus_max - 2.*np.pi #append the previous orbit intersection

    #sort the nus from smallest to largest
    for i in np.arange(nus.shape[0]):
        nus[i] = np.sort(nus[i])

    #calculate nus midpoints (for evaluating whether planets are visible within the range specified)
    nus_midpoints = (nus[:,1:] + nus[:,:-1])/2.

    #Calculate dmag and s for all midpoints
    Phi = (1.+np.sin(np.tile(inc,(17,1)).T)*np.sin(nus_midpoints+np.tile(w,(17,1)).T))**2./4.
    d = np.tile(sma*u.AU,(17,1)).T*(1.-np.tile(e,(17,1)).T**2.)/(np.tile(e,(17,1)).T*np.cos(nus_midpoints)+1.)
    dmags = deltaMag(np.tile(p,(17,1)).T,np.tile(Rp.to('AU'),(17,1)).T,d,Phi) #calculate dmag of the specified x-value
    ss = planet_star_separation(np.tile(sma,(17,1)).T,np.tile(e,(17,1)).T,nus_midpoints,np.tile(w,(17,1)).T,np.tile(inc,(17,1)).T)

    #Determine ranges where the planet is visible
    planetIsVisibleBool = (ss < np.ones((len(sma),17))*s_outer)*(ss > np.ones((len(sma),17))*s_inner)*(dmags < np.ones((len(sma),17))*dmag_upper)*(dmags > np.ones((len(sma),17))*dmag_lower)



    return nus, planetIsVisibleBool

def nukeKOE_winc(w,inc):
    #About +/- 4.7 deg from edge on and 0.343 deg from the two w locations
    indsToNuke = np.where( np.logical_or((np.abs(w-3.*np.pi/2.) < 0.006),(np.abs(w-np.pi/2.) < 0.006))*(np.abs(inc-np.pi/2.) < 0.095) )[0]
    w[indsToNuke] = np.pi/2. + np.random.randint(low=0,high=1,size=len(indsToNuke))*np.pi - 0.006 + np.random.randint(low=0,high=1,size=len(indsToNuke))*0.012 #generates random w at either w=pi/2 +/-0.006 or w=3pi/2+/-0.006
    inc[indsToNuke] = np.pi/2. - 0.095 + np.random.randint(low=0,high=1,size=len(indsToNuke))*0.19
    return w,inc

def nukeKOE(sma,e,W,w,inc,ar,er,Wr,wr,incr):
    """ Rounds the planet KOE because the KOE producee errors in the quartic solver
    """
    indsToNuke = np.where(np.abs(sma-ar).value + np.abs(e-er) + np.abs(W-Wr) + np.abs(w-wr) + np.abs(inc-incr) < 1e-2)[0]
    sma[indsToNuke] = np.round(sma[indsToNuke],2)
    e[indsToNuke] = np.round(e[indsToNuke],2)
    W[indsToNuke] = np.round(W[indsToNuke],2)
    w[indsToNuke] = np.round(w[indsToNuke],2) - 2.5*1e-3 + np.random.randint(low=0,high=1,size=len(indsToNuke))*5.*1e-3
    inc[indsToNuke] = np.round(inc[indsToNuke],2) - 2.5*1e-3 + np.random.randint(low=0,high=1,size=len(indsToNuke))*5.*1e-3
    return sma,e,W,w,inc

def nukeKOE_noWR(sma,e,w,inc,ar,er,wr,incr):
    """ Rounds the planet KOE because the KOE producee errors in the quartic solver
    """
    indsToNuke = np.where(np.abs(sma-ar).value + np.abs(e-er) + np.abs(w-wr) + np.abs(inc-incr) < 1e-2)[0]
    sma[indsToNuke] = np.round(sma[indsToNuke],2)
    e[indsToNuke] = np.round(e[indsToNuke],2)
    w[indsToNuke] = np.round(w[indsToNuke],2) - 2.5*1e-3 + np.random.randint(low=0,high=1,size=len(indsToNuke))*5.*1e-3
    inc[indsToNuke] = np.round(inc[indsToNuke],2) - 2.5*1e-3 + np.random.randint(low=0,high=1,size=len(indsToNuke))*5.*1e-3
    return sma,e,w,inc


def integrationTimeAdjustedCompletness(sma,e,W,w,inc,p,Rp,starMass,plotBool,periods, s_inner, s_outer, dmag_upper, tmax):
    """ Calculates the Integration Time Adjusted Completeness
    Args:
        sma (ndarray):
        e (ndarray):
        W (ndarray):
        w (ndarray):
        inc (ndarray):
        p (ndarray):
        Rp (ndarray):
        starMass (ndarray):
        plotBool (ndarray):
        periods (ndarray):
        s_inner (ndarray):
        s_outer (ndarray):
        dmag_upper (ndarray):
        tmax (ndarray):
    Returns:
        totalCompleteness_maxIntTimeCorrected (float):
            integration time adjusted completeness
    """
    nus, planetIsVisibleBool = planetVisibilityBounds(sma,e,W,w,inc,p,Rp,starMass,plotBool, s_inner, s_outer, dmag_upper, dmag_lower=None) #Calculate planet-star nu edges and visible regions
    ts = timeFromTrueAnomaly(nus,np.tile(periods,(18,1)).T*u.year.to('day'),np.tile(e,(18,1)).T) #Calculate the planet-star intersection edges
    dt = ts[:,1:] - ts[:,:-1] #Calculate time region widths
    gtIntLimit = dt > tmax #Create boolean array for inds
    totalVisibleTimePerTarget_maxIntTimeCorrected = np.nansum(np.multiply(np.multiply(dt-tmax,planetIsVisibleBool.astype('int')),gtIntLimit.astype('int')),axis=1) #We subtract the int time from the fraction of observable time
    totalCompletenessPerTarget_maxIntTimeCorrected = np.divide(totalVisibleTimePerTarget_maxIntTimeCorrected,periods*u.year.to('day')) # Fraction of time each planet is visible of its period
    totalCompleteness_maxIntTimeCorrected = np.mean(totalCompletenessPerTarget_maxIntTimeCorrected) #Calculates the total completenss by summing all the fractions and normalize by number of targets

    return totalCompleteness_maxIntTimeCorrected

def dynamicCompleteness(ts2,planetIsVisibleBool2,tobs1,tpast_startTimes,periods,ptypeBool=None):
    """
    Args:
        ts2 (ndarray):
            planet visibility window times
        planetIsVisibleBool2 (boolean array):
            visibility of the planet visibility windows with shape (ts2.shape[0],ts2.shape[1]-1)
        tobs1 (ndarray):
            time of first observation (should be randomly generated between 0 and T_k for each star)
        tpast_startTimes (float):
            time past first observation in days
        periods (ndarray):
            orbital periods of each planet in years with length ts2.shape[0]
        ptypeBool ():
            booleans indicating planet types. None means use all planets
    """
    if ptypeBool is None:
        ptypeBool = np.ones(len(tobs1))
    planetTypeBool = np.tile(ptypeBool,(7,1)).T
    planetIsVisibleBool2 = np.multiply(planetIsVisibleBool2,planetTypeBool) #Here we remove the planets that are not the desired type

    #Find all planets detectable at startTimes
    startTimes = np.tile(tobs1,(7,1)).T #startTime into properly sized array
    planetDetectedBools_times = np.multiply(ts2[:,:-1] < startTimes,np.multiply(ts2[:,1:] > startTimes,planetIsVisibleBool2)) #multiply time window bools by planetIsVisibleBool2. For revisit Completeness
    planetDetectedBools = np.nansum(planetDetectedBools_times,axis=1)
    planetNotDetectedBools = np.logical_not(planetDetectedBools) #for dynamic completeness

    tobs2 = np.tile(np.mod(tobs1+np.tile(tpast_startTimes,(len(tobs1),1)).T,periods*u.year.to('day')),(7,1)).T
    planetDetectedBools2_times = np.multiply(ts2[:,:-1] < tobs2,np.multiply(ts2[:,1:] > tobs2,planetIsVisibleBool2)) #is the planet visible at this time segment in time 2?
    planetDetectedBools2 = np.nansum(planetDetectedBools2_times,axis=1)
    planetNotDetectedBools2 = np.logical_not(planetDetectedBools2) #for dynamic completeness, the planet is not visible in this time segment at time 2

    #Revisit Comp.
    planetDetectedthenDetected = np.multiply(planetDetectedBools,planetDetectedBools2) #each planet detected at time 1 and time 2 #planets detected and still in visible region    
    #Dynamic Comp.
    planetNotDetectedThenDetected = np.multiply(planetNotDetectedBools,planetDetectedBools2) #each planet NOT detected at time 1 and detected at time 2 #planet not detected and now in visible region

    dynComp = np.sum(planetNotDetectedThenDetected)/len(planetNotDetectedThenDetected) #divide by all planets
    revisitComp = np.sum(planetDetectedthenDetected)/np.sum(planetDetectedBools) #divide by all planetes detected at startTimes

    return dynComp, revisitComp

