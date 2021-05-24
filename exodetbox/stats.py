#Statistics functions for twoDetMC
#By: Dean Keithly
import numpy as np

def average_weight(Nm,wmi):
    """ Calculates the average weight
    Args
        float:
            Nm is the number of elements in batch m
        ndarray:
            wmi is the array of weights for each data point xmi
    Returns:
        float:
            wm is the average weight of wmi
    """
    wm = np.nansum(wmi)/Nm
    return wm

def weighted_batch_mean(Nm,wmi,xmi,wm):
    """ Calculates the weighted batch mean
    Args:
        float:
            Nm is the number of elements in batch m
        ndarray:
            wmi is the array of weights for each data point xmi
        ndarray:
            xmi is the array of data points
        floar:
            wm is the average weight of wmi
    Returns:
        floar:
            xm is the weighted mean
    """
    xm = np.sum(wmi*xmi)/(wm*len(xmi))
    return xm

def incremental_weighted_mean(xbarj,j,xbarm,wbarm,wbarj):
    """ incremental weighted mean assuming constant batch sizes
    Args:
        float:
            xbarj is the cumulative average of all previous batches
        float:
            j is the number of previous batches
        float:
            xbarm is the average of the mth batch (the one being added)
        float:
            wbarm is the average weight of the mth batch (the one being added)
        float:
            wbarj is the average weight of previous batches
    Returns:
        float:
            xbarjp1 is the new weighted mean with the new data added
    """
    xbarjp1 = (j*xbarj*wbarj + xbarm*wbarm)/(j*wbarj + wbarm)
    return xbarjp1

def weighted_standard_deviation(xi,wmi,xbarj,Nm,wbarm):
    """ Calculates the standard deviation of the data
    Args:
        ndarray:
            xi is the array of data
        ndarray:
            wmi is the weighting of the data
        float:
            xbarj is the mean of the data
        float:
            Nm is the number of elements in xi
        float:
            wbarm is the average weight of the mth batch (the one being added)
    Returns:
        float:
            std is the standard deviation of the data
    """
    std = np.sqrt(np.abs(  np.nansum(wmi*(xi-np.average(xi,weights=wmi))**2.)/(np.nansum(wmi))  ))
    return std

def incremental_weights(j,wbarj,wbarm):
    """ Calculation of the incremental weight
    Args:
        float:
            j is the number of previous batches
        float:
            wbarj is the average weight of all previous batches
        float:
            wbarm is the average weight of the mth batch
    Returns:
        float:
            wbarjp1 is the new average weighting
    """
    wbarjp1 = (j*wbarj + wbarm)/(j+1)
    return wbarjp1


def incremental_weighted_standard_deviation(j,Nm,sigma_j,wbarm,wbarj,xbarm,xbarjp1,xi,wi):
    """ Calculates the incremental weighted standard deviation by adding some small batch to a large batch
    Args:
        float:
            j current batch number
        float:
            Nm number of elements in current batch
        float:
            sigma_j standard deviation of all previous batches
        float:
            wbarm mean weight of current batch
        float:
            wbarj mean weight of all previous
        float:
            xbarm mean of current batch
        float:
            xbarjp1 mean of all previous
        ndarray:
            xi array of values of current batch
        ndarray:
            wmi array of values of current batch
    """
    #std2 = np.sqrt(np.abs(np.nansum(wi*(xi-np.average(xi,weights=wi))**2.))/(np.nansum(wi)))
    wbaravg = (wbarj*j*Nm + wbarm*Nm)/(j*Nm+Nm)
    stdjp1 = np.sqrt(np.abs(  (j*Nm)/(j*Nm+Nm)*sigma_j**2. +  Nm/(j*Nm+Nm)*np.nansum(wi*(xi-xbarm)**2.)/np.nansum(wi) + j*Nm**2/(j*Nm+Nm)**2.*(xbarjp1-xbarm)**2.        ))  #this one works

    #stdjp1 = np.sqrt(np.abs((j*Nm)**2.*wbarj*sigma_j**2./(wbarj*(Nm+(j*Nm))**2.) + (Nm*wbarm)/(wbarj*(Nm+(j*Nm))**2.)*(-2.*xbarm*xbarjp1*Nm + Nm*xbarjp1**2.+np.nansum(xi**2.))))
    
    #stdjp1 = np.sqrt(np.abs((j*Nm)*wbarj*sigma_j**2./((Nm+(j*Nm))**2.)/wbaravg + 1./(wbarm*(Nm+(j*Nm)**2.))*(-2.*wbarj*xbarjp1*np.nansum(wi*xi)*Nm + wbarj*Nm*xbarjp1**2.+np.nansum(wi*xi**2.))/wbaravg))
    #stdjp1 = np.sqrt(np.abs(  ( (j*Nm)**2.*wbarj*sigma_j**2.  + Nm*wbarm*  (1./Nm)*(-2.*xbarjp1*np.nansum(xi)*Nm + Nm*xbarjp1**2.+np.nansum(xi**2.))  )/(j*Nm+Nm)  ))
    #stdjp1 = np.sqrt(np.abs(  ( (j*Nm)**2.*wbarj*sigma_j**2.  + Nm*wbarm*  np.nansum(wi*(xi-xbarm)**2.)  )/((j*Nm+Nm)**2*wbaravg)     ))
    #stdjp1 = np.sqrt(np.abs(  (wbarj*(j*Nm)/(j*Nm+Nm)*sigma_j**2. +  wbarm*(Nm)/(j*Nm+Nm)*np.nansum(wi*(xi-xbarm)**2.)/np.sum(wi) + wbarj*wbarm*j*Nm**2/(j*Nm+Nm)**2.*(xbarjp1-xbarm)**2.  )/(wbaravg*(j*Nm+Nm))      )) 
    #stdjp1 = np.sqrt(np.abs(  (  wbarj*(j*Nm)*(j*Nm)/(j*Nm+Nm)*sigma_j**2. +  wbarm*Nm*Nm/(j*Nm+Nm)*np.nansum(wi*(xi-xbarm)**2.)/np.nansum(wi) + (wbaravg*(j*Nm+Nm))*j*Nm**2/(j*Nm+Nm)**2.*(xbarjp1-xbarm)**2.)/(wbaravg*(j*Nm+Nm))        )) 


    # if stdjp1 > 2.*sigma_j:
    #     print(saltyburrito)
    return stdjp1


def incremental_mean_differentSizeBatches(xbarj,Nj,xbarm,Nm):
    """ Calculates the incremental mean different batch sizes
    Args:
        float:
            xbarj mean of all previous
        float:
            Nj number of planets in all previous batches
        ndarray:
            xbarm mean of values in current batch
        float:
            Nm number of elements in current batch
    """
    xbarjp1 = (Nj*xbarj + Nm*xbarm)/(Nj+Nm)
    return xbarjp1

def incremental_std_differentSizeBatches(Nj,Nm, sigma_j, xbarj, xi, xbarm):
    """ Calculates the incremental standard deviation for different batch sizes
    Args:
        float:
            Nj number of planets in all previous batches
        float:
            Nm number of elements in current batch
        float:
            sigma_j standard deviation of all previous batches
        float:
            xbarj mean of all previous
        ndarray:
            xi array of values of current batch
        ndarray:
            xbarm mean of values in current batch
    """
    #std_jp1 = np.sqrt(np.abs(Nj/(Nj+Nm)*sigma_j**2. + 1./(Nj+Nm) * (-2.*xbarm*xbarjp1*Nm + Nm**2.*xbarjp1**2. + np.nansum(xi**2.))))
    #std_jp1 = np.sqrt(np.abs(Nj/(Nj+Nm)*sigma_j**2. + 1./(Nj+Nm)**2 * (-2.*xbarm*xbarj*Nm + Nm**2.*xbarj**2. + np.nansum(xi**2.))))
    #std_jp1 = np.sqrt((Nj**2*sigma_j**2. + Nm*np.nansum((xi-xbarjp1)*(xi-xbarj)))/(Nj + Nm)**2)
    #std_jp1 = np.sqrt((Nj*(sigma_j**2. + np.mean(xi)**2.) + Nm*(np.std(xi)**2.+xbarj**2.))/(Nj + Nm) - ((Nj*np.mean(xbarj)+Nm*xbarm)/(Nj+Nm))**2)
    std_jp1_tmp = np.sqrt(1./Nm*np.nansum((xi-xbarj)**2.))
    std_jp1 = np.sqrt(Nj/(Nj+Nm)*sigma_j**2. + Nm/(Nj+Nm)*std_jp1_tmp**2.)
    # if std_jp1 > 2.*sigma_j:
    #     print(saltyburrito)
    return std_jp1

