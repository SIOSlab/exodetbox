import numpy as np
def trueAnomalyFromEccentricAnomaly(e,E):
    """ From https://en.wikipedia.org/wiki/True_anomaly #definitely exists in some other python scripts somewhere
    Args:
        ndarray:
            e
        ndarray:
            E
    Returns:
        ndarray:
            nu
    """
    nu = 2.*np.arctan(np.sqrt(1.+e)*np.tan(E/2.)/np.sqrt(1.-e))
    return nu