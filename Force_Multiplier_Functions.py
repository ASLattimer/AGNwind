import numpy as np
import pickle

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
def SatPow(normt):
    """ 
    Input
    -------------
        norm_t: dimensionless optical depth parameter t, normalized by t* - must not be logged!!
    Output
    -------------
        Output: force multiplier M(t), normalized by the turnover point t* and the saturation value Qbar
    
    """
    const = 1
    k = 500
    s = 0.3
    alpha = 0.9
    num = const*k
    par = (k**s) + normt**(alpha*s)
    denom = par**(1/s)
    frac = num/denom
    return frac


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
def t_star(log10_qbar):
    """ 
    For calculating the turnover point t* of the saturated M(t) power-law
    
    Input
    -----------
          log10_qbar: constant saturation value of the M(t) curve, either estimated from the KNN model or known a priori, must be provided in log10
    Output
    -----------
          t*: value of dimensionless optical depth t that marks the transition from power-law to constant regions
    """
    
    turnt_coeffs = np.array([-0.00887083, -1.18350552, -1.97363217]) # fit coefficients to calculate the turnover point t*
    val = np.polyval(turnt_coeffs,log10_qbar)
    
    return val    


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
def Qbar_estimate(log10_xi,gammaX,shape=1):
    """ 
    Estimates saturation value Qbar
    
    Input
    --------
        log10_xi: log base 10 of the ionization parameter
        gammaX  : X-ray spectral index
        shape   : denotes if xi and gammaX are given as single values (1) or arrays (2). Default is 1.
    """
    # Get value of Qbar
    KNN = pickle.load(open('qbar_knn_model.sav', 'rb'))        #load the KNN model
    if shape == 1:
        X = np.array((log10_xi,gammaX)).reshape(-1,1)          # if using one value each of xi and gamma, if using arrays no need to reshape: X = np.array((xi,gammaX))
    if shape == 2:
        X = np.array((log10_xi,gammaX))                        # if using arrays of of xi and gamma
    qbar_pred = KNN.predict(X.T)                               # logged value of Qbar
    
    return qbar_pred


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
def Mt_Calc(log10_xi,gammaX,shape=1):
    """ 
    Estimates force multiplier M(t)
    
    Input
    --------
        log10_xi: log base 10 of the ionization parameter
        gammaX  : X-ray spectral index
        shape   : denotes if xi and gammaX are given as single values (1) or arrays (2). Default is 1.
    """
    tnorm_range = np.logspace(-40,40,num=100) # set the range of the normalized optical depth

    # Get value of Qbar
    KNN = pickle.load(open('qbar_knn_model.sav', 'rb'))        #load the KNN model
    if shape == 1:
        X = np.array((log10_xi,gammaX)).reshape(-1,1)          # if using one value each of xi and gamma, if using arrays no need to reshape: X = np.array((xi,gammaX))
    if shape == 2:
        X = np.array((log10_xi,gammaX))                        # if using arrays of of xi and gamma
    qbar_pred = KNN.predict(X.T)                               # logged value of Qbar
    

    # Estimate t* 
    turnt_coeffs = np.array([-0.00887083, -1.18350552, -1.97363217]) # fit coefficients to calculate the turnover point t*
    tstar_val = 10**np.polyval(turnt_coeffs,qbar_pred)
    
    # Calculate normalized M(t) curve
    const = 1
    k = 500
    s = 0.3
    alpha = 0.9
    num = const*k
    par = (k**s) + tnorm_range**(alpha*s)
    denom = par**(1/s)
    Mt_norm = num/denom    
  
    Mt_pred = np.multiply(Mt_norm,qbar_pred)      # estimate M(t) given M_norm and Qbar
    t_range = np.multiply(tnorm_range,tstar_val)  # estimate range of t given t*
    
    return t_range,Mt_pred