#%%
import numpy as np
import scipy as sp 

def sinkhorn_with_cost(C, lambd = 10, sinkhorn_iter = 1000, thresh = 1e-11):

    '''
    Inputs:  C  is transport cost matrix
             lambd is Langrange multiplier
             sinklhorn iter counts iterations
             threshold is convergence threshold

    Outputs: Dictionary out = {"div": div, "gamma": gamma, "lambd": lambd}
             dic is squared objective
             gamma is Entropy regularized transport plan.
             lambd is lagrange multiplier. 

    '''
    m,n = C.shape
    assert (m == n),"Input and output data points must be equal in number"

    eps =  np.finfo(float).eps

    D = C - np.min(C)
    K = np.exp(-lambd * D)
    Kt = K.T 

    v = np.inf * np.ones([n,1])
    u = np.ones([m,1])/m
    lambd = 1.05 * lambd 

    while ( np.any(np.any(v) ==  np.inf ) or ( (np.max(C) - np.min(C)) > 1e3 ) ):
        lambd  = lambd / 1.05
        K = np.exp(-lambd * D)
        Ktu = K.T @ u
        v = 1 / Ktu
    
    K2  = 1/m * K
    u = np.ones([m,1]) / m
    v = np.ones([n,1]) / n 

    for inner in range(0, sinkhorn_iter):
        uprev = u
        vprev = v
        Ktu = m * (K.T @ u + eps )
        v = 1 / Ktu
        u = 1 / (K2@v + eps)

        if ( np.any( Ktu == 0) or np.any( u == np.nan ) or np.any( v == np.nan) or np.any( u == np.inf )  or np.any( v == np.inf) ):
            u = uprev
            v = vprev
            break

        if np.mod(inner - 1, 10) ==  0:
            err = np.sum((u - uprev)**2 )/sum(u**2) + np.sum((v - vprev)**2)/sum(v**2)
            if err < thresh:
                break
    
    div = u.T @ ((K*D) @ v)
    gamma = (np.diag(np.squeeze(u))) @ K @ np.diag(np.squeeze(v))

    #out = {"div": div, "gamma": gamma, "lambd": lambd}

    return gamma, div, lambd
        
