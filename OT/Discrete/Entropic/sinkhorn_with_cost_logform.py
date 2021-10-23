#%%
import numpy as np
import scipy as sp 

def sinkhorn_with_cost_logform(C, lambd = 1000, sinkhorn_iter = 1000, thresh = 1e-11):

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

    [m, n] = C.shape
    assert( m == n), "Source and target distributions must be of same size"

    D = C - np.min(C)
    epsilon = 1 / lambd

    f = np.zeros([m,1])
    g = np.zeros([n,1])

    for inner in range(0, sinkhorn_iter):
        fprev = f
        S = D - f - g.T
        Smin = np.min(S, axis = 1, keepdims=1)
        f = f + Smin - epsilon * \
            (np.log(m) + np.log(np.sum(np.exp(-1/epsilon * (S - Smin)), axis=1, keepdims = 1)))
        S = D - f - g.T
        Smin = np.min(S, axis = 0, keepdims = 1)
        g = g + Smin.T - epsilon * \
            ( np.log(m) + np.log(np.sum(np.exp(-1/epsilon * (S - Smin)), axis = 0, keepdims = 1))).T

        if np.mod(inner - 1, 10) == 0:
            delta = np.sum( np.abs (f - fprev))
            if delta < thresh:
                print("Connverged at %i"%inner)
                break

    div = np.mean(f) + np.mean(g)
    gamma = np.exp(-lambd * (D - f - g.T))


    return gamma


# %%
