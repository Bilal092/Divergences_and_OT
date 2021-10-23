#%%
import numpy as np
import scipy as sp
'''
This Module implements the functions from:
https://papers.nips.cc/paper/2017/file/491442df5f88c6aa018e86dac21d3606-Paper.pdf
'''

#%%
def round_transpoly(X,r,c):
    '''
    implementation of ROUND function in Algorithm 2
    https://papers.nips.cc/paper/2017/file/491442df5f88c6aa018e86dac21d3606-Paper.pdf

    inputs: 
    -- r : source marginal vector from probability simplex
    -- c : target marginal vector from probability simplex
    -- X : Positive matrix to be rounded onto transport polytope.

    outputs:
    -- A : Rounded version of matrix X.

    Note: This rounding function is written for the matrices which are ouput of exponentiation,
          therefore with non-zero rows and columns. For tesing and use only input positive matrices. 
          It returns NAN terms for inputs with zeros rows or zero columns.
    '''

    assert np.isclose(np.sum(r), 1), "source points must lie on probability simplex."
    assert np.isclose(np.sum(c), 1), "target points must lie on probability simplex."
    assert np.size(r) == np.size(c), "source and target distributions must be of same size."
    assert np.max(np.shape(r)) == np.size(np.squeeze(r)), "input distribution must be vectorized."
    assert np.max(np.shape(c)) == np.size(np.squeeze(c)), "input distribution must be vcctorized."

    assert np.size(np.shape(X)) == 2, "cost must be a square matrix"
    assert X.shape[0] == X.shape[1], "cost matrix must be a square"

    r = r.reshape([np.size(r), 1])
    c = c.reshape([np.size(c), 1])

    A = X
    n = A.shape[0]
    r_A = np.sum(A, axis=1, keepdims=1)  # returns column vector

    # for i in range(0,n):
    #     scaling = np.min([1, r[i]/r_A[i]])
    #     A[i,:] = scaling * A[i,:]
    
    ratio_r = np.divide(r, r_A, out=np.zeros_like(r_A), where=r_A != 0)
    #ratio_r = np.exp(np.log(r, out = np.ones_like(r), where = r!=0) - np.log(r_A, out = np.ones_like(r_A), where = r_A!=0))
    #print(ratio_r)
    scaling_r = np.minimum(1, ratio_r)  #returns column vector
    A = scaling_r * A
    
    c_A = np.sum(A, axis=0, keepdims=1)  #returns row vector

    # for j in range(0,n):
    #     scaling = np.min([1, c[j]/c_A[j]])
    #     A[:,j] = scaling * A[:,j]
    ratio_c = np.divide(c.T, c_A, out=np.zeros_like(c_A), where = c_A != 0)
    #ratio_c = np.exp(np.log(c.T, out = np.ones_like(c.T), where = c.T!=0) - np.log(c_A, out = np.ones_like(c_A), where = c_A!=0))
    #print(ratio_c)
    scaling_c = np.minimum(1, ratio_c)
    A = scaling_c * A

    r_A = np.sum(A, axis=1, keepdims=1)  # returns column vector
    c_A = np.sum(A, axis=0, keepdims=1)  # returns row vector

    err_r = r_A - r
    err_c = c_A - c.T

    if (np.linalg.norm(err_r, ord=1)) == 0:
        return A
    else:
        A = A + err_r @ err_c / (np.linalg.norm(err_r, ord = 1))
        return A


# %%
def sinkhorn(A, r, c, C, max_iter = 2000, epsilon = 1e-8, compute_OTvals = False, Disp_iter = False):
    '''
    inputs:
    -- A: positive matrix of shape n x n
    -- r : source marginal vector from probability simplex of shape n x 1
    -- c : target marginal vector from probability simplex of shape n x 1
    -- max_iter: maximum number sinkhorn iterations.
    -- epsilon: convergence threshold
    -- C: cost matrix for optimal transport
    outputs:
    -- P: Sinkhorn projection matrix n x n 
    -- err: sum of source and target errors in sikhorn iteration
    -- OTvals Sinkhorn objective function value obtained after rounding
    '''

    assert np.isclose(np.sum(r), 1), "source points must lie on probability simplex."
    assert np.isclose(np.sum(c), 1), "target points must lie on probability simplex."
    assert np.size(r) == np.size(c), "source and target distributions must be of same size."
    assert np.max(np.shape(r)) == np.size(np.squeeze(r)), "input distribution must be vectorized."
    assert np.max(np.shape(c)) == np.size(np.squeeze(c)), "input distribution must be vcctorized."

    assert np.size(np.shape(A)) == 2, "cost must be a square matrix"
    assert A.shape[0] == A.shape[1], "cost matrix must be a square"

    assert C.shape == A.shape, "Cost matrix and Kernel must be of same shape"

    r = r.reshape([np.size(r), 1])
    c = c.reshape([np.size(c), 1])

    P = A
    err = np.zeros([max_iter + 1, 1])
    r_P = np.sum(P, axis = 1, keepdims=1) #returns a column vector. 
    c_P = np.sum(P, axis = 0, keepdims=1) #return a row vector.

    err[0] = np.linalg.norm(r - r_P, ord = 1) + np.linalg.norm(c - c_P.T, ord = 1)

    if compute_OTvals == True:
        OTvals = np.zeros([max_iter + 1, 1])
        OTvals[0] = np.sum(round_transpoly(P,r,c) * C)

    if Disp_iter == True:
        iter = 0
        print("iter = %d \n" %iter)
        print("error = %f" % err[iter])
        
    
    for iter in range(0, max_iter):
        if np.mod(iter + 1, 2) == 1:
            r_P = np.sum(P, axis=1, keepdims=1) #returns a column vector.
            scaling_r = np.divide(r, r_P, out = np.zeros_like(r_P), where = r_P!=0)
            #scaling_r = np.exp(np.log(r, out = np.ones_like(r), where = r!=0) - np.log(r_P, out = np.ones_like(r_P), where = r_P!=0))
            #P = P * (r / r_P)
            P = scaling_r * P
            r_P = np.sum(P, axis=1, keepdims=1) #returns a column vector.
            c_P = np.sum(P, axis=0, keepdims=1) #returns a row vector.
            err[iter + 1] = np.linalg.norm(r - r_P, ord=1) + np.linalg.norm(c - c_P.T, ord=1)
        else:
            c_P = np.sum(P, axis=0, keepdims=1) #returns a row vector.
            scaling_c = np.divide(c.T, c_P, out=np.zeros_like(c_P), where=c_P!= 0)
            #scaling_c = np.exp(np.log(c.T, out = np.ones_like(c.T), where = c.T!=0) - np.log(c_P, out = np.ones_like(c_P), where = c_P!=0))
            #P = P * (c.T / c_P)
            P = scaling_c * P
            r_P = np.sum(P, axis=1, keepdims=1) #returns a column vector.
            c_P = np.sum(P, axis=0, keepdims=1) #returns a row vector.
            err[iter + 1] = np.linalg.norm(r - r_P, ord=1) + np.linalg.norm(c - c_P.T, ord=1)
        
        if compute_OTvals == True:
            OTvals[iter + 1] = np.sum(round_transpoly(P, r, c) * C)
        if Disp_iter == True:
            print("iter = %d \n"%iter)
            print("error = %f"%err[iter])
    
    if compute_OTvals == True:
        return P, OTvals,err
    else:
        return P, err


# def Greenkhorn(A, r, c, C, max_iter=2000, epsilon=1e-8, compute_OTvals=False):
#     '''
#     inputs:
#     -- A: positive matrix of shape n x n
#     -- r : source marginal vector from probability simplex of shape n x 1
#     -- c : target marginal vector from probability simplex of shape n x 1
#     -- max_iter: maximum number sinkhorn iterations.
#     -- epsilon: convergence threshold
#     -- C: cost matrix for optimal transport
#     outputs:
#     -- P: Sinkhorn projection matrix n x n 
#     -- err: sum of source and target errors in sikhorn iteration
#     -- OTvals Greenkhorn objective function value obtained after rounding
#     '''

#     assert np.isclose(np.sum(r), 1), "source points must lie on probability simplex."
#     assert np.isclose(np.sum(c), 1), "target points must lie on probability simplex."
#     assert np.size(r) == np.size(c), "source and target distributions must be of same size."
#     assert np.max(np.shape(r)) == np.size(np.squeeze(r)), "input distribution must be vectorized."
#     assert np.max(np.shape(c)) == np.size(np.squeeze(c)), "input distribution must be vcctorized."

#     assert np.size(np.shape(A)) == 2, "cost must be a square matrix"
#     assert A.shape[0] == A.shape[1], "cost matrix must be a square"

#     assert C.shape == A.shape, "Cost matrix and Kernel must be of same shape"

#     r = r.reshape([np.size(r), 1])
#     c = c.reshape([np.size(c), 1])

#     P = A
#     err = np.zeros([max_iter + 1, 1])
#     r_P = np.sum(P, axis=1, keepdims=1)  # returns a column vector.
#     c_P = np.sum(P, axis=0, keepdims=1)  # return a row vector.


#     r_gain = r_P - r + r * np.log(r/)

#     err[0] = np.linalg.norm(r - r_P, ord=1) + np.linalg.norm(c - c_P.T, ord=1)

#     if compute_OTvals == True:
#         OTvals = np.zeros([max_iter + 1, 1])
#         OTvals[0] = np.sum(round_transpoly(P, r, c) * C)

#     for iter in range(0, max_iter):
#         if np.mod(iter + 1, 2) == 1:
#             r_P = np.sum(P, axis=1, keepdims=1)  # returns a column vector.
#             scaling_r = np.divide(
#                 r, r_P, out=np.zeros_like(r_P), where=r_P != 0)
#             #P = P * (r / r_P)
#             P = scaling_r * P
#             r_P = np.sum(P, axis=1, keepdims=1)  # returns a column vector.
#             c_P = np.sum(P, axis=0, keepdims=1)  # returns a row vector.
#             err[iter + 1] = np.linalg.norm(r - r_P, ord=1) + \
#                 np.linalg.norm(c - c_P.T, ord=1)
#         else:
#             c_P = np.sum(P, axis=0, keepdims=1)  # returns a row vector.
#             scaling_c = np.divide(
#                 c.T, c_P, out=np.zeros_like(c_P), where=c_P != 0)
#             #P = P * (c.T / c_P)
#             P = scaling_c * P
#             r_P = np.sum(P, axis=1, keepdims=1)  # returns a column vector.
#             c_P = np.sum(P, axis=0, keepdims=1)  # returns a row vector.
#             err[iter + 1] = np.linalg.norm(r - r_P, ord=1) + \
#                 np.linalg.norm(c - c_P.T, ord=1)

#         if compute_OTvals == True:
#             OTvals = np.zeros([max_iter + 1, 1])
#             OTvals[0] = np.sum(round_transpoly(P, r, c) * C)

#     if compute_OTvals == True:
#         return P, OTvals, err
#     else:
#         return P, err

# %%

def OT(r, c, C, OT_iteration=2000, eps = 0.1, compute_obj=False, Method = "Sinkhorn"):
    '''
    inputs:
    -- r : source marginal vector from probability simplex of shape n x 1
    -- c : target marginal vector from probability simplex of shape n x 1
    -- max_iter: maximum number sinkhorn iterations.
    -- epsilon: convergence threshold
    -- C: cost matrix for optimal transport
    outputs:
    -- P: Sinkhorn projection matrix n x n 
    -- err: sum of source and target errors in sikhorn iteration
    -- OTvals Sinkhorn objective function value obtained after rounding
    '''
    assert np.isclose(np.sum(r), 1), "source points must lie on probability simplex."
    assert np.isclose(np.sum(c), 1), "target points must lie on probability simplex."
    assert np.size(r) == np.size(c), "source and target distributions must be of same size."
    assert np.max(np.shape(r)) == np.size(np.squeeze(r)), "input distribution must be vectorized."
    assert np.max(np.shape(c)) == np.size(np.squeeze(c)), "input distribution must be vcctorized."

    assert np.size(np.shape(C)) == 2, "cost must be a square matrix"
    assert C.shape[0] == C.shape[1], "cost matrix must be a square"
    

    N = C.shape[0]
    eta = (4 * np.log(N) )/ eps

    A = np.exp(- eta * C)
    if Method == "Sinkhorn":
        if compute_obj == True:
            B, OTvals, err = sinkhorn( A, r, c, C, max_iter=OT_iteration, epsilon=eps, compute_OTvals=compute_obj)
        else:
            B, err, = sinkhorn(A, r, c, C, max_iter=OT_iteration, epsilon=eps, compute_OTvals=compute_obj)

    P_approx = round_transpoly(B, r, c)
    if compute_obj == True:
        return P_approx, OTvals, err
    else:
        return P_approx,  err 




# %%

# %%
