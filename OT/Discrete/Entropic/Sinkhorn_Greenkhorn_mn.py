#%%
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import os
# %%
def round_transpoly(F, r, c):
    '''
    implementation of ROUND function in Algorithm 1
    https://papers.nips.cc/paper/2017/file/491442df5f88c6aa018e86dac21d3606-Paper.pdf
    written by: Bilal Riaz bilalria@udel.edu
    inputs: 
    -- r : source marginal vector from probability simplex R^m
    -- c : target marginal vector from probability simplex R^n
    -- A : Positive matrix to be rounded onto transport polytope.
    outputs:
    -- A : Rounded version of matrix X.
    Note: This rounding function is written for the matrices which are ouput of exponentiation,
          therefore with non-zero rows and columns. For tesing and use only input positive matrices. 
          It returns NAN terms for inputs with zeros rows or zero columns.
    '''

    assert np.isclose(np.sum(r), 1), "source points must lie on probability simplex."
    assert np.isclose(np.sum(c), 1), "target points must lie on probability simplex."

    r = r.reshape([np.size(r), 1])
    c = c.reshape([np.size(c), 1])

    A = F
    n = F.shape[1]

    # row sums 
    r_A = np.sum(A, axis=1, keepdims=1)  # returns column vector
    ratio_r = np.divide(r, r_A, out=np.zeros_like(r_A), where= r_A!=0) # returns column vector
    scaling_r = np.minimum(1, ratio_r)  # returns column vector
    # row scaling
    A = scaling_r * A

    # column sum
    c_A = np.sum(A, axis=0, keepdims=1)  # returns row vector
    ratio_c = np.divide(c.T, c_A, out=np.zeros_like(c_A), where = c_A!=0)  # returns row vector
    scaling_c = np.minimum(1, ratio_c)   # returns row vector
    # column scaling
    A = scaling_c * A

    r_A = np.sum(A, axis=1, keepdims=1)  # returns column vector
    c_A = np.sum(A, axis=0, keepdims=1)  # returns row vector

    err_r = r_A - r
    err_c = c_A - c.T

    if (np.linalg.norm(err_r, ord=1)) == 0:
        return A
    else:
        A = A + err_r @ err_c / (np.linalg.norm(err_r, ord=1))
        return A


def sinkhorn(A, r, c, C, max_iter=2000, compute_otvals=False, disp_iter=False):
    '''
    implementation of SINKHORN function in Algorithm 2
    https://papers.nips.cc/paper/2017/file/491442df5f88c6aa018e86dac21d3606-Paper.pdf
    written by: Bilal Riaz bilalria@udel.edu
    inputs:
    -- A: positive matrix of shape m x n
    -- r : source marginal vector from probability simplex of shape m x 1
    -- c : target marginal vector from probability simplex of shape n x 1
    -- max_iter: maximum number sinkhorn iterations.
    -- epsilon: convergence threshold
    -- C: cost matrix for optimal transport
    outputs:
    -- P: Sinkhorn projection matrix m x n 
    -- err: sum of source and target errors in sikhorn iteration
    -- OTvals Sinkhorn objective function value obtained after rounding
    '''

    assert np.isclose( np.sum(r), 1), "source points must lie on probability simplex."
    assert np.isclose(np.sum(c), 1), "target points must lie on probability simplex."
    assert np.size(np.shape(A)) == 2, "cost must be a square matrix"

    assert C.shape == A.shape, "Cost matrix and Kernel must be of same shape"

    r = r.reshape([np.size(r), 1])
    c = c.reshape([np.size(c), 1])

    P = A
    err = np.zeros([max_iter + 1, 1])
    r_P = np.sum(P, axis=1, keepdims=1)  # returns a column vector.
    c_P = np.sum(P, axis=0, keepdims=1)  # return a row vector.

    err[0] = np.linalg.norm(r - r_P, ord=1) + np.linalg.norm(c - c_P.T, ord=1)

    if compute_otvals == True:
        otvals = np.zeros([max_iter + 1, 1])
        otvals[0] = np.sum(round_transpoly(P, r, c) * C)

    if disp_iter == True:
        iter = 0
        print("iter = %d \n" % iter)
        print("error = %f" % err[iter])

    for iter in range(0, max_iter):
        if np.mod(iter + 1, 2) == 1:
            r_P = np.sum(P, axis=1, keepdims=1)  # returns a column vector.
            scaling_r = np.divide(r, r_P, out=np.zeros_like(r_P), where=r_P != 0)
            #scaling_r = np.exp(np.log(r, out = np.ones_like(r), where = r!=0) - np.log(r_P, out = np.ones_like(r_P), where = r_P!=0))
            #P = P * (r / r_P)
            P = scaling_r * P
            r_P = np.sum(P, axis=1, keepdims=1)  # returns a column vector.
            c_P = np.sum(P, axis=0, keepdims=1)  # returns a row vector.
            err[iter + 1] = np.linalg.norm(r - r_P, ord=1) + np.linalg.norm(c - c_P.T, ord=1)
        else:
            c_P = np.sum(P, axis=0, keepdims=1)  # returns a row vector.
            scaling_c = np.divide(c.T, c_P, out=np.zeros_like(c_P), where=c_P != 0)
            #scaling_c = np.exp(np.log(c.T, out = np.ones_like(c.T), where = c.T!=0) - np.log(c_P, out = np.ones_like(c_P), where = c_P!=0))
            #P = P * (c.T / c_P)
            P = scaling_c * P
            r_P = np.sum(P, axis=1, keepdims=1)  # returns a column vector.
            c_P = np.sum(P, axis=0, keepdims=1)  # returns a row vector.
            err[iter + 1] = np.linalg.norm(r - r_P, ord=1) + \
                np.linalg.norm(c - c_P.T, ord=1)

        if compute_otvals == True:
            otvals[iter + 1] = np.sum(round_transpoly(P, r, c) * C)
        if disp_iter == True:
            print("iter = %d \n" % iter)
            print("error = %f" % err[iter])

    if compute_otvals == True:
        return P, otvals, err
    else:
        return P, err


def greenkhorn(A, r, c, C, max_iter=2000, compute_otvals=False, disp_iter=False):
    '''
    implementation of GREENKHORN function in Algorithm 2
    https://papers.nips.cc/paper/2017/file/491442df5f88c6aa018e86dac21d3606-Paper.pdf
    written by: Bilal Riaz bilalria@udel.edu
    inputs:
    -- A: positive matrix of shape n x n
    -- r : source marginal vector from probability simplex of shape m x 1
    -- c : target marginal vector from probability simplex of shape n x 1
    -- max_iter: maximum number sinkhorn iterations.
    -- epsilon: convergence threshold
    -- C: cost matrix for optimal transport
    outputs:
    -- P: Sinkhorn projection matrix m x n
    -- err: sum of source and target errors in sikhorn iteration
    -- OTvals Greenkhorn objective function value obtained after rounding
    '''

    assert np.isclose(np.sum(r), 1), "source points must lie on probability simplex."
    assert np.isclose(np.sum(c), 1), "target points must lie on probability simplex."
    assert np.size(np.shape(A)) == 2, "cost must be a square matrix"
    assert C.shape == A.shape, "Cost matrix and Kernel must be of same shape"

    r = r.reshape([np.size(r), 1])
    c = c.reshape([1, np.size(c)])
    err = np.zeros([max_iter + 1, 1])

    P = np.copy(A)
    P = P / np.sum(P)
    r_P = np.sum(P, axis=1, keepdims=1)  # returns a column vector.
    c_P = np.sum(P, axis=0, keepdims=1)  # return a row vector.

    r_gain = r_P - r + r*np.log(np.divide(r, r_P, out=np.ones_like(r_P), where=r_P != 0))
    c_gain = c_P - c + c*np.log(np.divide(c, c_P, out=np.ones_like(c_P), where=c_P != 0))

    err[0] = np.linalg.norm(r - r_P, ord=1) + np.linalg.norm(c - c_P, ord=1)

    def max_val_index(x):
        max_index = np.argmax(x)
        max_val = np.amax(x)
        return max_val, max_index

    if compute_otvals == True:
        otvals = np.zeros([max_iter + 1, 1])
        otvals[0] = np.sum(round_transpoly(P, r, c) * C)
    
    if disp_iter == True:
        iter = 0
        print("iter = %d \n" % iter)
        print("error = %f" % err[iter])

    for iter in range(0, max_iter):
        r_gain_max, i = max_val_index(r_gain)
        c_gain_max, j = max_val_index(c_gain)

        if r_gain_max > c_gain_max:
            #scaling = np.divide(r[i],r_P[i], out=np.zeros_like(r_P[i]), where=r_P[i] != 0)
            scaling = r[i]/r_P[i]
            old_row = P[i, :]
            new_row = old_row*scaling
            P[i, :]  = new_row

            # renormalize(can also be done implicitly if one wants to optimize)
            P = P/np.sum(P)

            # compute full row and column marginals
            r_P = np.sum(P, axis=1, keepdims=1)
            c_P = np.sum(P, axis=0, keepdims=1)

            # compute gains for each row and column
            r_gain = r_P - r + r*np.log(np.divide(r, r_P, out=np.zeros_like(r_P), where=r_P != 0))
            c_gain = c_P - c + c*np.log(np.divide(c, c_P, out=np.zeros_like(c_P), where=c_P != 0))

            err[iter+1] = np.linalg.norm(r_P-r, ord = 1)+np.linalg.norm(c_P-c, ord = 1)
        
        else:
            #scaling = np.divide(c[:,j], c_P[:,j], out=np.zeros_like(c_P[:,j]), where=c_P[:,j] != 0)
            scaling = c[:,j]/c_P[:,j]
            old_col = P[: , j]
            new_col = old_col*scaling
            P[: , j]  = new_col

            # renormalize(can also be done implicitly if one wants to optimize)
            P = P/np.sum(P)

            # compute full row and column marginals
            r_P = np.sum(P, axis = 1, keepdims=1)
            c_P = np.sum(P, axis = 0, keepdims=1)

            #compute gains for each row and column
            r_gain = r_P - r + r*np.log(np.divide(r, r_P, out=np.zeros_like(r_P), where=r_P != 0))
            c_gain = c_P - c + c*np.log(np.divide(c, c_P, out=np.zeros_like(c_P), where=c_P != 0))

            err[iter+1] = np.linalg.norm(r_P-r, ord = 1)+np.linalg.norm(c_P-c, ord = 1)

        if compute_otvals == True:
            otvals[iter + 1] = np.sum(round_transpoly(P, r, c) * C)
            #otvals[iter + 1] = np.sum(P * C)

        if disp_iter == True:
            os.system('cls')
            print("iter = %d \n" % iter)
            print("error = %f" % err[iter])

    if compute_otvals == True:
        return P, otvals, err
    else:
        return P, err

#%%

m = 500
n = 800

def gauss(q, a, c): return a*np.random.randn(2, q) + \
    np.transpose(np.tile(c, (q, 1)))


X = np.random.randn(2, m)*.3
Y = np.hstack((gauss(int(n/2), .5, [0, 1.6]), np.hstack(
    (gauss(int(n/4), .3, [-1, -1]), gauss(int(n/4), .3, [1, -1])))))
Ly = np.hstack((0 * np.ones((1, int(n/2))), 1 *
                np.ones((1, int(n/4))), 2 * np.ones((1, int(n/4)))))

X = X.T
Y = Y.T
Ly = Ly.T


def normalize(a): return a/np.sum(a)


a = normalize(np.random.rand(m, 1))
b = normalize(np.random.rand(n, 1))
mu = a


def myplot(x, y, ms, col): return plt.scatter(
    x, y, s=ms*20, edgecolors="k", c=col, linewidths=2)


plt.figure(figsize=(10, 7))
plt.axis("off")
for i in range(len(a)):
    myplot(X[i, 0], X[i, 1], a[i]*len(a)*10, 'b')
for j in range(len(b)):
    myplot(Y[j, 0], Y[j, 1], b[j]*len(b)*10, 'r')
plt.xlim(np.min(Y[:, 0])-.1, np.max(Y[:, 0])+.1)
plt.ylim(np.min(Y[:, 1])-.1, np.max(Y[:, 1])+.1)
plt.show()

#%%

def Gibbs_Kernel(M, gamma):
    return np.exp(-M/gamma)


def P_star(u, v, M, gamma):
    K = Gibbs_Kernel(M, gamma)
    P_star = np.diag(np.exp(- alpha / gamma)) @ (K @
                                                 np.diag(np.exp(- beta / gamma)))
    return P_star


def distmat(X, Y):
    return np.sum(X**2, 1)[None, :] + np.sum(Y**2, 1)[:, None] - 2*Y@X.T


def sinkhorn_cost(P, M, gamma):
    return np.sum(P*M) + gamma*np.sum(np.log(P**P))


M = distmat(X, Y).T
L = 5
v0 = 1*np.ones([n, 1])
alpha = 0.001
# u,v = class_reweighted_wasserstein(mu, M, beta0, L, gamma)
# # %%
gamma = 0.1
K = Gibbs_Kernel(M, gamma)

# %%
A, obj_vals, error = sinkhorn(K, a, b, M, max_iter=2000,compute_otvals=True, disp_iter=True)
P = round_transpoly(A, a, b)
# %%
P[P<1e-4] = 0
I, J = np.nonzero(P)
plt.figure(figsize=(10, 7))
plt.axis('off')
Xv = X.T
Yv = Y.T
for k in range(len(I)):
    h = plt.plot(np.hstack((Xv[0, I[k]], Yv[0, J[k]])), np.hstack(
        ([Xv[1, I[k]], Yv[1, J[k]]])), 'k', lw=2)
for i in range(len(a)):
    myplot(Xv[0, i], Xv[1, i], a[i]*len(a)*10, 'b')
for j in range(len(b)):
    myplot(Yv[0, j], Yv[1, j], b[j]*len(b)*10, 'r')
plt.xlim(np.min(Yv[0, :])-.1, np.max(Yv[0, :])+.1)
plt.ylim(np.min(Yv[1, :])-.1, np.max(Yv[1, :])+.1)
plt.show()



#%%
A, obj_vals, error = greenkhorn(K, a, b, M, max_iter=5000, compute_otvals=True, disp_iter=True)

P = round_transpoly(A, a, b)
# %%
P[P < 1e-4] = 0
I, J = np.nonzero(P)
plt.figure(figsize=(10, 7))
plt.axis('off')
Xv = X.T
Yv = Y.T
for k in range(len(I)):
    h = plt.plot(np.hstack((Xv[0, I[k]], Yv[0, J[k]])), np.hstack(
        ([Xv[1, I[k]], Yv[1, J[k]]])), 'k', lw=2)
for i in range(len(a)):
    myplot(Xv[0, i], Xv[1, i], a[i]*len(a)*10, 'b')
for j in range(len(b)):
    myplot(Yv[0, j], Yv[1, j], b[j]*len(b)*10, 'r')
plt.xlim(np.min(Yv[0, :])-.1, np.max(Yv[0, :])+.1)
plt.ylim(np.min(Yv[1, :])-.1, np.max(Yv[1, :])+.1)
plt.show()

# %%

# u,v, Ps = reweighted_sinkhorn(K, a, L, M, alpha=0.01, max_iter=2, epsilon=1e-8,
#                     compute_OTvals=False, Disp_iter=False)

# # %%
# Ps[Ps < 1e-4] = 0
# I, J = np.nonzero(Ps)
# plt.figure(figsize=(10, 7))
# plt.axis('off')
# Xv = X.T
# Yv = Y.T
# for k in range(len(I)):
#     h = plt.plot(np.hstack((Xv[0, I[k]], Yv[0, J[k]])), np.hstack(
#         ([Xv[1, I[k]], Yv[1, J[k]]])), 'k', lw=2)
# for i in range(len(a)):
#     myplot(Xv[0, i], Xv[1, i], a[i]*len(a)*10, 'b')
# for j in range(len(b)):
#     myplot(Yv[0, j], Yv[1, j], b[j]*len(b)*10, 'r')
# plt.xlim(np.min(Yv[0, :])-.1, np.max(Yv[0, :])+.1)
# plt.ylim(np.min(Yv[1, :])-.1, np.max(Yv[1, :])+.1)
# plt.show()

# %%

# %%
# %%
# def reweighted_sinkhorn(K, mu, L, C, alpha, max_iter=2000, epsilon=1e-8, compute_OTvals=False, Disp_iter=False):
#     '''
#     inputs:
#     -- K: positive matrix of shape m x n
#     -- mu : source marginal vector from probability simplex of shape m x 1
#     -- L : target marginal reweighting factor
#     -- max_iter: maximum number sinkhorn iterations.
#     -- epsilon: convergence threshold
#     -- M: cost matrix for optimal transport
#     outputs:
#     -- P: Sinkhorn projection matrix m x n
#     -- err: sum of source and target errors in sikhorn iteration
#     -- OTvals Sinkhorn objective function value obtained after rounding
#     '''

#     assert np.isclose(np.sum(mu), 1), "source points must lie on probability simplex."
#     assert np.size(np.shape(K)) == 2, "cost must be a square matrix"
#     assert C.shape == K.shape, "Cost matrix and Kernel must be of same shape"

#     mu = mu.reshape([np.size(mu), 1])
#     v = np.ones([K.shape[1],1]) / K.shape[1]

#     P = np.copy(K)
#     err = np.zeros([max_iter + 1, 1])
#     r_P = np.sum(P, axis=1, keepdims=1)  # returns a column vector.
#     # c_P = np.sum(P, axis=0, keepdims=1)  # return a row vector.

#     for iter in range(0, max_iter):
#         if np.mod(iter + 1, 2) == 1:
#             den = K@v
#             print(den)
#             u = np.divide(mu, den, out=np.zeros_like(mu), where=den!= 0)
#             print(u)
#             P = u*P


#         else:
#             vinv = np.divide(1, v, out=np.zeros_like(v), where=v != 0)
#             grad = K.T@u - 1/L * vinv
#             v = v - alpha * grad
#             print(v)
#             v[v>1] = 1
#             v[v<0] = 0
#             P =  P * v.T

#     return u, v, P

# if compute_OTvals == True:
#     OTvals[iter + 1] = np.sum(round_transpoly(P, r, c) * C)
# if Disp_iter == True:
#     print("iter = %d \n" % iter)
#     print("error = %f" % err[iter])

# if compute_OTvals == True:
#     return P, OTvals, err
# else:
#     return P, err