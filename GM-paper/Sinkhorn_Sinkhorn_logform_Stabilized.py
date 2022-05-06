#%%
import numpy as np
import scipy as sp


#%%
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
    ratio_r = np.divide(r, r_A, out=np.zeros_like(
        r_A), where=r_A != 0)  # returns column vector
    scaling_r = np.fmin(1, ratio_r)  # returns column vector
    # row scaling
    A = scaling_r * A

    # column sum
    c_A = np.sum(A, axis=0, keepdims=1)  # returns row vector
    ratio_c = np.divide(c.T, c_A, out=np.zeros_like(c_A), where=c_A != 0)  # returns row vector
    scaling_c = np.fmin(1, ratio_c)   # returns row vector
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

def sinkhorn_ot(mu, nu, K, M, max_iter=10000, disp_iter=False):
    '''
    implementation of vanilla-SINKHORN iterations for optimal transport
    written by: Bilal Riaz bilalria@udel.edu
    inputs:
    -- K: positive matrix of shape m x n
    -- mu : source marginal vector from probability simplex of shape m x 1
    -- nu : target marginal vector from probability simplex of shape n x 1
    -- M: Cost Matrix of shape m x n
    -- max_iter: maximum number sinkhorn iterations.
    -- compute_otvals: default varible to computer sikhorn transport objective value 
    -- disp_iter: default variable to display iteration progress
    -- C: cost matrix for optimal transport
    outputs:
    -- u: sinkhorn dual corresponding to source distribution
    -- v: sinkhorn dual corresponding to target distribution
    -- P: Final Transport map
    -- sinkhorn_div: sinkhorn objective value
    '''

    v = np.ones([nu.size, 1])

    mu = mu.reshape(mu.size,1)
    nu = nu.reshape(nu.size,1)
    P = np.copy(K)

    sinkhorn_div = np.zeros([max_iter + 1])
    sinkhorn_div[0] = np.sum(round_transpoly(P, mu, nu)*M)

    if disp_iter == True:
        print("iter = %d \n" % 0)
        print("error = %f" % sinkhorn_div[0])
    
    for i in range(0, max_iter):
        r_P = K@v
        if np.any(r_P == np.nan):
            raise Exception("nan error in K@v")
        u = np.divide(mu, r_P , out=np.zeros_like(r_P), where=r_P != 0)
        #u = mu/r_P
        P = np.diag(np.squeeze(u))@K@np.diag(np.squeeze(v))

        c_P = K.T@u
        if np.any(c_P == np.nan):
            raise Exception("NAN error in K.T@v")
        v = np.divide(nu, c_P , out=np.zeros_like(c_P), where=c_P != 0)
        #v = nu / c_P
        P = np.diag(np.squeeze(u))@K@np.diag(np.squeeze(v))
        
        if np.any(P == np.nan):
            raise Exception("NAN error in P")
        
        sinkhorn_div[i+1] = np.sum(round_transpoly(P, mu, nu)*M)
        
        if disp_iter == True:
            print("iter = %d \n" % i)
            print("error = %f" % sinkhorn_div[i+1])
        
    P = round_transpoly(P, mu, nu)

    return u, v, sinkhorn_div, P


def sinkhorn_block_ascent_ot(mu, nu, K, M, epsilon = 1e-1, max_iter=10000, disp_iter=False):
    '''
    implementation of logform-SINKHORN iterations for optimal transport
    written by: Bilal Riaz bilalria@udel.edu
    inputs:
    -- K: positive matrix of shape m x n
    -- mu : source marginal vector from probability simplex of shape m x 1
    -- nu : target marginal vector from probability simplex of shape n x 1
    -- M: Cost Matrix of shape m x n
    -- max_iter: maximum number sinkhorn iterations.
    -- compute_otvals: default varible to computer sikhorn transport objective value 
    -- disp_iter: default variable to display iteration progress
    -- C: cost matrix for optimal transport
    outputs:
    -- f: sinkhorn dual corresponding to source distribution
    -- g: sinkhorn dual corresponding to target distribution
    -- P: Final Transport map
    -- sinkhorn_div: sinkhorn objective value
    '''
    g_epsilon = np.ones_like(nu)
    P = np.copy(K)

    sinkhorn_div = np.zeros([max_iter + 1])
    sinkhorn_div[0] = np.sum(round_transpoly(P, mu, nu)*M)

    if disp_iter == True:
        print("iter = %d \n" % 0)
        print("error = %f" % sinkhorn_div[0])

    for i in range(0, max_iter):

        f_K = K@np.exp(g_epsilon)
        f_K[f_K == 0] = 1
        f_epsilon = np.log( mu)  - np.log(f_K)
        g_K = K.T @ np.exp(f_epsilon)
 
        g_K[g_K == 0] = 1
        g_epsilon =  np.log(nu) -  np.log(g_K)
        #P = np.diag(np.squeeze(np.exp(f_epsilon)))@K@np.diag(np.squeeze(np.exp(g_epsilon)))
        P = np.exp(-M/epsilon + f_epsilon + g_epsilon.T)
        if np.any(P == np.nan):
            raise Exception("NAN error in P")
        
        sinkhorn_div[i+1] = np.sum(round_transpoly(P, mu, nu)*M)

        if disp_iter == True:
            print("iter = %d \n" % i)
            print("error = %f" % sinkhorn_div[i+1])
        
    P = round_transpoly(P, mu, nu)
    
    return f_epsilon, g_epsilon, sinkhorn_div, P


def sinkhorn_lse_ot(mu, nu, K, M, epsilon=1e-1, max_iter=10000, disp_iter=False):
    '''
    implementation of log-sum-exp  stabilized-sinkhorn iterations for optimal transport
    written by: Bilal Riaz bilalria@udel.edu
    inputs:
    -- K: positive matrix of shape m x n
    -- mu : source marginal vector from probability simplex of shape m x 1
    -- nu : target marginal vector from probability simplex of shape n x 1
    -- M: Cost Matrix of shape m x n
    -- max_iter: maximum number sinkhorn iterations.
    -- compute_otvals: default varible to computer sikhorn transport objective value 
    -- disp_iter: default variable to display iteration progress
    -- C: cost matrix for optimal transport
    outputs:
    -- f: sinkhorn dual corresponding to source distribution
    -- g: sinkhorn dual corresponding to target distribution
    -- P: Final Transport map
    -- sinkhorn_div: sinkhorn objective value
    '''
    mu = mu.reshape(mu.size, 1)
    nu = nu.reshape(nu.size, 1)
    f  = np.zeros_like(mu)
    g = np.zeros_like(nu)

    Mc = M - np.min(M)
    
    def Min_eps(A, eps, dim):
        '''
        A:  must be a 2-D array of shape m x n
        eps: is non-negative regularization parameter
        dim: must be either 0 or 1
        '''
        if dim == 0 or dim == 1:
            z_bar = np.min(A, axis=dim, keepdims=1)
            Z = np.exp(-(A - z_bar)/eps)
            return (z_bar - eps * np.log(np.sum(Z, axis = dim, keepdims=1))).reshape(z_bar.size,1)
        else:
            raise Exception("Invalid argument")

    P = np.copy(K)
    sinkhorn_div = np.zeros([max_iter + 1])
    sinkhorn_div[0] = np.sum(round_transpoly(P, mu, nu)*M)  # np.sum(P*M) 

    if disp_iter == True:
        print("iter = %d \n" % 0)
        print("error = %f" % sinkhorn_div[0])
        
    for i in range(0, max_iter):
        f = Min_eps(Mc - f - g.T, epsilon, 1) + f + epsilon * np.log(mu)
        g = Min_eps(Mc- f - g.T, epsilon, 0) + g + epsilon * np.log(nu)

        # P = np.diag(np.squeeze(np.exp(f/epsilon)))@K@np.diag(np.squeeze(np.exp(g/epsilon)))
        P = np.exp(-(Mc - f - g.T)/epsilon)
        sinkhorn_div[i+1] = np.sum(round_transpoly(P, mu, nu)*M)

        if disp_iter == True:
            print("iter = %d \n" % i)
            print("error = %f" % sinkhorn_div[i])
    
    P = round_transpoly(P, mu, nu)

    return f, g, sinkhorn_div, P


def sinkhorn_stabilized_ot(mu, nu, K, M, tau = 20, epsilon = 1e-1, max_iter=10000, disp_iter=False):
    '''
    
    implementation of stabilized-SINKHORN iterations for optimal transport
    The algorithm-2 from https://epubs.siam.org/doi/abs/10.1137/16M1106018?mobileUi=0&
    written by: Bilal Riaz bilalria@udel.edu
    inputs:
    -- mu : source marginal vector from probability simplex of shape m x 1
    -- nu : target marginal vector from probability simplex of shape n x 1
    -- M: Cost Matrix of shape m x n
    -- max_iter: maximum number sinkhorn iterations.
    -- compute_otvals: default varible to computer sikhorn transport objective value 
    -- disp_iter: default variable to display iteration progress
    -- C: cost matrix for optimal transport
    outputs:
    -- u: sinkhorn dual corresponding to source distribution
    -- v: sinkhorn dual corresponding to target distribution
    -- P: Final Transport map
    -- sinkhorn_div: sinkhorn objective value
    '''

    
    mu = mu.reshape(mu.size, 1)
    nu = nu.reshape(nu.size, 1)

    u = np.ones([mu.size, 1])
    v = np.ones([nu.size, 1])
    

    alpha = np.zeros_like(mu)
    beta = np.zeros_like(nu)

    P = np.copy(K)

    sinkhorn_div = np.zeros([max_iter + 1])
    sinkhorn_div[0] = np.sum(round_transpoly(P, mu, nu)*M)

    if disp_iter == True:
        print("iter = %d \n" % 0)
        print("error = %f" % sinkhorn_div[0])

    for i in range(0, max_iter):
        while(np.max(np.abs(u)) <= tau and np.max(np.abs(v)) <= tau):
            r_P = (K@v)
            u = np.divide(mu, r_P, out=np.zeros_like(r_P), where=r_P != 0)
            c_P = (K.T@u)
            v = np.divide(nu, c_P, out=np.zeros_like(c_P), where=c_P != 0)
            P = np.diag(np.squeeze(u))@K@np.diag(np.squeeze(v))
        
        alpha = alpha + epsilon * np.log(u)
        beta = beta + epsilon * np.log(v)
        u = np.ones([mu.size, 1])
        v = np.ones([nu.size, 1])
        K = np.exp(-(M - alpha - beta.T)/epsilon)
            

        if np.any(P == np.nan):
            raise Exception("NAN error in P")

        sinkhorn_div[i+1] =  np.sum(round_transpoly(P, mu, nu)*M)

        if disp_iter == True:
            print("iter = %d \n" % i)
            print("error = %f" % sinkhorn_div[i+1])

    P = round_transpoly(P, mu, nu)

    return u, v, sinkhorn_div, P
from matplotlib import pyplot as plt
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

# %%
def Gibbs_Kernel(M, gamma):
    return np.exp(-M/gamma)

def P_star(u, v, M, gamma):
    K = Gibbs_Kernel(M, gamma)
    P_star = np.diag(np.exp(- alpha / gamma)) @ (K @
                                                 np.diag(np.exp(- beta / gamma)))
    return P_star

def distmat(X, Y):
    return np.sum(X**2, 1)[None, :] + np.sum(Y**2, 1)[:, None] - 2*Y@X.T

# def sinkhorn(P, M, gamma):
#     return np.sum(P*M) + gamma*np.sum(np.log(P**P))

def objective(alpha, beta, gamma, mu, nu, K):
    temp1 = np.sum(np.diag(np.squeeze(np.exp(-alpha/gamma))) @ K @ np.diag(np.squeeze(np.exp(-beta/gamma))))
    temp2 = np.sum(alpha*mu)  + np.sum(beta*nu)
    return temp1 + temp2


M = distmat(X, Y).T
L = n
v0 = 1*np.ones([n, 1])

gamma = 1e-4
K = Gibbs_Kernel(M, gamma)
mu = a
nu = b
# %%
max_iter = 1000
x, y, obj, A = sinkhorn_ot(mu, nu, K, M, max_iter = max_iter, disp_iter=False)
plt.plot(obj)
print(np.linalg.norm(np.sum(A, axis=1, keepdims=1) - a, ord=1) +
      np.linalg.norm(np.sum(A, axis=0, keepdims=1) - b.T, ord=1))

# %%
alpha, beta, obj1,  B = sinkhorn_block_ascent_ot(
    mu, nu, K, M, epsilon=gamma, max_iter=max_iter, disp_iter=False)
plt.plot(obj1)
print(np.linalg.norm(np.sum(B, axis=1, keepdims=1) - a, ord=1) +
      np.linalg.norm(np.sum(B, axis=0, keepdims=1) - b.T, ord=1))

#%%
alpha1, beta1, obj2, W = sinkhorn_lse_ot(
    mu, nu, K, M, epsilon=gamma, max_iter=max_iter, disp_iter=False)
plt.plot(obj2)
print(np.linalg.norm(np.sum(W, axis=1, keepdims=1) - a, ord=1) +
      np.linalg.norm(np.sum(W, axis=0, keepdims=1) - b.T, ord=1))


#%%
## Not as stable as lse
x1, y1, obj3, W1 = sinkhorn_stabilized_ot(mu, nu, K, M, epsilon = 1e-1, tau=1000,
                       max_iter=max_iter, disp_iter=False)
plt.plot(obj3)
print(np.linalg.norm(np.sum(W1, axis=1, keepdims=1) - a, ord=1) +
      np.linalg.norm(np.sum(W1, axis=0, keepdims=1) - b.T, ord=1))



# %%
