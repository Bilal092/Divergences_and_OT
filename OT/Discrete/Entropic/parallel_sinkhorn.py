import numpy as np
from scipy.special import logsumexp

def parallel_sinkhorn(MU, NU, M, gamma, max_iter=10000, conv_eval_iter = 10, tol=1e-9, eval_costs=False):
    '''
    Parallelized log-domain Sinkhorn algorithm for multiple OT problems.
    
    Parameters:
    - MU: (N, m) array of source histograms (rows sum to 1)
    - NU: (N, n) array of target histograms (rows sum to 1)
    - M: (m, n) cost matrix
    - gamma: float, entropic regularization coefficient
    - max_iter: int, maximum number of iterations
    - tol: float, convergence tolerance

    Returns:
    - Alpha: (N, m) dual potentials for MU
    - Beta:  (N, n) dual potentials for NU
     with optional primal and dual costs if eval_costs is True.
    - primal_costs: list of primal costs at each iteration (if eval_costs is True)  
    - dual_costs: list of dual costs at each iteration (if eval_costs is True)
    written by Bilal Riaz 2025 <bilalria@udel.edu>
    '''
    N, m = MU.shape
    _, n = NU.shape

    # Initialize dual potentials
    Alpha = np.zeros((N, m))
    Beta = np.zeros((N, n))

    # Normalize and stabilize cost
    M = M / np.max(M)

    # Precompute log-mass terms
    neg_gamma_log_MU = - gamma * np.log(MU + 1e-20)
    neg_gamma_log_NU = -gamma * np.log(NU + 1e-20)
    
    if eval_costs:
        primal_costs = []
        dual_costs = []

    for step in range(max_iter):
        Alpha_prev = Alpha.copy()
        # Alpha update: shape (N, m)
        Alpha = neg_gamma_log_MU + gamma * logsumexp(-(Beta[:, None, :] + M[None, :, :]) / gamma, axis=2)

        # Beta update: shape (N, n)
        Beta = neg_gamma_log_NU + gamma * logsumexp(-(Alpha[:, :, None] + M[None, :, :]) / gamma, axis=1)
        
        if eval_costs:
            # Compute costs for convergence evaluation
            dual_cost = np.sum(MU*Alpha) + np.sum(NU*Beta)
            primal_cost = np.sum(np.exp(-(Alpha[:, :, None] + Beta[:, None, :] + M) / gamma) * M[None, :, :])
            primal_costs.append(primal_cost)
            dual_costs.append(dual_cost)

        # Convergence check
        if step % conv_eval_iter == 0:
            delta = np.max(np.abs(Alpha - Alpha_prev))
            if delta < tol:
                break

    if eval_costs:
        return Alpha, Beta, primal_costs, dual_costs
    else:
        return Alpha, Beta
