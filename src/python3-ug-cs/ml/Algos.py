import math
import numpy as np

# populate list of samples from desired distribution
def invTrans(file, n, dist, **kwargs):
    # input: file with uniform sample and sample size
    # input: dist (expo or gamma)
    # input: additional arguments/parameters
    samples = []
    uniform = []
    with open(file,'r') as f:
        for _ in range(n):
            uniform.append(float(f.readline()))
    if dist == 'exponential':
        # inverse distribution for expo
        def invF(x): return - math.log(1 - x) / kwargs['lambda']
    else:
        # inverse Cauchy: https://tinyurl.com/m64td84a
        def invF(x): return kwargs['gamma'] * math.tan(math.pi * (x - 0.5)) + kwargs['peak_x']
    for u in uniform:
        samples.append(invF(u))
    assert len(samples) == n
    return samples

# confidence intervals
def confInt(samples, variance, epsilons):
    # input: samples, variance, threshold (epsilon)
    n = len(samples)
    mean = sum(samples) / n
    std = math.sqrt(variance) / math.sqrt(n)
    # Chebyshev's Inequality: https://tinyurl.com/3kpew3wn
    def chebyshev(e): return (std / e) ** 2 if (std / e) < 1 else 1
    sample_mean = mean
    deltas = [chebyshev(e) for e in epsilons]
    assert len(deltas) == len(epsilons)
    return sample_mean, deltas

# Doolittle's LU decomposition
def LU(A: np.array):
    # input: A (n * n)
    L, U = np.zeros(A.shape), np.zeros(A.shape)
    # Doolittle's LUP Decomposition: https://tinyurl.com/vuvm7343
    n = len(A)
    for j in range(n):
        L[j][j] = 1.0
        for i in range(j + 1):
            U[i][j] = A[i][j] - np.sum(np.dot(L[i][:i],U.T[j][:i]))
        for i in range(j,n):
            L[i][j] = (A[i][j] - np.sum(np.dot(L[i][:j],U.T[j][:j]))) / U[j][j]
    assert L.shape == A.shape and U.shape == A.shape, "Return matrices of the same shape as A"
    return L, U

# pairwise ranking loss
def pwRankingLoss(P, N):
    # input: P (n1 * 1), N (n2 * 1)
    loss = 0
    # https://tinyurl.com/55rvxr57
    loss = max(loss,np.max(np.subtract.outer(N,P)))
    return loss

# Discrete Cosine Transform
def DCT(X: np.ndarray):
    # input: X (m * n)
    # https://tinyurl.com/bdfuem8c
    m, n = X.shape
    M = np.zeros((m,m))
    N = np.zeros((n,n))
    for i in range(m):
        for j in range(m):
            M[i][j] = np.cos((np.pi / m) * (j + 0.5) * i)
    for i in range(n):
        for j in range(n):
            N[i][j] = np.cos((np.pi / n) * (i + 0.5) * j)
    DCT = 4 * np.matmul(M,np.matmul(X,N))
    return DCT

# query-based document ranking
def docRank(D: np.ndarray, Q: np.ndarray):
    # input: D (n vectors * w length * k slices), Q (m vectors * w length)
    _, _, k = D.shape
    R = np.zeros((k))
    for i in range(k):
        # https://tinyurl.com/2p988rm7
        slice = D[:,:,i]
        # https://tinyurl.com/4a5tbx99
        R[i] = np.sum(np.amax(np.matmul(slice,Q.T), axis = 0))
    hash = {i:R[i] for i in range(k)}
    hash = dict(sorted(hash.items(), key = lambda x: x[1], reverse = True))
    R = np.array(list(hash.keys()))
    return R

# Markov chain state probabilities
def stateProb(M: np.ndarray,S_T: int, T: int):
    # input: M (n * n), terminal state, horizon, initial state
    # https://tinyurl.com/yxj9bmk4
    P = np.diag(np.linalg.matrix_power(M,T))
    return P[S_T]