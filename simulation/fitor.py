import numpy as np
import random
import itertools
import utils

def evaluation(real_parameters, fitted_parameters):
    """
    return the mean relative error
    """
    U1, U2 = real_parameters['miu'], fitted_parameters['miu']
    A1, A2 = real_parameters['alpha'], fitted_parameters['alpha']
    w1, w2 = real_parameters['beta'], fitted_parameters['beta']
    X1, X2 = [], []
    X1.extend(U1)
    X2.extend(U2)
    for a1, a2 in zip(A1, A2):
        X1.extend(a1)
        X2.extend(a2)
    if w1 != w2:
        X1.append(w1)
        X2.append(w2)
    X1, X2 = np.array(X1), np.array(X2)
    down = X1.copy()
    down[np.where(down == 0)] = 1.0
    return np.mean(np.abs(X1-X2) / down)

def fit_one_step(tau, T, beta=None, eval=None, max_step=50, eps=1e-4):
    T = max([max(t) for t in tau])

    M = len(tau)
    needEstimateBeta = beta is None

    miu = np.random.uniform(0, 0.1, size=M)
    alpha = np.random.uniform(0, 0.1, size=(M, M))
    if beta is None:
        beta = np.random.uniform(0, 1, size=1)
    params = {'miu':miu, 'alpha':alpha, 'beta':beta}

    e = []
    # for i, t in enumerate(tau):
    #     for j in t:
    #         e.append((j, i))
    # e = sorted(e, key=lambda eve: eve[0])
    for index, seq in enumerate(tau):
        e.extend(zip(seq, itertools.repeat(index)))
    e = sorted(e, key=lambda event: event[0])
    n = len(e)
    p = np.zeros((n,n))

    eval_history = []


    for s in range(max_step):
        print("Step:{}".format(s))
        # E-step
        for i in range(n):
            for j in range(i):
                p[i, j] = alpha[e[i][1], e[j][1]] * np.exp(-beta * (e[i][0] - e[j][0]))
            p[i, i] = miu[e[i][1]]
            p[i] = p[i] / np.sum(p[i])

        # M-step

        # Update miu
        for i in range(M):
            miu[i] = 1./T * sum([p[j,j] for j in range(n) if e[j][1] == i])
        # Update alpha
        for u in range(M):
            for v in range(M):
                up, down = 0, 0
                for i in range(n):
                    if e[i][1] != u:
                        continue
                    for j in range(i):
                        if e[j][1] == v:
                            up += p[i,j]
                for j in range(n):
                    if e[j][1] == v:
                        down += (1 - np.exp(-beta*(T-e[j][0]))) / beta
                alpha[u, v] = up / down


        # update beta
        if needEstimateBeta:
            up, down = 0, 0
            for i in range(n):
                for j in range(i):
                    up += p[i,j]
                    down += (e[i][0] - e[j][0]) * p[i,j]
            beta = up / down

        err = evaluation(eval, params)
        print(params, eval)
        print("err: {}".format(err))

    return params, eval_history

def fit_single(seqs, T, w=None, max_step=30, eps=1e-5, realParams=None):
    """
    inference the multi-hawkes point process parameters
    :param seqs: the list of event sequences, M = len(seqs) is the dimension of the process
    :param T: the data was sampled from time interval [0, T]
    :param w: when w is None, we inference w, otherwise we regard w is known
    :param max_step: the maximum number of steps
    :param eps: the epsilon, when the 2-norm of change is less or equal to epsilon, stop iteration
    :return: parameters, {'U': U, 'A', A, 'w': w}, where U is the background intensity
        and A is the infectivity matrix. A[n][m] is the infectivity of n to m
    """
    T = max([max(seq) for seq in seqs])
    print(T)
    M = len(seqs)
    w_known = w is not None
    U = np.random.uniform(0, 0.1, size=M)
    A = np.random.uniform(0, 0.1, size=(M, M))
    if not w_known:
        w = np.random.uniform(0, 1, size=1)

    e = []
    for index, seq in enumerate(seqs):
        e.extend(zip(seq, itertools.repeat(index)))
    e = sorted(e, key=lambda event: event[0])
    N = len(e)
    p = np.zeros((N, N))

    for step in range(max_step):
        old_U = np.copy(U)
        old_A = np.copy(A)
        old_w = np.copy(w)

        # update p
        for i in range(N):
            for j in range(i):
                p[i, j] = old_A[e[i][1], e[j][1]] * np.exp(-w * (e[i][0] - e[j][0]))
            p[i, i] = old_U[e[i][1]]
            p[i] = p[i] / np.sum(p[i])

        # update U
        for d in range(M):
            U[d] = sum([p[i, i] for i in range(N) if e[i][1] == d]) / T

        # update A
        for du in range(M):
            for dv in range(M):
                up, down = 0.0, 0.0
                for i in range(N):
                    if e[i][1] != du: continue
                    for j in range(i):
                        if e[j][1] != dv: continue
                        up += p[i, j]
                for j in range(N):
                    if e[j][1] != dv: continue
                    down += (1.0 - np.exp(-old_w * (T - e[j][0]))) / old_w
                A[du, dv] = up / down

        # update w
        if not w_known:
            up, down = 0.0, 0.0
            for i in range(N):
                for j in range(i):
                    pij = p[i, j]
                    up += pij
                    down += (e[i][0] - e[j][0]) * pij
            w = up / down
        else:
            w = old_w

        eva = evaluation(realParams, {'miu': U, 'alpha': A, 'beta': w})
        print("\nStep  {} EVA {}".format(step, eva), end="")
    print()
    return {'miu': U, 'alpha': A, 'beta': w}

def fit(tauList, T, beta=None, eval=None, max_step=20, eps=1e-4):
    miuList = []
    alphaList = []
    betaList = []
    for i, tau in enumerate(tauList):
        #params, _ = fit_one_step(tau, T, beta, eval, max_step, eps)
        params = fit_single(tau, T, beta, max_step, eps, eval)
        miuList.append(params['miu'])
        alphaList.append(params['alpha'])
        betaList.append(params['beta'])
    miu = np.mean(betaList, axis=0)
    alpha = np.mean(alphaList, axis=0)
    beta = np.mean(betaList, axis=0)
    return {'miu': miu.tolist(), 'alpha': alpha.tolist(), 'beta': beta.tolist()}

def run_fitting(args, params, tauList):
    miu, alpha, beta, T = params['miu'], params['alpha'], params['beta'], params['T']

    fitting_result = fit(tauList, T, beta=None, eval=params)
    print(fitting_result)
    fitting_result['mean_relative_error'] = evaluation(params, fitting_result)
    utils.dump_all([fitting_result], ["fitting_result"])
    print("MRE is: {}".format(fitting_result['mean_relative_error']))
