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
        # print(params, eval)
        print("err: {}".format(err))

    params['beta'] = beta
    return params, eval_history

def fit(tauList, T, beta=None, eval=None, max_step=20, eps=1e-4):
    miuList = []
    alphaList = []
    betaList = []
    for i, tau in enumerate(tauList):
        params, _ = fit_one_step(tau, T, beta, eval, max_step, eps)
        # params = fit_single(tau, T, beta, max_step, eps, eval)
        miuList.append(params['miu'])
        alphaList.append(params['alpha'])
        betaList.append(params['beta'])
    miu = np.mean(miuList, axis=0)
    alpha = np.mean(alphaList, axis=0)
    beta = np.mean(betaList, axis=0)
    return {'miu': miu.tolist(), 'alpha': alpha.tolist(), 'beta': beta.tolist()}

def run_fitting(args, params, tauList):
    miu, alpha, beta, T = params['miu'], params['alpha'], params['beta'], params['T']

    fitting_result = fit(tauList, T, beta=None, eval=params, max_step=args.fit_epochs)
    print(fitting_result)
    fitting_result['mean_relative_error'] = evaluation(params, fitting_result)
    utils.dump_all([fitting_result], ["fitting_result"])
    print("MRE is: {}".format(fitting_result['mean_relative_error']))
