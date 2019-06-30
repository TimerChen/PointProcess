import random
import math
import copy
import numpy as np

import utils
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from math import pow, e, log
import bisect
import scipy.stats

def simulate(miu, alpha, beta, T):
    """
        miu:    M*1
        alpha:  M*M
        beta:   1*1
    """
    M = len(miu)
    tau = [[] for _ in range(M)]
    s = 0.0

    def get_lambda(s, m):
        lamb = miu[m]
        for i in range(M):
            for j in tau[i]:
                lamb += alpha[m][i] * math.exp(-beta * (s - j))
        return lamb

    while s < T:
        lamb = sum([get_lambda(s,m) for m in range(M)])
        u = 1 - random.random()
        s += -math.log(u) / lamb
        if s > T:
            break
        D = 1 - random.random()
        lambList = [get_lambda(s, m) for m in range(M)]
        lamb_sum = sum(lambList)
        if D * lamb <= sum(lambList):
            for k in range(M):
                if D * lamb <= sum(lambList[:k+1]):
                    tau[k].append(s)
                    break
    return tau

def draw_qq_plot(parameters, seqs, file=None):
    """
    draw the quantile-quantile plot
    :param parameters: parameters of the hawkes process
    :param seqs: list of sequence
    :param file: None or string, if file is None, draw and show on screen, otherwise save the image using file as filename
    """
    w = parameters['beta']
    U = parameters['miu']
    A = parameters['alpha']
    samples = []
    seqs = copy.deepcopy(seqs)
    for seq in seqs:
        seq.insert(0, 0.0)
    M = len(seqs)
    R = [[None for _ in range(M)] for __ in range(M)]
    for m in range(M):
        for n in range(M):
            R[m][n] = [0.0 for _ in range(len(seqs[m]))]
            R[m][n][0] = 0.0
            for k in range(1, len(seqs[m])):
                R[m][n][k] = pow(e, -w * (seqs[m][k] - seqs[m][k-1])) * R[m][n][k-1]
                begin = bisect.bisect_left(seqs[n], seqs[m][k-1])
                end = bisect.bisect_right(seqs[n], seqs[m][k])
                for i in range(begin, end):
                    R[m][n][k] += pow(e, -w * (seqs[m][k] - seqs[n][i]))
    for m in range(M):
        for k in range(1, len(seqs[m])):
            s = U[m] * (seqs[m][k] - seqs[m][k-1])
            for n in range(M):
                s += (A[m][n] / w) * (1.0 - pow(e, -w * (seqs[m][k] - seqs[m][k-1]))) * R[m][n][k-1]
                begin = bisect.bisect_left(seqs[n], seqs[m][k-1])
                end = bisect.bisect_right(seqs[n], seqs[m][k])
                for i in range(begin, end):
                    s += (A[m][n] / w) * (1.0 - pow(e, -w * (seqs[m][k] - seqs[n][i])))
            samples.append(s)
    scipy.stats.probplot(x=samples, dist=scipy.stats.expon(), fit=False, plot=plt)
    if file is not None:
        plt.savefig(file)
    else:
        plt.show()
    plt.close()

def get_A(params, tau):
    miu, alpha, beta, T = params['miu'], params['alpha'], params['beta'], params['T']
    M = len(tau)
    A = [[[] for j in range(M)] for i in range(M)]

    # calculate
    for m in range(M):
        for n in range(M):
            A[m][n] = [0.0 for i in range(len(tau[m]))]
            for i in range(1, len(tau[m])):
                A[m][n][i] = math.exp(-beta*(tau[m][i] - tau[m][i-1])) * A[m][n][i-1]
                left = bisect.bisect_left(tau[n], tau[m][i-1])
                right = bisect.bisect_right(tau[n], tau[m][i])
                for t in tau[n][left:right]:
                    A[m][n][i] += math.exp(-beta*(tau[m][i] - t))
    return A

def get_bigLambda(params, tau):
    miu, alpha, beta, T = params['miu'], params['alpha'], params['beta'], params['T']
    M = len(tau)
    oldTau = tau
    for i in range(M):
        tau.append([0.0,] + oldTau[i])

    A = get_A(params, tau)

    sample_data = []
    # calculate for
    for m in range(M):
        for ii in range(len(tau[m])-1):
            i = ii+1
            dura = (tau[m][i] - tau[m][i-1])
            # First term
            tmp = miu[m] * dura

            # Second term (No j)
            for n in range(M):
                tmp += (alpha[m][n] / beta) * (1 - math.exp(-beta * dura)) * A[m][n][i-1]

            # Third term
            for n in range(M):
                left = bisect.bisect_left(tau[n], tau[m][i-1])
                right = bisect.bisect_right(tau[n], tau[m][i])
                for t in tau[n][left:right]:
                    tmp += (alpha[m][n] / beta) * (1 - math.exp(-beta * (tau[m][i] - t)))
            sample_data.append(tmp)

    return sample_data


def draw_result(params, tau, dirname):
    sample_data = get_bigLambda(params, tau)
    scipy.stats.probplot(x = sample_data, dist = scipy.stats.expon(),plot = plt)
    plt.savefig(os.path.join(dirname, "qq_plot.svg"))
    plt.close()

def run_simu(args, params, runTimes = 1):
    miu, alpha, beta, T = params['miu'], params['alpha'], params['beta'], params['T']

    # tau = simulation(miu, alpha, beta, T)
    tau = simulate(miu, alpha, beta, T)

    # deal with the first result
    draw_result(params, tau, args.result)

    # simulate for step2
    # tauList = [simulation(miu, alpha, beta, T) for i in range(runTimes)]
    tauList = [simulate(miu, alpha, beta, T) for i in range(runTimes)]

    # dump the first result
    utils.dump_all([params, tauList], ["parameters", "simulation_data"])

    return tauList
