#!/usr/bin/env python3

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def exact(x):
    return lambda y: scipy.stats.gamma.pdf(y, 1+len(x), scale=1/(1+np.sum(x)))
def approx(x):
    tau_est = np.mean(x)
    return lambda y: scipy.stats.norm.pdf(y, tau_est, np.sqrt(tau_est**2/len(x)))

ns = [5, 20, 100]
tau = 1
xs = [np.random.exponential(tau, size=n) for n in ns]

taus = np.linspace(0, 2, 1000)
for n, x in zip(ns, xs):
    plt.axvline(x=tau)
    plt.plot(taus, exact(x)(taus), label='exact')
    plt.plot(taus, approx(x)(taus), label='approx')
    plt.legend()
    plt.show()

