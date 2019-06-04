#!/usr/bin/env python3

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def exact(n, x):
    return lambda y: scipy.stats.beta.pdf(y, 1+np.sum(x), 1+n*len(x)-np.sum(x))
def approx(n, x):
    p_est = np.mean(x)/n
    return lambda y: scipy.stats.norm.pdf(y, p_est, np.sqrt(p_est*(1-p_est)/(n*len(x))))

ns = [5, 20, 100]
p = 0.3
m = 1
xs = [np.random.binomial(n, p, size=m) for n in ns]

ps = np.linspace(0, 1, 1000)
for n, x in zip(ns, xs):
    plt.axvline(x=p)
    plt.plot(ps, exact(n, x)(ps), label='exact')
    plt.plot(ps, approx(n, x)(ps), label='approx')
    plt.legend()
    plt.show()

