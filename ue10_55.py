#!/usr/bin/env python3

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def exact(x):
    return lambda y: scipy.stats.beta.pdf(y, 1+len(x), 1+np.sum(x)-len(x))
def approx(x):
    m = np.mean(x)
    return lambda y: scipy.stats.norm.pdf(y, 1/m, np.sqrt((m-1)/(m**3*len(x))))

ns = [5, 20, 100]
p = 0.7
xs = [np.random.geometric(p, size=n) for n in ns]

ps = np.linspace(0, 1, 1000)
for n, x in zip(ns, xs):
    plt.axvline(x=p)
    plt.plot(ps, exact(x)(ps), label='exact')
    plt.plot(ps, approx(x)(ps), label='approx')
    plt.legend()
    plt.show()

