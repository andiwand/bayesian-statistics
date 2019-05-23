#!/usr/bin/env python3

import numpy as np
from scipy import stats, integrate, optimize
import matplotlib.pyplot as plt

g21 = lambda x: stats.gamma.pdf(x, 7, scale=1/4.378)
g11 = lambda x: stats.gamma.pdf(x, 6, scale=1/4.378)
g42 = lambda x: stats.gamma.pdf(x, 9, scale=1/5.378)
eq_ = lambda x: stats.gamma.pdf(x, 6, scale=1/3.378)
eq_norm = integrate.quad(eq_, 0, 5)[0]
eq = lambda x: eq_(x) / eq_norm

xs = np.linspace(0, 5, 1000)
plt.plot(xs, g21(xs), label='g21')
plt.plot(xs, g11(xs), label='g11')
plt.plot(xs, g42(xs), label='g42')
plt.plot(xs, eq(xs), label='eq')
#plt.legend()
#plt.show()


fs = [g21, g11, g42]
B = np.array([[integrate.quad(lambda x: f1(x)*f2(x), 0, np.inf)[0] for f1 in fs] for f2 in fs])
#b = np.array([[1] for f in fs])
#B = np.block([[B, b], [b.T, 0]])
c = np.array([integrate.quad(lambda x: 1/5*f(x), 0, 5)[0] for f in fs])
#c = np.block([c, np.ones((1))])
p = np.linalg.solve(B, c)
print(p)

def f(x):
    y = np.dot(B, x) - c
    return np.dot(y, y)
cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1})
opt = optimize.minimize(f, [1/3, 1/3, 1/3], bounds=[(0, 1), (0, 1), (0, 1)], method='SLSQP', constraints=cons)
print(opt)
p = opt.x

mix = lambda x, p: p[0] * stats.gamma.pdf(x, 2, scale=1/1) + \
    p[1] * stats.gamma.pdf(x, 1, scale=1/1) + \
    p[2] * stats.gamma.pdf(x, 4, scale=1/2)
print(integrate.quad(lambda x: mix(x, p), 0, np.inf)[0])
xs = np.linspace(0, 5, 1000)
plt.plot(xs, mix(xs, p), label='mix')
plt.legend()
plt.show()

