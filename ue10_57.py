#!/usr/bin/env python3

import numpy as np

d = lambda a,b: np.mean((a - b)**2)
t1 = lambda x: x[0]**2
t2 = lambda x: ((x[0] + x[1]) / 2)**2
t3 = lambda x: (x[0]**2 + x[1]**2) / 2
t4 = lambda x: x[0] * x[1]
p1 = lambda m,s: 4*m**2*s**2+3*s**4
p2 = lambda m,s: 2*m**2*s**2+6/8*s**4
p3 = lambda m,s: 2*m**2*s**2+2*s**4
p4 = lambda m,s: 2*m**2*s**2+s**4

mu = 7
sig = 3
m = 10**6
x = np.random.normal(loc=mu, scale=3, size=(2, m))

ts = [t1, t2, t3, t4]
ps = [p1, p2, p3, p4]

for t, p in zip(ts, ps):
    print(d(mu**2, t(x)), p(mu, sig))

