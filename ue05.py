#!/usr/bin/env python3

import numpy as np
from scipy import stats, integrate, optimize
import matplotlib.pyplot as plt

data = np.array([
    1.1, 1.1, -1.0, 5.5, 3.5, 9.6, -1.2, 10.0, -98.1, 0.1, 6.1, 0.0,
    -4.0, 1.6, -9.2, -9.7, 7.2, -0.2, -2.2, -1.5, 0.0, 32.0, 10.0, 11.3
])
w = 1
n = len(data)

def pri_gen():
    return lambda x: 1/2 * np.exp(x) if x < 0 else 1 - 1/2 * np.exp(-x)
def emp_gen(data):
    return lambda x: np.sum(data < x) / len(data)
def post_gen(pri, emp, p):
    return lambda x: p * pri(x) + (1-p) * emp(x)

pri = pri_gen()
emp = emp_gen(data)
post = post_gen(pri, emp, w / (w + n))

x = np.linspace(-5, 5, 1000)
pri_y = np.vectorize(pri)(x)
emp_y = np.vectorize(emp)(x)
post_y = np.vectorize(post)(x)

plt.plot(x, pri_y, label='prior')
plt.plot(x, emp_y, label='emp')
plt.plot(x, post_y, label='post')
plt.legend()
plt.show()

def inverse(f, y):
    x0 = -1e2
    x1 = +1e2
    for _ in range(100):
        x = (x0 + x1) / 2
        if f(x) < y:
            x0 = x
        else:
            x1 = x
    return x

post_alpha = np.vectorize(lambda y: inverse(post, y))(np.random.random(10000))
plt.plot(x, 1/2 * np.exp(-np.abs(x)), label='pri')
plt.hist(post_alpha, bins=100, density=True, label='post')
plt.legend()
plt.show()

