#!/usr/bin/env python3

import numpy as np
from scipy import stats, integrate
import matplotlib.pyplot as plt

data = np.array([
    12.96, 12.10, 6.36, 10.08, 18.80, 7.38, 11.83, 24.02, 2.31, 11.42
])

print(np.mean(data), np.std(data), np.median(data))

def likelihood_gen(data, mu=None, sig=None):
    return lambda x: np.prod(stats.cauchy.pdf(data, loc=x if mu is None else mu, scale=x if sig is None else sig))
def post_gen(like, pri, norm=None):
    result = lambda x: like(x) * pri(x)
    if norm is not None:
        n = integrate.quad(result, norm[0], norm[1])[0]
        print(n)
        return lambda x: result(x) / n
    return result

pri1 = lambda x: 1
like1 = likelihood_gen(data, sig=np.std(data))
post1 = post_gen(like1, pri1, norm=(-10e2, 10e2))

x = np.linspace(-50, 50, 1000)
post1_y = np.vectorize(post1)(x)

plt.plot(x, post1_y, label='post1')
plt.legend()
plt.show()

pri2 = lambda x: 1/x
like2 = likelihood_gen(data, mu=np.median(data))
post2 = post_gen(like2, pri2, norm=(10e-2, 10e2))

x = np.linspace(10e-2, 10, 1000)
post2_y = np.vectorize(post2)(x)

plt.plot(x, post2_y, label='post2')
plt.legend()
plt.show()

