import random
import math
import numpy as np
import scipy.stats as st
from collections import Counter
import matplotlib.pyplot as plt
import functools
from sklearn.neighbors import KernelDensity

def flip(weight=0.5):
    return random.choices((True, False), weights=(weight, 1-weight))[0]

def repeat(n, f):
    return [f() for _ in range(n)]

uniformDraw = random.choice

def mem(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_func


class Bernoulli:
    def __init__(self, params_dict):
        self.dist = st.bernoulli(params_dict['p'])

    def sample(self):
        return bool(self.dist.rvs())

    def score(self, value):
        return self.dist.logpmf(value)

def bernoulli(p):
    return Bernoulli({'p':p}).sample()


class Gaussian:
    def __init__(self, params_dict):
        self.dist = st.norm(params_dict['mu'], params_dict['sigma'])

    def sample(self):
        return self.dist.rvs()

    def score(self, value):
        return self.dist.pdf(value)

def gaussian(mu,sigma):
    return Gaussian({'mu':mu,'sigma':sigma}).sample()


class Infer:
    def __init__(self, params=None, f=None):
        if not params:
            self.params = {'method': 'forward', 'samples': 1000}
        else:
            self.params = params
        self.sample_draws = [f() for _ in range(params['samples'])]
        self.sample_max = math.ceil(max(self.sample_draws))
        self.sample_min = math.floor(min(self.sample_draws))
        self.sample_unique = np.unique(self.sample_draws)
        self.kde = KernelDensity(bandwidth=0.3, kernel='gaussian')
        self.kde.fit(np.array(self.sample_draws).reshape(-1,1))

    def sample(self):
        return(self.kde.sample(n_samples=1)[0][0])

    def viz(self):
        if isinstance(self.sample_unique[0], np.int64):
            plt.hist(self.sample_draws, bins=self.sample_unique, density=True)
        else:
            x = np.array(self.sample_draws).reshape(self.params['samples'],1)
            x_d = np.linspace(self.sample_min, self.sample_max, self.params['samples'])
            logprob = self.kde.score_samples(x_d.reshape(self.params['samples'],1))
            plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
            plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
            plt.ylim(0, 1)
        plt.show()
