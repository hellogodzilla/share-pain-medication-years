"""Rao-Wu rescaled bootstrap on the SHARE design.

Per stratum with n_h > 1 PSUs: draw n_h - 1 PSUs with replacement, scale the counts by
n_h / (n_h - 1). All records of a PSU (both waves) share the multiplier. Single-PSU
strata keep 1.
"""

import numpy as np
import pandas as pd


class RaoWu:
    def __init__(self, d):
        keys = pd.MultiIndex.from_frame(d[["stratum", "psu"]])
        self.psu_of_record, psus = pd.factorize(keys, sort=True)
        strata = np.asarray(psus.get_level_values(0))
        self.strata = [np.flatnonzero(strata == s) for s in sorted(set(strata))]
        self.n_psu = len(psus)

    def draw(self, rng):
        m = np.ones(self.n_psu)
        for idx in self.strata:
            n = len(idx)
            if n > 1:
                m[idx] = np.bincount(rng.integers(0, n, n - 1), minlength=n) * n / (n - 1)
        return m[self.psu_of_record]


def multipliers(d, n_reps, seed):
    design = RaoWu(d)
    rng = np.random.default_rng(seed)
    for _ in range(n_reps):
        yield design.draw(rng)


def interval(x):
    return np.quantile(x, 0.025, axis=0), np.quantile(x, 0.975, axis=0)


def ranks(x):
    return (-x).argsort(-1).argsort(-1) + 1


def rank_interval(x):
    r = ranks(x)
    return np.floor(np.quantile(r, 0.025, axis=0)).astype(int), np.ceil(np.quantile(r, 0.975, axis=0)).astype(int)


def spearman(a, b):
    # also works row by row on replicate arrays
    a = pd.DataFrame(np.atleast_2d(a)).rank(axis=1).to_numpy()
    b = pd.DataFrame(np.atleast_2d(b)).rank(axis=1).to_numpy()
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return r if len(r) > 1 else r[0]
