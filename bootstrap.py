"""Design-based bootstrap: 95% intervals, rank intervals and profile probabilities
(Table 2, Fig 1; Additional file 1: Tables S3, S4).

Rao-Wu rescaled bootstrap over the stratum and primary sampling unit (PSU) identifiers
released in the SHARE gv_weights modules. Life tables are held fixed.
"""

import numpy as np
import pandas as pd


def design_keys(d):
    """Household-level sampling strata and PSUs.

    d has one row per respondent-wave with country, hhid, stratum1 and psu. A household
    keeps one stratum and one PSU across waves (its first non-missing value); a household
    without a PSU identifier is its own PSU; strata with a single PSU are pooled within
    country.
    """
    d = d.copy()
    d["hh"] = d.country + "::" + d.hhid
    for c in ["stratum1", "psu"]:
        first = (d.loc[d[c].notna(), ["hh", c]].sort_values(c, kind="stable")
                 .drop_duplicates("hh").set_index("hh")[c])
        d[c] = d.hh.map(first)
    d["stratum"] = d.country + "|" + d.stratum1.fillna("_NOSTR")
    d["psu"] = (d.stratum + "|" + d.psu).where(d.psu.notna(), d.country + "|hh|" + d.hhid)
    single = d.groupby("stratum").psu.transform("nunique").eq(1)
    d.loc[single, "stratum"] = d.country + "|_SINGLETON_POOL"
    return d


class RaoWu:
    """In each stratum draw n_h - 1 of its n_h PSUs with replacement and rescale the
    multiplicities by n_h / (n_h - 1); every record of a PSU gets its multiplier."""

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


def replicates(design, statistic, n_reps=1000, seed=20260922):
    """Evaluate statistic(multiplier) in each replicate. A replicate for which it returns
    None (an empty country-age cell) is discarded, for all countries at once."""
    rng = np.random.default_rng(seed)
    out = [statistic(design.draw(rng)) for _ in range(n_reps)]
    return [x for x in out if x is not None]


def percentile_ci(x):
    """x: replicates x countries."""
    return np.quantile(x, 0.025, axis=0), np.quantile(x, 0.975, axis=0)


def rank_interval(x):
    """Ranks recomputed in every replicate (1 = highest); 2.5th and 97.5th percentiles."""
    r = (-x).argsort(1).argsort(1) + 1
    return np.floor(np.quantile(r, 0.025, axis=0)).astype(int), np.ceil(np.quantile(r, 0.975, axis=0)).astype(int)


def profile_probability(labels, replicate_labels):
    """Share of replicates that reproduce each country's profile."""
    return (np.asarray(replicate_labels) == labels).mean(0)
