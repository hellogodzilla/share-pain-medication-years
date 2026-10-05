"""Standardisation for measured health and functional status by G-computation
(Additional file 1: Table S7, Supplementary methods S5).

Two weighted logistic models with country indicators, age group and sex: moderate-to-
severe pain (P), and no weekly pain medication among those in pain (U, also adjusted for
severe pain). Every respondent is assigned to each country in turn; predictions are
averaged within age group over the pooled sample and passed through the Sullivan
calculation. Unadjusted models are fitted on the same complete cases without the
indicators. Intervals: the Rao-Wu bootstrap of bootstrap.py, refitting all models in
every replicate (seed 20260921).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import spearmanr

from sullivan import AGE_GROUPS

ANCHOR_SETS = {
    "full": ["arthre", "adla", "iadla", "gripsum", "grossa", "lgmusa", "finea", "hipe", "fall_s"],
    "objective_only": ["gripsum", "hipe", "fall_s"],
}


class Model:
    def __init__(self, d, countries, outcome, covariates):
        cols = [np.ones(len(d))]
        cols += [d.country.eq(c) for c in countries[1:]]           # first country is the reference
        cols += [d.age_group.eq(a) for a in AGE_GROUPS[1:]]
        cols += [d[c] for c in covariates]
        self.X = np.column_stack(cols).astype(float)
        self.y = d[outcome].to_numpy(float)
        self.w = d.w.to_numpy()
        self.rows = d.index.to_numpy()                             # rows of the bootstrap multiplier
        self.country = pd.Categorical(d.country, countries).codes
        self.age = pd.Categorical(d.age_group, AGE_GROUPS).codes
        self.k = len(countries)
        self.start = None

    def standardised_rates(self, mult):
        w = self.w * mult[self.rows]
        keep = w > 0
        b = sm.GLM(self.y[keep], self.X[keep], family=sm.families.Binomial(), var_weights=w[keep]).fit(
            start_params=self.start, maxiter=100, tol=1e-9, tol_criterion="params").params
        if self.start is None:                                     # replicates start from the point fit
            self.start = b
        effects = np.r_[0.0, b[1:self.k]]
        eta = self.X @ b - effects[self.country]
        rates = np.empty((self.k, len(AGE_GROUPS)))
        for a in range(len(AGE_GROUPS)):
            s = (self.age == a) & keep
            rates[:, a] = np.average(expit(eta[s, None] + effects[None, :]), axis=0, weights=w[s])
        return effects, rates


def models(d, countries, anchors):
    """d: waves 8-9 records with country, age_group, w, ms, un, female, severe and the
    indicators; complete cases for the indicator set are used."""
    cc = d.dropna(subset=anchors)
    pain = cc[cc.ms.astype(bool)]
    return [Model(cc, countries, "ms", ["female"]),
            Model(cc, countries, "ms", ["female"] + anchors),
            Model(pain, countries, "un", ["female", "severe"]),
            Model(pain, countries, "un", ["female", "severe"] + anchors)]


def estimate(fitted, mult, primary, L):
    """Country effects (logit), standardised years and summary statistics.
    primary: the pooled respondent-wave estimate on the full sample, for comparison."""
    (ep0, p0), (ep1, p1), (eu0, u0), (eu1, u1) = (m.standardised_rates(mult) for m in fitted)
    y0, y1 = (L * p0 * u0).sum(1), (L * p1 * u1).sum(1)
    by_country = {"P_effect": ep1, "U_effect": eu1, "years_standardised": y1,
                  "P_effect_unadjusted": ep0, "U_effect_unadjusted": eu0,
                  "years_unadjusted": y0, "years_primary_full_sample": primary}
    stats = {
        "spearman_P": spearmanr(ep0, ep1).statistic,
        "spearman_U": spearmanr(eu0, eu1).statistic,
        "spearman_years_same_sample": spearmanr(y0, y1).statistic,
        "spearman_primary_vs_same_sample": spearmanr(primary, y0).statistic,
        "spearman_primary_vs_standardised": spearmanr(primary, y1).statistic,
        "range_standardised": np.ptp(y1),
        "range_unadjusted": np.ptp(y0),
    }
    for name, before, after in [("P", ep0, ep1), ("U", eu0, eu1)]:
        for label, f in [("SD", lambda x: np.std(x, ddof=1)), ("MAD", lambda x: np.mean(np.abs(x - x.mean())))]:
            stats[f"{name}_{label}_unadjusted"] = f(before)
            stats[f"{name}_{label}_adjusted"] = f(after)
            stats[f"{name}_{label}_shrinkage_pct"] = 100 * (1 - f(after) / f(before))
    return by_country, stats
