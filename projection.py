"""Fixed-rate standardisation scenarios for 2022 and 2050 (Fig 5; Additional file 1:
Table S9).

Counts are sum over age groups of pop * p * u, with the Eurostat baseline population
projection (proj_23np) for 2022 and 2050. Each scenario replaces one or both age-specific
rates by min(own value, benchmark). Intervals: the benchmarks are recomputed in every
replicate of the primary bootstrap (seed 20260922).
"""

import numpy as np

SCENARIOS = ["baseline", "U_to_best", "U_to_p25", "U_to_median", "P_to_median", "U_and_P_to_median"]


def weighted_median(x, w):
    """Smallest value whose cumulative weight reaches half the total."""
    order = np.argsort(x, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    return x[order][np.searchsorted(cum, 0.5)]


def median(x, pop, convention):
    if convention == "country_equal":
        return np.median(x, axis=0)
    # population_weighted: each country weighted by its 2022 population in the age group
    return np.array([weighted_median(x[:, g], pop[:, g]) for g in range(x.shape[1])])


def project(p, u, pop22, pop50, convention="country_equal"):
    """p, u, pop22, pop50: country x age group. Returns counts in 2022 and 2050 and the
    2050 count avoided relative to the baseline, by scenario and country."""
    u_med, p_med = median(u, pop22, convention), median(p, pop22, convention)
    rates = {
        "baseline": (p, u),
        "U_to_best": (p, np.minimum(u, u.min(axis=0))),               # lowest observed
        "U_to_p25": (p, np.minimum(u, np.percentile(u, 25, axis=0))),
        "U_to_median": (p, np.minimum(u, u_med)),
        "P_to_median": (np.minimum(p, p_med), u),
        "U_and_P_to_median": (np.minimum(p, p_med), np.minimum(u, u_med)),
    }
    base50 = (pop50 * p * u).sum(1)
    out = {}
    for s, (ps, us) in rates.items():
        n22, n50 = (pop22 * ps * us).sum(1), (pop50 * ps * us).sum(1)
        out[s] = {"n2022": n22, "n2050": n50, "avoided_2050": base50 - n50}
    return out
