"""Number of people aged 65+ in the primary state in 2022 and 2050 under fixed rates, and
static standardisation scenarios.

Fig. 5, Additional file 1: Table S9 and Supplementary methods S8.

count = sum over age groups of pop_g * p_g * u_g, with the Eurostat baseline projection
(proj_23np). The rates are held at their observed values; a scenario replaces one or both
by min(own value, benchmark), so no country's rate is raised. Benchmarks, by age group:
the lowest observed u, its 25th percentile and its median across countries, and the median
of p. Medians are taken with countries weighted equally or by their 2022 population in the
age group; the lowest value and the 25th percentile always weight countries equally.

Intervals use the replicates of the primary run (seed 20260922), with the benchmarks
recomputed in each replicate; the projection is held fixed. No interval is given for the
lowest-value and 25th-percentile benchmarks, whose percentile intervals are biased
(Supplementary methods S3).

The 85+ sensitivity analysis borrows the rise of the primary state within the open age
group (85-89, 90-94, 95+), pooled over all countries, and reweights it to each country's
projected age structure within 85+ in 2050.
"""

import numpy as np
import pandas as pd

from bootstrap import interval
from prepare import PUBLIC, geo_codes, options, population, primary_records, results_dir
from sullivan import primary_bootstrap

SCENARIOS = ["baseline", "u_lowest", "u_p25", "u_median", "p_median", "both_median"]
NO_INTERVAL = ["u_lowest", "u_p25"]


def weighted_median(x, w):
    """Smallest value whose cumulative weight reaches half the total."""
    order = np.argsort(x, kind="stable")
    cum = np.cumsum(w[order]) / w.sum()
    return x[order][np.searchsorted(cum, 0.5)]


def median(x, pop, convention):
    if convention == "country_equal":
        return np.median(x, axis=0)
    return np.array([weighted_median(x[:, g], pop[:, g]) for g in range(x.shape[1])])


def scenario_rates(p, u, pop22, convention):
    u_med, p_med = median(u, pop22, convention), median(p, pop22, convention)
    return {"baseline": (p, u),
            "u_lowest": (p, np.minimum(u, u.min(axis=0))),
            "u_p25": (p, np.minimum(u, np.percentile(u, 25, axis=0))),
            "u_median": (p, np.minimum(u, u_med)),
            "p_median": (np.minimum(p, p_med), u),
            "both_median": (np.minimum(p, p_med), np.minimum(u, u_med))}


def project(p, u, pop22, pop50, convention):
    """Counts by country in 2022 and 2050, and the 2050 reduction from the baseline."""
    base50 = (pop50 * p * u).sum(1)
    out = {}
    for s, (ps, us) in scenario_rates(p, u, pop22, convention).items():
        n50 = (pop50 * ps * us).sum(1)
        out[s] = {"n2022": (pop22 * ps * us).sum(1), "n2050": n50, "reduction": base50 - n50}
    return out


def oldest_old_shift(d, countries):
    """Relative change in each country's 85+ prevalence of the primary state when the pooled
    within-85+ age pattern is reweighted from the 2022 to the 2050 age structure."""
    old = d[d.agey.ge(85)]
    band = pd.cut(old.agey, [85, 90, 95, np.inf], right=False, labels=["85-89", "90-94", "95+"])
    prev = old.w.mul(old.un).groupby(band, observed=True).sum() / old.w.groupby(band, observed=True).sum()
    pop = pd.read_csv(PUBLIC / "population_85plus_2022_2050.csv").set_index(["geo", "age_band"])
    shift = []
    for geo in geo_codes(countries):
        share = pop.loc[geo]
        share = share / share.sum()
        shift.append((share.pop_2050 * prev).sum() / (share.pop_2022 * prev).sum() - 1)
    print("85+ prevalence of the primary state by band:", prev.round(3).to_dict())
    return np.array(shift)


def main():
    args = options()
    out = results_dir()
    d = primary_records()
    countries = sorted(d.country.unique())
    pop22, pop50 = population(countries, 2022), population(countries, 2050)

    point, reps = primary_bootstrap(d, countries, args.reps)
    p, q = point["wave_combined"]
    reps = [r["wave_combined"] for r in reps]

    rows, totals = [], []
    for convention in ["country_equal", "population_weighted"]:
        est = project(p, q / p, pop22, pop50, convention)
        boot = [project(a, b / a, pop22, pop50, convention) for a, b in reps]
        for s in SCENARIOS:
            rows.append(pd.DataFrame({"convention": convention, "scenario": s, "country": countries,
                                      "n2022": est[s]["n2022"], "n2050": est[s]["n2050"],
                                      "reduction_2050": est[s]["reduction"]}))
            n50 = np.array([b[s]["n2050"].sum() for b in boot])
            red = np.array([b[s]["reduction"].sum() for b in boot])
            row = {"convention": convention, "scenario": s, "n2022": est[s]["n2022"].sum(),
                   "n2050": est[s]["n2050"].sum(), "n2050_ci_low": np.percentile(n50, 2.5),
                   "n2050_ci_high": np.percentile(n50, 97.5), "reduction_2050": est[s]["reduction"].sum(),
                   "reduction_ci_low": np.percentile(red, 2.5), "reduction_ci_high": np.percentile(red, 97.5)}
            if s in NO_INTERVAL:
                row.update(n2050_ci_low=np.nan, n2050_ci_high=np.nan, reduction_ci_low=np.nan, reduction_ci_high=np.nan)
            totals.append(row)
        # the medication-use scenario against the pain scenario, both at the median
        diff = np.array([b["u_median"]["reduction"].sum() - b["p_median"]["reduction"].sum() for b in boot])
        totals.append({"convention": convention, "scenario": "u_median minus p_median",
                       "reduction_2050": est["u_median"]["reduction"].sum() - est["p_median"]["reduction"].sum(),
                       "reduction_ci_low": np.percentile(diff, 2.5), "reduction_ci_high": np.percentile(diff, 97.5)})
    pd.concat(rows).to_csv(out / "tableS9_projection_by_country.csv", index=False)
    pd.DataFrame(totals).to_csv(out / "tableS9_projection_totals.csv", index=False)

    # 85+ within-group ageing (Supplementary methods S8), point estimates
    shift = oldest_old_shift(d, countries)
    p_old = p.copy()
    p_old[:, -1] *= 1 + shift
    rows = []
    for convention in ["country_equal", "population_weighted"]:
        before = project(p, q / p, pop22, pop50, convention)
        after = project(p_old, q / p, pop22, pop50, convention)
        for s in SCENARIOS:
            rows.append({"convention": convention, "scenario": s,
                         "n2050": before[s]["n2050"].sum(), "n2050_85plus_reweighted": after[s]["n2050"].sum(),
                         "reduction_2050": before[s]["reduction"].sum(),
                         "reduction_2050_85plus_reweighted": after[s]["reduction"].sum()})
    pd.DataFrame(rows).to_csv(out / "projection_85plus_sensitivity.csv", index=False)
    pd.DataFrame({"country": countries, "geo": geo_codes(countries), "relative_change_85plus_pct": 100 * shift}
                 ).to_csv(out / "projection_85plus_shift.csv", index=False)
    print(f"85+ shift: median {100 * np.median(shift):.2f}%, largest {100 * shift.max():.2f}%")


if __name__ == "__main__":
    main()
