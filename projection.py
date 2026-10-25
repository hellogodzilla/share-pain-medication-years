"""Counts in 2022 and 2050 at fixed rates, and the 2050 scenarios (Fig. 5, Table S9, S8).

count = sum_g pop_g * p_g * u_g (Eurostat proj_23np). A scenario sets a rate to
min(own, benchmark). Benchmarks by age group: lowest u, 25th percentile of u, median u,
median p. Medians: countries equal or weighted by 2022 population; lowest and p25 are
always country-equal and get no interval (methods S3). Benchmarks are recomputed in
every replicate of the primary run.
"""

import numpy as np
import pandas as pd

from bootstrap import interval
from prepare import PUBLIC, geo_codes, options, population, primary_records, results_dir
from sullivan import primary_bootstrap

SCENARIOS = ["baseline", "u_lowest", "u_p25", "u_median", "p_median", "both_median"]
NO_INTERVAL = ["u_lowest", "u_p25"]


def weighted_median(x, w):
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
    base50 = (pop50 * p * u).sum(1)
    out = {}
    for s, (ps, us) in scenario_rates(p, u, pop22, convention).items():
        n50 = (pop50 * ps * us).sum(1)
        out[s] = {"n2022": (pop22 * ps * us).sum(1), "n2050": n50, "reduction": base50 - n50}
    return out


def oldest_old_shift(d, countries):
    # pooled prevalence in 85-89 / 90-94 / 95+, reweighted from the 2022 to the 2050
    # age structure of each country
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
