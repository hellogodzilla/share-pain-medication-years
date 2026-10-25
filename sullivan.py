"""Sullivan estimates: Table 2, Fig. 1, Tables S4 and S14.

years = sum_g L_g * p_g * u_g, with L_g = nL_g / l65 (EU27 2019), p_g the prevalence of
moderate-to-severe pain and u_g the share of them with no weekly pain medication.
"""

import numpy as np
import pandas as pd

from bootstrap import interval, multipliers, rank_interval, ranks, spearman
from decomposition import das_gupta
from prepare import AGE_GROUPS, PUBLIC, options, primary_records, results_dir

PRIMARY_SEED = 20260922
POOLED_SEED = 20260920


def life_table(country="EU27", sex="T", year=2019):
    lt = pd.read_csv(PUBLIC / "life_tables.csv")
    lt = lt[lt.country.eq(country) & lt.sex.eq(sex) & lt.year.eq(year)]
    lt = lt.set_index("age_group").loc[AGE_GROUPS]
    return (lt.nL / lt.l65).to_numpy()


def cell_codes(d, countries, waves=(8, 9)):
    shape = (len(countries), len(waves), len(AGE_GROUPS))
    code = np.ravel_multi_index((pd.Categorical(d.country, countries).codes,
                                 pd.Categorical(d.wave, list(waves)).codes,
                                 pd.Categorical(d.age_group, AGE_GROUPS).codes), shape)
    return code, shape


def totals(d, cells, mult=1.0):
    # weighted den / ms / un by country x wave x age group; mult = bootstrap multipliers
    code, shape = cells
    w = d.w.to_numpy() * mult
    n = np.prod(shape)
    return [np.bincount(code, w * x, n).reshape(shape)
            for x in (1.0, d.ms.to_numpy(float), d.un.to_numpy(float))]


def wave_combined(den, ms, un):
    # primary estimator: prevalence within each wave, then the mean over the waves
    # (NaN for Portugal's wave 8 is skipped). Returns p and q = p * u.
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(ms / den, axis=1), np.nanmean(un / den, axis=1)


def pooled(den, ms, un):
    den, ms, un = den.sum(1), ms.sum(1), un.sum(1)
    return ms / den, un / den


def years(p, q, L):
    return {"primary": q @ L, "complementary": (p - q) @ L, "any_pain": p @ L}


def most_recent(d):
    last = d.sort_values("wave", kind="stable").drop_duplicates("mergeid", keep="last").index
    return d.index.isin(last)


def unit_totals(d, cells, recent, mult=1.0):
    both = totals(d, cells, mult)
    last = [t.sum(1) for t in totals(d, cells, mult * recent)]
    return both, last


def empty_cell(tot, present):
    # a replicate with an empty cell is dropped for all units
    (den, ms, _), (rden, rms, _) = tot
    return ((den[present] <= 0).any() or (ms[present] <= 0).any()
            or (rden <= 0).any() or (rms <= 0).any())


def unit_rates(tot):
    (den, ms, un), (rden, rms, run) = tot
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"wave_combined": wave_combined(den, ms, un),
                "wave8": (ms[:, 0] / den[:, 0], un[:, 0] / den[:, 0]),
                "wave9": (ms[:, 1] / den[:, 1], un[:, 1] / den[:, 1]),
                "recent": (rms / rden, run / rden),
                "pooled": pooled(den, ms, un)}


def point_rates(d, countries):
    return unit_rates(unit_totals(d, cell_codes(d, countries), most_recent(d)))


def primary_bootstrap(d, countries, n_reps, seed=PRIMARY_SEED):
    cells = cell_codes(d, countries)
    recent = most_recent(d)
    tot = unit_totals(d, cells, recent)
    present = tot[0][0] > 0                # Portugal has no wave-8 cells
    point = unit_rates(tot)
    reps = []
    for m in multipliers(d, n_reps, seed):
        tot = unit_totals(d, cells, recent, m)
        if not empty_cell(tot, present):
            reps.append(unit_rates(tot))
    print(f"primary run: {len(reps)} of {n_reps} replicates valid")
    return point, reps


def pooled_bootstrap(d, countries, n_reps, seed=POOLED_SEED):
    cells = cell_codes(d, countries)
    point = pooled(*totals(d, cells))
    reps = []
    for m in multipliers(d, n_reps, seed):
        den, ms, un = (t.sum(1) for t in totals(d, cells, m))
        if (den > 0).all() and (ms > 0).all():
            reps.append((ms / den, un / den))
    print(f"pooled run: {len(reps)} of {n_reps} replicates valid")
    return point, reps


def estimates(p, q, L):
    out = years(p, q, L)
    _, out["P"], out["U"] = das_gupta(p, q / p, L)
    return out


def summarise(countries, point, reps):
    rows = []
    for name, est in point.items():
        x = np.array([r[name] for r in reps])
        lo, hi = interval(x)
        rlo, rhi = rank_interval(x)
        rows.append(pd.DataFrame({"country": countries, "quantity": name, "estimate": est,
                                  "ci_low": lo, "ci_high": hi, "rank": ranks(est),
                                  "rank_low": rlo, "rank_high": rhi}))
    return pd.concat(rows, ignore_index=True)


def unit_table(countries, point, reps, L):
    rows = []
    for unit in ["wave_combined", "wave8", "wave9", "recent"]:
        p, q = point[unit]
        keep = np.isfinite(p[:, 0])                       # Portugal is absent from wave 8
        est = years(p[keep], q[keep], L)
        boot = [years(r[unit][0][keep], r[unit][1][keep], L) for r in reps]
        rows.append(summarise(list(np.array(countries)[keep]), est, boot).assign(unit=unit))
    return pd.concat(rows, ignore_index=True)


def unit_summary(table):
    t = table[table.quantity.eq("primary")]
    main = t[t.unit.eq("wave_combined")].set_index("country").estimate
    rows = []
    for unit, g in t.groupby("unit", sort=False):
        g = g.set_index("country")
        top, low = g.estimate.idxmax(), g.estimate.idxmin()
        rows.append({"unit": unit, "countries": len(g),
                     "highest": top, "highest_years": g.estimate[top],
                     "highest_ci": f"{g.ci_low[top]:.2f} to {g.ci_high[top]:.2f}",
                     "lowest": low, "lowest_years": g.estimate[low],
                     "lowest_ci": f"{g.ci_low[low]:.2f} to {g.ci_high[low]:.2f}",
                     "range": g.estimate.max() - g.estimate.min(),
                     "spearman_with_primary": spearman(g.estimate, main[g.index]),
                     "median_abs_difference": (g.estimate - main[g.index]).abs().median()})
    return pd.DataFrame(rows)


def main():
    args = options()
    out = results_dir()
    d = primary_records()
    countries = sorted(d.country.unique())
    L = life_table()
    print(f"{len(countries)} countries, {len(d)} respondent-wave records, "
          f"life expectancy at 65 {L.sum():.3f} years")

    point, reps = primary_bootstrap(d, countries, args.reps)
    p, q = point["wave_combined"]
    est = estimates(p, q, L)
    boot = [estimates(*r["wave_combined"], L) for r in reps]
    table2 = summarise(countries, est, boot)

    own = np.array([life_table(c) @ q[i] for i, c in enumerate(countries)])
    extra = pd.DataFrame({"country": countries, "share_of_le65_pct": 100 * est["primary"] / L.sum(),
                          "primary_own_life_table": own, "rank_own_life_table": ranks(own),
                          "le65_own_life_table": [life_table(c).sum() for c in countries]})
    table2.to_csv(out / "table2_main_estimates.csv", index=False)
    extra.to_csv(out / "table2_own_life_table.csv", index=False)

    # agreement between the states, with intervals from the same replicates
    x = np.array([b["primary"] for b in boot])
    corr = []
    for other in ["complementary", "any_pain"]:
        y = np.array([b[other] for b in boot])
        lo, hi = interval(spearman(x, y))
        corr.append({"pair": f"primary vs {other}", "spearman": spearman(est["primary"], est[other]),
                     "ci_low": lo, "ci_high": hi})
    pd.DataFrame(corr).to_csv(out / "rank_correlations.csv", index=False)

    # Table S14: the four analysis units, and the pooled estimator from its own run
    units = unit_table(countries, point, reps, L)
    (pp, pq), preps = pooled_bootstrap(d, countries, args.reps)
    pooled_rows = summarise(countries, years(pp, pq, L), [years(*r, L) for r in preps]).assign(unit="pooled")
    units = pd.concat([units, pooled_rows], ignore_index=True)
    units.to_csv(out / "tableS14_analysis_units.csv", index=False)
    unit_summary(units).to_csv(out / "tableS14_summary.csv", index=False)

    # paired differences between units, from the primary run
    diffs = []
    for a, b in [("wave_combined", "pooled"), ("recent", "wave_combined")]:
        dx = years(*point[a], L)["primary"] - years(*point[b], L)["primary"]
        bx = np.array([years(*r[a], L)["primary"] - years(*r[b], L)["primary"] for r in reps])
        lo, hi = interval(bx)
        diffs.append(pd.DataFrame({"difference": f"{a} minus {b}", "country": countries,
                                   "estimate": dx, "ci_low": lo, "ci_high": hi}))
    pd.concat(diffs).to_csv(out / "tableS14_unit_differences.csv", index=False)

    print(f"country mean {est['primary'].mean():.3f} years; range {np.ptp(est['primary']):.3f}")


if __name__ == "__main__":
    main()
