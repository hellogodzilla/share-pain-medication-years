"""Point-estimate sensitivity analyses: Tables S5, S13, S17.

S13: severe pain only. S5: pool composition and the 2019-2021 mean life table (pooled
estimator). S17: specifications computable from these files.
"""

import numpy as np
import pandas as pd

from bootstrap import spearman
from decomposition import das_gupta, profile
from prepare import AGE_GROUPS, MIN_CELL, load_records, options, primary_records, results_dir
from sullivan import cell_codes, life_table, point_rates, pooled, totals, wave_combined


def severe_only(d, countries, L):
    s = d.assign(ms=d.painlv.eq(3), un=d.painlv.eq(3) & d.painmed.eq(0))
    _, q = wave_combined(*totals(s, cell_codes(s, countries)))
    n = s[s.ms].groupby(["country", "age_group"]).size().unstack(fill_value=0)
    n = n.reindex(index=countries, columns=AGE_GROUPS, fill_value=0)
    return q @ L, (n < MIN_CELL).sum(1).to_numpy()


def compare(name, estimator, est, base):
    common = est.index.intersection(base.index)
    change = est[common] - base[common]
    big = change.abs().idxmax()
    return {"specification": name, "estimator": estimator, "countries": len(est),
            "range": est.max() - est.min(), "highest": est.idxmax(), "lowest": est.idxmin(),
            "spearman_with_reference": spearman(est[common], base[common]),
            "median_abs_change": change.abs().median(),
            "largest_change": change[big], "largest_change_country": big}


def pool_scenario(name, p, q, L, countries):
    yrs, P, U = das_gupta(p, q / p, L)
    labels = pd.Series(profile(yrs, P, U), countries)
    U = pd.Series(U, countries)
    yrs = pd.Series(yrs, countries)
    return {"scenario": name, "countries": len(countries), "range": yrs.max() - yrs.min(),
            "highest": yrs.idxmax(), "lowest": yrs.idxmin(), "country_mean": yrs.mean(),
            "U_Greece": U.get("Greece"), "U_Malta": U.get("Malta"),
            "profile_Greece": labels.get("Greece"), "profile_Malta": labels.get("Malta")}


def main():
    options()
    out = results_dir()
    d = primary_records()
    countries = sorted(d.country.unique())
    L = life_table()
    L_mean = np.mean([life_table(year=y) for y in (2019, 2020, 2021)], axis=0)

    rates = point_rates(d, countries)
    primary = pd.Series(rates["wave_combined"][1] @ L, countries)

    # S13
    severe, small = severe_only(d, countries, L)
    s13 = pd.DataFrame({"country": countries, "severe_pain_years": severe, "primary_years": primary.values,
                        "severe_cells_below_30": small}).sort_values("severe_pain_years")
    s13.to_csv(out / "tableS13_severe_pain.csv", index=False)
    print(f"severe pain only: Spearman with primary {spearman(severe, primary.values):.4f}; "
          f"{small.sum()} of {5 * len(small)} cells below 30; {(small == 0).sum()} countries complete")

    # S5, on the pooled estimator
    p, q = rates["pooled"]
    rows = [pool_scenario("26 countries", p, q, L, countries)]
    israel = load_records([8, 9]).query("country == 'Israel'")
    pi, qi = pooled(*totals(israel, cell_codes(israel, ["Israel"])))
    with_israel = sorted(countries + ["Israel"])
    order = np.argsort(countries + ["Israel"])
    rows.append(pool_scenario("add Israel", np.r_[p, pi][order], np.r_[q, qi][order], L, with_israel))
    for drop in [["Portugal"], ["Malta"], ["Malta", "Portugal"]]:
        keep = ~np.isin(countries, drop)
        rows.append(pool_scenario("exclude " + " and ".join(drop), p[keep], q[keep], L, list(np.array(countries)[keep])))
    rows.append(pool_scenario("mean 2019-2021 life table", p, q, L_mean, countries))
    pd.DataFrame(rows).to_csv(out / "tableS5_pool_and_life_table.csv", index=False)
    print(f"Israel, pooled estimator, EU27 table: {(qi @ L)[0]:.3f} years")

    # S17
    pooled_years = pd.Series(q @ L, countries)
    own = pd.Series([life_table(c) @ rates["wave_combined"][1][i] for i, c in enumerate(countries)], countries)
    rows = [compare("primary: common reference life table", "wave-combined", primary, primary),
            compare("each country's own life table", "wave-combined", own, primary),
            compare("mean 2019-2021 life table", "pooled", pd.Series(q @ L_mean, countries), pooled_years),
            compare("Portugal excluded", "pooled", pooled_years.drop("Portugal"), pooled_years),
            compare("Portugal and Malta excluded", "pooled", pooled_years.drop(["Portugal", "Malta"]), pooled_years)]
    for unit, label in [("wave8", "wave 8 only"), ("wave9", "wave 9 only"),
                        ("recent", "one most-recent record per person"), ("pooled", "pooled respondent-wave")]:
        est = pd.Series(rates[unit][1] @ L, countries).dropna()
        rows.append(compare(label, label, est, primary))
    pd.DataFrame(rows).to_csv(out / "tableS17_specifications.csv", index=False)


if __name__ == "__main__":
    main()
