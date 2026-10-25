"""Change in years lived from age 65 in the primary state between two waves.

Fig. 4 and Additional file 1: Table S8.

Windows: 2013 (wave 5) to 2021-22 (wave 9), and the two shorter windows 2015 (wave 6) to
2019-20 (wave 8) and wave 8 to wave 9. Each uses the countries of the primary pool with at
least 30 respondents in every age group in both waves. Each wave is estimated with its own
weights, and the change is split into a pain (P) and a medication-use (U) component by the
two-point symmetric Das Gupta formula, so that P + U equals the change.

Intervals: Rao-Wu bootstrap over the PSUs of both waves, so that a panel member carries the
same multiplier at both ends; the seed (20260923) is reset for each window. A country is
left out of a replicate in which one of its age cells is empty.
"""

import numpy as np
import pandas as pd

from bootstrap import multipliers
from prepare import AGE_GROUPS, MIN_CELL, add_design, load_records, options, primary_sample, results_dir
from sullivan import cell_codes, life_table, totals

SEED = 20260923
WINDOWS = {"2013 to 2021-22": (5, 9), "2015 to 2019-20": (6, 8), "2019-20 to 2021-22": (8, 9)}
YEAR = {5: 2013, 6: 2015, 8: 2019.5, 9: 2021.5}


def change(den, ms, un, L):
    """Years at the start and end wave (country x 2) and the P and U components of the change."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p, u = ms / den, un / ms
    yrs = (L * p * u).sum(2)
    P = (L * (u[:, 1] + u[:, 0]) / 2 * (p[:, 1] - p[:, 0])).sum(1)
    U = (L * (p[:, 1] + p[:, 0]) / 2 * (u[:, 1] - u[:, 0])).sum(1)
    ok = (den > 0).all((1, 2)) & (ms > 0).all((1, 2))
    return yrs, np.where(ok, P, np.nan), np.where(ok, U, np.nan)


def window_countries(d, waves, pool):
    n = d[d.wave.isin(waves)].groupby(["country", "wave", "age_group"]).size()
    n = n.unstack(["wave", "age_group"]).reindex(columns=pd.MultiIndex.from_product([waves, AGE_GROUPS]))
    return sorted(set(n.index[(n.fillna(0) >= MIN_CELL).all(axis=1)]) & pool)


def main():
    args = options()
    out = results_dir()
    L = life_table()
    pool = set(primary_sample(load_records([8, 9])).country)
    d = add_design(load_records([5, 6, 8, 9]))

    tables = []
    for name, waves in WINDOWS.items():
        countries = window_countries(d, waves, pool)
        w = d[d.wave.isin(waves) & d.country.isin(countries)]
        cells = cell_codes(w, countries, waves)
        yrs, P, U = change(*totals(w, cells), L)

        boot = np.array([change(*totals(w, cells, m), L)[1:] for m in multipliers(w, args.reps, SEED)])
        P_rep, U_rep = boot[:, 0], boot[:, 1]

        res = pd.DataFrame({"window": name, "country": countries,
                            "years_start": yrs[:, 0], "years_end": yrs[:, 1],
                            "change": yrs[:, 1] - yrs[:, 0],
                            "change_per_decade": (yrs[:, 1] - yrs[:, 0]) * 10 / (YEAR[waves[1]] - YEAR[waves[0]]),
                            "P_component": P, "U_component": U})
        for label, x in [("change", P_rep + U_rep), ("P_component", P_rep), ("U_component", U_rep)]:
            res[f"{label}_ci_low"] = np.nanquantile(x, 0.025, axis=0)
            res[f"{label}_ci_high"] = np.nanquantile(x, 0.975, axis=0)
        tables.append(res)
        print(f"{name}: {len(countries)} countries")

    pd.concat(tables).to_csv(out / "tableS8_trend.csv", index=False)


if __name__ == "__main__":
    main()
