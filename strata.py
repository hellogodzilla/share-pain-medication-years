"""By sex and education (Tables S10, S11), as domains of the full sample.

Women vs men on three life tables: two-sex EU27, EU27 by sex, own country by sex.
Education uses the two-sex EU27 table. Replays the draws of the primary run.
"""

import warnings

import numpy as np
import pandas as pd

from bootstrap import multipliers
from prepare import AGE_GROUPS, MIN_CELL, options, primary_records, results_dir
from sullivan import PRIMARY_SEED, cell_codes, life_table, totals

SEX = {"women": 2, "men": 1}                                  # ragender
EDUCATION = {"low": 1, "middle": 2, "high": 3}                # raeducl, ISCED-based


def domain_years(tot, took_part, L):
    # NaN if the domain has an empty age cell in a wave the country took part in
    den, _, un = tot
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(took_part[:, :, None], un / den, np.nan)
    ok = ((den > 0) | ~took_part[:, :, None]).all((1, 2))
    yrs = (L * np.nanmean(q, axis=1)).sum(1)
    return np.where(ok, yrs, np.nan)


def quantities(d, cells, mult, took_part, life):
    def yrs(col, code, L):
        return domain_years(totals(d, cells, mult * d[col].eq(code).to_numpy()), took_part, L)

    out = {}
    for sex in SEX:
        out[f"{sex}, two-sex EU27 table"] = yrs("ragender", SEX[sex], life["EU27 T"])
        out[f"{sex}, EU27 table of own sex"] = yrs("ragender", SEX[sex], life[f"EU27 {sex}"])
        out[f"{sex}, own country table of own sex"] = yrs("ragender", SEX[sex], life[f"country {sex}"])
    for kind in ["two-sex EU27 table", "EU27 table of own sex", "own country table of own sex"]:
        out[f"women minus men, {kind}"] = out[f"women, {kind}"] - out[f"men, {kind}"]
    for level, code in EDUCATION.items():
        out[f"education {level}"] = yrs("raeducl", code, life["EU27 T"])
    out["education low minus high"] = out["education low"] - out["education high"]
    return out


def shown_for_education(d, countries):
    n = d.groupby(["country", "raeducl", "age_group"]).size().unstack(fill_value=0)
    n = n.reindex(columns=AGE_GROUPS, fill_value=0)
    full = (n >= MIN_CELL).all(axis=1)
    return np.array([full.get((c, 1), False) and full.get((c, 3), False) for c in countries])


def main():
    warnings.simplefilter("ignore", RuntimeWarning)      # empty education cells give all-NaN slices
    args = options()
    out = results_dir()
    d = primary_records()
    countries = sorted(d.country.unique())
    cells = cell_codes(d, countries)
    took_part = totals(d, cells)[0].sum(2) > 0                # country x wave
    life = {"EU27 T": life_table(),
            "EU27 women": life_table(sex="F"), "EU27 men": life_table(sex="M"),
            "country women": np.array([life_table(c, "F") for c in countries]),
            "country men": np.array([life_table(c, "M") for c in countries])}

    point = quantities(d, cells, 1.0, took_part, life)
    reps = [quantities(d, cells, m, took_part, life) for m in multipliers(d, args.reps, PRIMARY_SEED)]

    rows = []
    for name, est in point.items():
        x = np.array([r[name] for r in reps])
        rows.append(pd.DataFrame({"country": countries, "quantity": name, "estimate": est,
                                  "ci_low": np.nanquantile(x, 0.025, axis=0),
                                  "ci_high": np.nanquantile(x, 0.975, axis=0),
                                  "valid_replicates": np.isfinite(x).sum(0)}))
    res = pd.concat(rows, ignore_index=True)
    sex = res[~res.quantity.str.startswith("education")]
    edu = res[res.quantity.str.startswith("education")].copy()
    edu["shown_in_table_S11"] = edu.country.map(dict(zip(countries, shown_for_education(d, countries))))
    sex.to_csv(out / "tableS10_sex.csv", index=False)
    edu.to_csv(out / "tableS11_education.csv", index=False)
    print(f"education: {edu.groupby('country').shown_in_table_S11.first().sum()} countries shown")


if __name__ == "__main__":
    main()
