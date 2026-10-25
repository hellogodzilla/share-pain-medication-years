"""Wave 8 -> wave 9 transitions out of state A (Table S18, panel B; wave-8 counts of A).

A: moderate-to-severe pain, no weekly pain medication; B: pain with medication;
C: medication, no moderate-to-severe pain; D: neither. Wave-8 weights, 25 countries.
Deaths and non-response (panel A) are not in these files.
"""

import numpy as np
import pandas as pd

from bootstrap import interval, multipliers
from prepare import options, primary_records, results_dir

SEED = 20260922
STATES = ["A", "B", "C", "D"]


def state(d):
    return pd.Series(np.select([d.ms & d.painmed.eq(0), d.ms, d.painmed.eq(1)], ["A", "B", "C"], "D"), d.index)


def shares(start, end, w):
    tot = pd.Series(w, start.index)[start.eq("A") & end.notna()].groupby(end).sum()
    tot = tot.reindex(STATES, fill_value=0.0)
    out = 100 * tot / tot.sum()
    out["B among B or C"] = 100 * tot["B"] / (tot["B"] + tot["C"])
    return out


def main():
    args = options()
    out = results_dir()
    d = primary_records()
    d = d[d.country.isin(d.loc[d.wave.eq(8), "country"].unique())].reset_index(drop=True)
    d["state"] = state(d)

    w8 = d.wave.eq(8)
    start = d.state.where(w8)
    end = d.mergeid.map(d.loc[d.wave.eq(9)].set_index("mergeid").state).where(w8)
    point = shares(start, end, d.w.to_numpy())

    reps = pd.DataFrame([shares(start, end, d.w.to_numpy() * m) for m in multipliers(d, args.reps, SEED)])
    lo, hi = interval(reps.to_numpy())
    res = pd.DataFrame({"wave9_state": point.index, "weighted_pct": point.values, "ci_low": lo, "ci_high": hi})
    res.to_csv(out / "tableS18_transitions_from_A.csv", index=False)

    counts = start.value_counts().reindex(STATES).rename("n_wave8")
    counts = pd.concat([counts, start[end.notna()].value_counts().reindex(STATES).rename("n_reinterviewed_valid_state")], axis=1)
    counts.to_csv(out / "tableS18_wave8_counts.csv", index_label="wave8_state")
    print(f"{d.country.nunique()} countries; {int(counts.loc['A', 'n_reinterviewed_valid_state'])} in A followed to wave 9")


if __name__ == "__main__":
    main()
