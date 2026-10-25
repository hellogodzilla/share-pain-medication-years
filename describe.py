"""Table 1A, the first rows of Table 1B, and the pain-item distribution (S1).

1A: pooled respondent weights. 1B and S1: within-wave weights, waves averaged.
"""

import numpy as np
import pandas as pd

from prepare import options, primary_records, results_dir


def by_country(d):
    def summary(g):
        w = g.w
        return pd.Series({
            "observations": len(g),
            "households": g.hh.nunique(),
            "mean_age": np.average(g.agey, weights=w),
            "women_pct": 100 * np.average(g.ragender.eq(2), weights=w),
            "aged_85_plus_pct": 100 * np.average(g.agey.ge(85), weights=w),
            "moderate_severe_pain_pct": 100 * np.average(g.ms, weights=w),
            "no_medication_among_pain_pct": 100 * np.average(g.un[g.ms], weights=w[g.ms]),
            "primary_state_pct": 100 * np.average(g.un, weights=w),
        })
    t = d.groupby("country").apply(summary, include_groups=False)
    return t.astype({"observations": int, "households": int})


def wave_averaged(d, value):
    return np.mean([np.average(value[g.index], weights=g.w) for _, g in d.groupby("wave")])


def by_state(d):
    state = np.select([d.un, d.ms], ["primary state", "complementary state"], "no moderate-to-severe pain")
    rows = {}
    for name in ["primary state", "complementary state", "no moderate-to-severe pain"]:
        g = d[state == name]
        rows[name] = {
            "observations": len(g),
            "mean_age": wave_averaged(g, g.agey),
            "aged_85_plus_pct": 100 * wave_averaged(g, g.agey.ge(85)),
            "women_pct": 100 * wave_averaged(g, g.ragender.eq(2)),
            "education_low_pct": 100 * wave_averaged(g, g.raeducl.eq(1)),
            "education_middle_pct": 100 * wave_averaged(g, g.raeducl.eq(2)),
            "education_high_pct": 100 * wave_averaged(g, g.raeducl.eq(3)),
            "severe_pain_pct": 100 * wave_averaged(g, g.painlv.eq(3)),
        }
    return pd.DataFrame(rows)


def response_distribution(d):
    rows = []
    for level, label in enumerate(["none", "mild", "moderate", "severe"]):
        at = d.painlv.eq(level)
        g = d[at]
        rows.append({"pain_intensity": label, "observations": int(at.sum()),
                     "share_pct": 100 * wave_averaged(d, at),
                     "weekly_pain_medication_pct": 100 * wave_averaged(g, g.painmed.eq(1))})
    return pd.DataFrame(rows)


def main():
    options()
    out = results_dir()
    d = primary_records()
    by_country(d).to_csv(out / "table1A_by_country.csv")
    by_state(d).to_csv(out / "table1B_by_state.csv")
    response_distribution(d).to_csv(out / "S1_response_distribution.csv", index=False)
    print(f"{len(d)} observations, {d.mergeid.nunique()} people, {d.hh.nunique()} households, "
          f"{d.country.nunique()} countries")


if __name__ == "__main__":
    main()
