"""G-computation case-mix standardisation (Table S7, methods S5).

Weighted logits on complete cases, with country, age group and sex: pain among everyone
(P model) and no weekly pain medication among those in pain (U model, + severe pain).
Each country's coefficient is applied to the whole pooled sample, predictions averaged by
age group, then the Sullivan sum. "Unadjusted" = same models without the indicators.
Spread of country effects: mean absolute deviation (does not depend on the reference).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import spearmanr

from bootstrap import interval, multipliers
from prepare import AGE_GROUPS, add_design, data_dir, eligible, options, primary_sample, results_dir
from sullivan import cell_codes, life_table, pooled, totals

SEED = 20260921
INDICATORS = {
    "nine": ["arthre", "adla", "iadla", "gripsum", "grossa", "lgmusa", "finea", "hipe", "fall_s"],
    "three": ["gripsum", "hipe", "fall_s"],      # grip strength, hip fracture, any fall
}


class Logit:

    def __init__(self, d, countries, outcome, covariates):
        cols = [np.ones(len(d))]
        cols += [d.country.eq(c) for c in countries[1:]]        # first country is the reference
        cols += [d.age_group.eq(a) for a in AGE_GROUPS[1:]]
        cols += [d[c] for c in covariates]
        self.X = np.column_stack(cols).astype(float)
        self.y = d[outcome].to_numpy(float)
        self.w = d.w.to_numpy()
        self.rows = d.index.to_numpy()                          # positions in the full sample
        self.country = pd.Categorical(d.country, countries).codes
        self.age = pd.Categorical(d.age_group, AGE_GROUPS).codes
        self.k = len(countries)
        self.start = None

    def standardised(self, mult):
        w = self.w * mult[self.rows]
        keep = w > 0
        fit = sm.GLM(self.y[keep], self.X[keep], family=sm.families.Binomial(), var_weights=w[keep])
        b = fit.fit(start_params=self.start, maxiter=100, tol=1e-9, tol_criterion="params").params
        if self.start is None:
            self.start = b
        effect = np.r_[0.0, b[1:self.k]]
        eta = self.X @ b - effect[self.country]                 # linear predictor without country
        rates = np.empty((self.k, len(AGE_GROUPS)))
        for a in range(len(AGE_GROUPS)):
            s = (self.age == a) & keep
            rates[:, a] = np.average(expit(eta[s, None] + effect[None, :]), axis=0, weights=w[s])
        return effect, rates


def mad(x):
    return np.mean(np.abs(x - x.mean()))


def fit_models(d, countries, indicators):
    cc = d.dropna(subset=indicators)                            # complete cases for this set
    pain = cc[cc.ms]
    print(f"{len(cc)} of {len(d)} records with all of {', '.join(indicators)}")
    return [Logit(cc, countries, "ms", ["female"]),
            Logit(cc, countries, "ms", ["female"] + indicators),
            Logit(pain, countries, "un", ["female", "severe"]),
            Logit(pain, countries, "un", ["female", "severe"] + indicators)]


def estimate(models, mult, primary, L):
    # primary = pooled respondent-wave estimate on all records, for the comparison
    (p0_eff, p0), (p1_eff, p1), (u0_eff, u0), (u1_eff, u1) = (m.standardised(mult) for m in models)
    y0, y1 = (L * p0 * u0).sum(1), (L * p1 * u1).sum(1)
    by_country = {"P_effect": p1_eff, "U_effect": u1_eff, "years_standardised": y1,
                  "P_effect_unadjusted": p0_eff, "U_effect_unadjusted": u0_eff,
                  "years_unadjusted": y0, "years_primary_full_sample": primary}
    stats = {"spearman_P": spearmanr(p0_eff, p1_eff).statistic,
             "spearman_U": spearmanr(u0_eff, u1_eff).statistic,
             "spearman_years_same_sample": spearmanr(y0, y1).statistic,
             "spearman_primary_vs_same_sample": spearmanr(primary, y0).statistic,
             "spearman_primary_vs_standardised": spearmanr(primary, y1).statistic,
             "range_standardised": np.ptp(y1),
             "range_unadjusted": np.ptp(y0)}
    for name, before, after in [("P", p0_eff, p1_eff), ("U", u0_eff, u1_eff)]:
        for label, f in [("SD", lambda x: np.std(x, ddof=1)), ("MAD", mad)]:
            stats[f"{name}_{label}_unadjusted"] = f(before)
            stats[f"{name}_{label}_adjusted"] = f(after)
            stats[f"{name}_{label}_shrinkage_pct"] = 100 * (1 - f(after) / f(before))
    return by_country, stats


def main():
    args = options()
    out = results_dir()
    d = add_design(primary_sample(eligible(pd.read_parquet(data_dir() / "share_anchors.parquet"))))
    d["female"] = d.ragender.eq(2).astype(float)
    d["severe"] = d.painlv.eq(3).astype(float)
    countries = sorted(d.country.unique())
    L = life_table()
    cells = cell_codes(d, countries)

    def primary_years(mult):
        _, q = pooled(*totals(d, cells, mult))
        return q @ L

    models = {name: fit_models(d, countries, ind) for name, ind in INDICATORS.items()}
    one = np.ones(len(d))
    point = {name: estimate(m, one, primary_years(one), L) for name, m in models.items()}

    boot = {name: [] for name in models}
    for r, mult in enumerate(multipliers(d, args.reps, SEED), start=1):
        primary = primary_years(mult)
        for name, m in models.items():
            boot[name].append(estimate(m, mult, primary, L))
        if r % 100 == 0:
            print(f"{r} replicates", flush=True)        # ~15 min for 1,000

    by_country, summary = [], []
    for name, (est_c, est_s) in point.items():
        for q, est in est_c.items():
            lo, hi = interval(np.array([b[0][q] for b in boot[name]]))
            by_country.append(pd.DataFrame({"indicators": name, "country": countries, "quantity": q,
                                            "estimate": est, "ci_low": lo, "ci_high": hi}))
        for s, est in est_s.items():
            lo, hi = interval(np.array([b[1][s] for b in boot[name]]))
            summary.append({"indicators": name, "statistic": s, "estimate": est, "ci_low": lo, "ci_high": hi})
    pd.concat(by_country).to_csv(out / "tableS7_casemix_by_country.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "tableS7_casemix_summary.csv", index=False)


if __name__ == "__main__":
    main()
