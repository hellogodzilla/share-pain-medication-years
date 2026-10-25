"""Symmetric Das Gupta decomposition (Table 2, Fig. 2, Table S3, methods S4).

For countries c and j:
  P_cj = sum_g L_g (u_cg + u_jg)/2 (p_cg - p_jg)
  U_cj = sum_g L_g (p_cg + p_jg)/2 (u_cg - u_jg)
so P_cj + U_cj = years_c - years_j. Averaging over all j (c included) gives P + U = the
difference from the country mean; population weights give the weighted mean instead.
"""

import numpy as np
import pandas as pd

from bootstrap import spearman
from prepare import options, population, primary_records, results_dir


def das_gupta(p, u, L, weight=None):
    yrs = (L * p * u).sum(1)
    P = (L * (u[:, None] + u[None]) / 2 * (p[:, None] - p[None])).sum(2)
    U = (L * (p[:, None] + p[None]) / 2 * (u[:, None] - u[None])).sum(2)
    if weight is None:
        return yrs, P.mean(1), U.mean(1)
    return yrs, P @ weight, U @ weight


def profile(yrs, P, U, reference=None):
    if reference is None:
        reference = yrs.mean()
    side = np.where(yrs > reference, "above mean", "below mean")
    main = np.where(np.abs(P) >= np.abs(U), "mainly pain", "mainly medication use")
    return np.char.add(np.char.add(side, ", "), main)


def leave_one_out(p, u, L):
    labels = profile(*das_gupta(p, u, L))
    changes = np.zeros(len(p), int)
    for j in range(len(p)):
        keep = np.arange(len(p)) != j
        changes[keep] += profile(*das_gupta(p[keep], u[keep], L)) != labels[keep]
    return changes


def population_share(countries):
    pop = population(countries, 2022).sum(1)
    return pop / pop.sum()


def main():
    from sullivan import life_table, point_rates, pooled_bootstrap  # sullivan.py imports this module

    args = options()
    out = results_dir()
    d = primary_records()
    countries = sorted(d.country.unique())
    L = life_table()

    # Table S3, on the pooled respondent-wave estimator
    (p, q), reps = pooled_bootstrap(d, countries, args.reps)
    yrs, P, U = das_gupta(p, q / p, L)
    labels = profile(yrs, P, U)
    same = np.mean([profile(*das_gupta(a, b / a, L)) == labels for a, b in reps], axis=0)
    s3 = pd.DataFrame({"country": countries, "years": yrs, "P": P, "U": U,
                       "difference_from_mean": P + U, "U_minus_P": U - P, "profile": labels,
                       "leave_one_out_changes": leave_one_out(p, q / p, L),
                       "bootstrap_probability": same})
    s3.sort_values("bootstrap_probability").to_csv(out / "tableS3_profiles.csv", index=False)

    # Reference point: country mean against population-weighted mean (wave-combined)
    p, q = point_rates(d, countries)["wave_combined"]
    share = population_share(countries)
    yrs, P_eq, U_eq = das_gupta(p, q / p, L)
    _, P_pw, U_pw = das_gupta(p, q / p, L, share)
    ref = pd.DataFrame({"country": countries, "population_share": share, "years": yrs,
                        "P_country_mean": P_eq, "U_country_mean": U_eq,
                        "P_population_weighted": P_pw, "U_population_weighted": U_pw})
    ref.to_csv(out / "decomposition_reference.csv", index=False)
    print(f"country mean {yrs.mean():.3f}, population-weighted mean {yrs @ share:.3f}")
    print(f"additivity: {np.abs(P_eq + U_eq - (yrs - yrs.mean())).max():.1e} and "
          f"{np.abs(P_pw + U_pw - (yrs - yrs @ share)).max():.1e}")
    print(f"Spearman P {spearman(P_eq, P_pw):.3f}, U {spearman(U_eq, U_pw):.3f}")


if __name__ == "__main__":
    main()
