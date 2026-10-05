"""Change in years lived in the primary state between two waves (Fig 4;
Additional file 1: Table S8).

Windows: wave 5 (2013) to wave 9 (2021-22), wave 6 (2015) to wave 8 (2019-20) and wave 8
to wave 9, each limited to countries with at least 30 respondents in every age group in
both waves. Each wave's estimate uses its own respondent weights. Intervals: Rao-Wu
bootstrap (bootstrap.py) over the PSUs of both waves, so panel members carry the same
multiplier at both ends; the seed (20260923) is reset for each window, and a country is
dropped from a replicate if one of its age cells is empty.
"""

import numpy as np

WAVE_YEAR = {5: 2013, 6: 2015, 8: 2019.5, 9: 2021.5}


def change(den, ms, un, L):
    """Two-point symmetric Das Gupta decomposition of the change.

    den, ms, un: weighted totals by country x (start, end wave) x age group.
    Returns years at both waves (country x 2) and the pain (P) and medication-use (U)
    components of the change; P + U = years[:, 1] - years[:, 0].
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        p, u = ms / den, un / ms
    years = (L * p * u).sum(2)
    P = (L * (u[:, 1] + u[:, 0]) / 2 * (p[:, 1] - p[:, 0])).sum(1)
    U = (L * (p[:, 1] + p[:, 0]) / 2 * (u[:, 1] - u[:, 0])).sum(1)
    ok = (den > 0).all((1, 2)) & (ms > 0).all((1, 2))
    return years, np.where(ok, P, np.nan), np.where(ok, U, np.nan)


def per_decade(years, waves):
    return (years[:, 1] - years[:, 0]) * 10 / (WAVE_YEAR[waves[1]] - WAVE_YEAR[waves[0]])
