"""Sullivan estimates of years lived from age 65 in moderate-to-severe pain without
reported weekly pain medication (Table 2; Additional file 1: Table S4)."""

import numpy as np
import pandas as pd

AGE_GROUPS = ["65-69", "70-74", "75-79", "80-84", "85+"]

# Common reference life table: Eurostat demo_mlifetable, EU27_2020, both sexes, 2019.
# nL = T(x) - T(x+5), and T(85) for 85+; l65 = survivors at 65.
NL_EU27 = np.array([427272, 393528, 346827, 280246, 314786])
L65_EU27 = 88218
L_EU27 = NL_EU27 / L65_EU27      # sums to life expectancy at 65, 19.981 years


def weighted_totals(d, countries, waves, mult=1.0):
    """Weighted totals by country x wave x age group of all respondents, of those in
    moderate-to-severe pain (ms) and of those in pain without weekly medication (un).

    d has one row per respondent-wave: country, wave, age_group, w (respondent weight),
    ms and un (0/1). mult is the bootstrap multiplier of each row.
    """
    shape = (len(countries), len(waves), len(AGE_GROUPS))
    code = np.ravel_multi_index((pd.Categorical(d.country, countries).codes,
                                 pd.Categorical(d.wave, waves).codes,
                                 pd.Categorical(d.age_group, AGE_GROUPS).codes), shape)
    w = d.w.to_numpy() * mult
    return [np.bincount(code, w * x, np.prod(shape)).reshape(shape)
            for x in (1.0, d.ms.to_numpy(float), d.un.to_numpy(float))]


def wave_combined(den, ms, un):
    """Primary estimator: prevalence estimated within each wave with that wave's weights
    and averaged with equal weight over the waves a country took part in (Portugal has
    no wave-8 interviews and contributes wave 9 only).

    Returns p, the prevalence of moderate-to-severe pain, and q, the prevalence of the
    primary state, by country x age group.
    """
    with np.errstate(invalid="ignore"):
        return np.nanmean(ms / den, axis=1), np.nanmean(un / den, axis=1)


def pooled(den, ms, un):
    """Pooled respondent-wave estimator (weights of the two waves summed).
    Returns p and the conditional rate u = q / p."""
    den, ms, un = den.sum(1), ms.sum(1), un.sum(1)
    return ms / den, un / ms


def sullivan(p, q, L=L_EU27):
    """Years from 65 in the primary state, in pain with reported medication, and in
    moderate-to-severe pain overall: sum over age groups of L * prevalence."""
    u = q / p
    return (L * p * u).sum(1), (p - q) @ L, p @ L
