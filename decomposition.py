"""Pairwise symmetric Das Gupta decomposition and component profiles
(Table 2, Fig 2; Additional file 1: Table S3)."""

import numpy as np


def das_gupta(p, u, L):
    """Split each country's difference from the country mean into a pain component (P)
    and a medication-use component (U).

    years = sum(L * p * u), with p the prevalence of moderate-to-severe pain and u the
    proportion of those in pain without weekly medication. Each country is decomposed
    against every country in the pool, itself included (which contributes zero), and the
    pairwise components are averaged, so that P + U = years - mean(years).
    """
    years = (L * p * u).sum(1)
    P = (L * ((u[:, None] + u[None]) / 2) * (p[:, None] - p[None])).sum(2).mean(1)
    U = (L * ((p[:, None] + p[None]) / 2) * (u[:, None] - u[None])).sum(2).mean(1)
    return years, P, U


def profile(years, P, U):
    """Above or below the country mean, and the component larger in absolute value."""
    side = np.where(years > years.mean(), "above mean", "below mean")
    main = np.where(np.abs(P) >= np.abs(U), "mainly pain", "mainly medication use")
    return np.char.add(np.char.add(side, ", "), main)


def leave_one_out(p, u, L):
    """Number of times each country's profile changes when one other country is left out."""
    labels = profile(*das_gupta(p, u, L))
    changes = np.zeros(len(p), int)
    for j in range(len(p)):
        keep = np.arange(len(p)) != j
        changes[keep] += profile(*das_gupta(p[keep], u[keep], L)) != labels[keep]
    return changes
