# Years lived in moderate-to-severe pain without reported weekly pain-medication use after age 65

Code for the main analyses of: Li X, Wu X, Liu S, Zhou G. *Years lived in moderate-to-severe pain without reported weekly pain-medication use after age 65 in 26 European countries: a cross-sectional Sullivan analysis of the Survey of Health, Ageing and Retirement in Europe.*

| File | Analysis | Article |
|---|---|---|
| `sullivan.py` | Wave-combined prevalence and Sullivan estimates | Table 2, Table S4 |
| `decomposition.py` | Pairwise symmetric Das Gupta decomposition; component profiles | Table 2, Fig 2, Table S3 |
| `bootstrap.py` | Rao–Wu bootstrap over the SHARE design; 95% intervals, rank intervals, profile probabilities | Table 2, Fig 1, Tables S3–S4 |
| `gcomputation.py` | Standardisation for nine health and functional-status indicators | Table S7, Methods S5 |
| `trend.py` | Change between waves and its pain and medication-use components | Fig 4, Table S8 |
| `projection.py` | 2050 fixed-rate standardisation scenarios | Fig 5, Table S9 |

Bootstrap: 1,000 replicates; seeds 20260922 (main estimates and 2050 scenarios), 20260920 (profile probabilities, pooled respondent-wave estimator), 20260923 (change over time) and 20260921 (G-computation).

SHARE data are available to registered users at https://share-eric.eu/data/become-a-user and cannot be redistributed. The analyses use Gateway Harmonized SHARE (version G) waves 5, 6, 8 and 9 and the `gv_weights` modules of SHARE release 9.0.0. Country-level estimates are in Additional file 3 of the article.

Python 3.14, numpy 2.3.4, pandas 2.3.3, scipy 1.17.1, statsmodels 0.14.6.
