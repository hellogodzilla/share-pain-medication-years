# share-pain-medication-years

Analysis code for

Li X, Wu X, Liu S, Zhou G. Years lived in moderate-to-severe pain without reported weekly pain-medication use after age 65 in 26 European countries: a cross-sectional Sullivan analysis of the Survey of Health, Ageing and Retirement in Europe.

The primary state is moderate-to-severe pain without reported weekly pain-medication use; the complementary state is moderate-to-severe pain with reported weekly pain-medication use. The code starts from analysis-ready files (described below) and produces the country estimates, intervals and sensitivity analyses reported in the article. It does not include the step that extracts these files from the SHARE releases, and it contains no SHARE data.

## Files

| File | What it does | Article |
|---|---|---|
| `prepare.py` | Eligibility, age groups, pain states, primary country pool; stratum and PSU keys at household level | Methods |
| `bootstrap.py` | Rao–Wu rescaled bootstrap over the released design; percentile and rank intervals | Additional file 1, S9 |
| `sullivan.py` | Life tables, the wave-combined primary estimator, the four analysis units and the pooled respondent-wave estimator; Sullivan years with intervals and rank intervals; estimates on each country's own life table | Table 2, Fig. 1, Fig. S1; Tables S4, S14 |
| `decomposition.py` | Pairwise symmetric Das Gupta decomposition into a pain (P) and a medication-use (U) component; profiles, leave-one-country-out stability and bootstrap probabilities; country-mean against population-weighted reference | Table 2, Fig. 2; Table S3; S4 |
| `describe.py` | Sample by country; demographic rows of the comparison by pain and medication state; response distribution of the pain item | Table 1A, Table 1B (first eight rows); S1 |
| `trend.py` | Country changes from 2013 to 2021–22 and over the two shorter windows, with P and U components, and their cross-country medians | Fig. 4; Table S8 |
| `projection.py` | Counts in 2022 and 2050 at fixed rates; static standardisation scenarios under both benchmark conventions; within-85+ reweighting | Fig. 5; Table S9; S8 |
| `strata.py` | Primary outcome by sex (three sets of life tables) and by education | Tables S10, S11 |
| `casemix.py` | G-computation standardisation for nine health and functional-status indicators, and for three of them | Table S7; S5 |
| `sensitivity.py` | Severe pain only; pool composition and the 2019–2021 life table; the specification comparison | Tables S5, S13, S17 |
| `transitions.py` | Wave-8 to wave-9 transitions from the primary state among those re-interviewed | Table S18, panel B |
| `run_all.py` | Runs all of the above | |

Each script writes CSV files to `results/`. Bootstrap runs have 1,000 replicates. Seeds: 20260922 for the primary run (Table 2, Tables S4, S9, S14, and the sex and education strata, which replay its draws); 20260920 for the pooled respondent-wave estimator (Table S3 and the pooled row of Table S14); 20260921 for the case-mix models; 20260923 for the change over time. Transitions use 20260922 over the design of the 25 countries with wave-8 interviews.

## Data

SHARE data cannot be redistributed. They are free for scientific use to registered users: see https://share-eric.eu/data/become-a-user (registration with a signed User Statement sent to the SHARE Research Data Center). The analyses use the Gateway Harmonized SHARE, Version G (release 9.0.0), waves 5, 6, 8 and 9, and the `gv_weights` modules of the same releases for the design variables (DOIs 10.6103/SHARE.w5.900, 10.6103/SHARE.w6.900, 10.6103/SHARE.w8.900, 10.6103/SHARE.w9.900 and 10.25553/xxf8-j737).

The scripts read three files from the folder given by `--data` or the environment variable `SHARE_DATA` (default `data/share/`). Variable names are those of the harmonised files, without the wave prefix.

`share_records.parquet`, one row per respondent and wave, waves 5, 6, 8 and 9:

| Column | Content |
|---|---|
| `mergeid`, `hhid` | Person and household identifiers |
| `country` | Harmonised country code (11 Austria, 12 Germany, ... 63 Slovakia; see `prepare.py`) |
| `wave` | 5, 6, 8 or 9 |
| `iwstat` | Interview status; 1 = respondent interview |
| `wtresp` | Calibrated cross-sectional respondent weight |
| `agey` | Age at interview, years |
| `ragender` | 1 man, 2 woman |
| `raeducl` | Education: 1 less than upper secondary, 2 upper secondary or vocational, 3 tertiary |
| `painlv` | Usual pain intensity: 0 none, 1 mild, 2 moderate, 3 severe |
| `painmed` | 1 if drugs are taken at least weekly for joint pain or joint inflammation or for other pain (show-card item PH011), else 0 |

`share_design.parquet`: `mergeid`, `wave`, `stratum1` and `psu` from the `gv_weights` modules, waves 5, 6, 8 and 9.

`share_anchors.parquet`: waves 8 and 9, the columns of `share_records.parquet` (without `raeducl`) and the indicators `arthre`, `adla`, `iadla`, `gripsum`, `grossa`, `lgmusa`, `finea`, `hipe` and `fall_s`.

The folder `data/` holds the public inputs, taken from Eurostat:

- `life_tables.csv`: `demo_mlifetable`, EU27_2020 and the study countries, both sexes, men and women, 2019–2021. `nL` is the person-years lived in each five-year age group, obtained as T(x) − T(x+5) and T(85) for 85 and over; `l65` is the number of survivors at 65. The scripts use L = nL / l65.
- `population_2022_2050.csv`: `proj_23np`, baseline projection, persons aged 65 and over by five-year group, 2022 and 2050.
- `population_85plus_2022_2050.csv`: the same projection summed over single years into 85–89, 90–94 and 95 and over.

## Running

```
pip install -r requirements.txt
export SHARE_DATA=/path/to/analysis-ready-files
python run_all.py
```

Any script can be run on its own, for example `python sullivan.py --reps 200`. The case-mix bootstrap refits four logistic models per indicator set in every replicate and takes about 15 minutes on a recent laptop; everything else runs in under a minute.

## Not included

Analyses that need variables outside the three files above are not part of this repository: proxy-assisted and nursing-home interviews (Tables S2 and S15 and their rows in Table S17), the characteristics in Table 1B beyond its first eight rows and in Table S12, the within-question calibration (Table S16), diagnosed arthritis (Table S19, Fig. 3), panel A of Table S18, and the participant flow (Table S1). Also not included: partial pooling (Table S6), the Health and Retirement Study scenarios (Table S5) and the representation of the oldest old (Additional file 1, S10).

## Software

Python 3.14.0 with numpy 2.3.4, pandas 2.3.3, scipy 1.17.1, statsmodels 0.14.6 and pyarrow 25.0.0.

## License

MIT; see `LICENSE`.
