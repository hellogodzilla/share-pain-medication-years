"""Sample selection and design keys shared by the scripts. Input files: see README."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "data"

AGE_GROUPS = ["65-69", "70-74", "75-79", "80-84", "85+"]
AGE_EDGES = [65, 70, 75, 80, 85, np.inf]
MIN_CELL = 30

# Gateway Harmonized SHARE country codes
COUNTRIES = {
    11: "Austria", 12: "Germany", 13: "Sweden", 14: "Netherlands", 15: "Spain", 16: "Italy",
    17: "France", 18: "Denmark", 19: "Greece", 20: "Switzerland", 23: "Belgium", 25: "Israel",
    28: "Czech Republic", 29: "Poland", 31: "Luxembourg", 32: "Hungary", 33: "Portugal",
    34: "Slovenia", 35: "Estonia", 47: "Croatia", 48: "Lithuania", 51: "Bulgaria", 53: "Cyprus",
    55: "Finland", 57: "Latvia", 59: "Malta", 61: "Romania", 63: "Slovakia",
}


def options(reps=1000):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.environ.get("SHARE_DATA", HERE / "data" / "share"),
                        help="folder with the analysis-ready SHARE files")
    parser.add_argument("--out", default=os.environ.get("RESULTS", HERE / "results"))
    parser.add_argument("--reps", type=int, default=reps, help="bootstrap replicates")
    args = parser.parse_args()
    os.environ["SHARE_DATA"] = str(args.data)
    os.environ["RESULTS"] = str(args.out)
    return args


def data_dir():
    return Path(os.environ.get("SHARE_DATA", HERE / "data" / "share"))


def results_dir():
    out = Path(os.environ.get("RESULTS", HERE / "results"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def clean_id(s):
    s = s.astype("string").str.strip()
    return s.mask(s.str.lower().isin(["", "nan", "none", ".", "<na>"]))


def eligible(d):
    # respondent interviews at 65+, valid pain and medication answers, weight > 0
    d = d[d.iwstat.eq(1) & d.agey.ge(65) & d.wtresp.gt(0)
          & d.painlv.isin([0, 1, 2, 3]) & d.painmed.isin([0, 1])].copy()
    d["country"] = d.country.astype(int).map(COUNTRIES)
    d = d[d.country.notna()].copy()
    d["mergeid"] = clean_id(d.mergeid)
    d["hhid"] = clean_id(d.hhid)
    d["hh"] = d.country + "::" + d.hhid
    d["age_group"] = pd.cut(d.agey, AGE_EDGES, right=False, labels=AGE_GROUPS).astype(str)
    d["w"] = d.wtresp.astype("float64")
    d["ms"] = d.painlv.isin([2, 3])          # moderate or severe pain
    d["un"] = d.ms & d.painmed.eq(0)         # primary state
    return d.reset_index(drop=True)


def load_records(waves=(8, 9)):
    d = pd.read_parquet(data_dir() / "share_records.parquet")
    return eligible(d[d.wave.isin(waves)])


def primary_sample(d):
    d = d[d.wave.isin([8, 9]) & d.country.ne("Israel")]
    d = d[~(d.country.eq("Portugal") & d.wave.eq(8))]       # no regular wave-8 fieldwork in PT
    return d[d.country.isin(countries_with_full_cells(d))].reset_index(drop=True)


def countries_with_full_cells(d):
    n = d.groupby(["country", "age_group"]).size().unstack(fill_value=0)
    n = n.reindex(columns=AGE_GROUPS, fill_value=0)
    return sorted(n.index[(n >= MIN_CELL).all(axis=1)])


def add_design(d):
    design = pd.read_parquet(data_dir() / "share_design.parquet",
                             columns=["mergeid", "wave", "stratum1", "psu"])
    design["wave"] = design.wave.astype(int)
    for c in ["mergeid", "stratum1", "psu"]:
        design[c] = clean_id(design[c])
    d = d.merge(design, on=["mergeid", "wave"], how="left", validate="many_to_one")

    # one stratum/PSU per household (first non-missing in any wave); a household with no
    # PSU is its own PSU, and single-PSU strata are pooled within country
    for c in ["stratum1", "psu"]:
        first = (d.loc[d[c].notna(), ["hh", c]].sort_values(c, kind="stable")
                 .drop_duplicates("hh").set_index("hh")[c])
        d[c] = d.hh.map(first)
    d["stratum"] = d.country + "|" + d.stratum1.fillna("_NOSTR")
    d["psu"] = (d.stratum + "|" + d.psu).where(d.psu.notna(), d.country + "|hh|" + d.hhid)
    single = d.groupby("stratum").psu.transform("nunique").eq(1)
    d.loc[single, "stratum"] = d.country + "|_SINGLETON_POOL"
    return d


def primary_records():
    # 78,127 respondent-wave records
    return add_design(primary_sample(load_records([8, 9])))


def geo_codes(countries):
    lt = pd.read_csv(PUBLIC / "life_tables.csv").drop_duplicates("country").set_index("country")
    return lt.geo.loc[countries].to_numpy()


def population(countries, year):
    # Eurostat proj_23np, country x age group
    pop = pd.read_csv(PUBLIC / "population_2022_2050.csv").set_index(["geo", "age_group"])
    idx = pd.MultiIndex.from_product([geo_codes(countries), AGE_GROUPS])
    return pop.loc[idx, f"pop_{year}"].to_numpy(float).reshape(len(countries), -1)
