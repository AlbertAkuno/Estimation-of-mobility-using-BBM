#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 17:00:39 2022

@author: albertakuno
"""

import json
import logging
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
tqdm.pandas()
from residence import calculate_residence_matrix, calculate_sigma

from pandarallel import pandarallel

pandarallel.initialize(nb_workers=48, progress_bar=True)

logging.basicConfig(filename="run.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser(description="Residence-time runner file.")
parser.add_argument(
    "--period", type=str, choices=["First", "Second", "Third"], help="Time period to run."
)
parser.add_argument(
    "--part", type=str, choices=["First", "Second"], help="Part in time period to run."
)
parser.add_argument("--exp", type=str, choices=["sigma", "resmat"], help="Experiment to run")

args = parser.parse_args()

PERIOD = args.period
PART = args.part

logger.debug("Loading GPS data")
df_ids = pd.read_csv(
    f"/home/est_posgrado_albert.akuno/final_time_periods/combined/{PERIOD}Period_{PART}Part_comb.csv",
    sep=";",
    header=0,
    names=["id", "ageb_crit2", "ageb_crit1", "loose"],
)
# df_ids = pd.read_csv("/home/est_posgrado_albert.akuno/temp.csv", sep=";", header=0, names=["id", "ageb_crit2", "ageb_crit1", "loose"])
df_ids["loose"] = df_ids["loose"].astype(int)
df_full = pd.read_csv(
    f"/home/est_posgrado_albert.akuno/final_time_periods/criterion_1/{PERIOD}Period_{PART}Part_final.csv",
    sep=";",
    header=0,
    names=["id_adv", "timestamp", "lat", "lon", "polygon"],
)
df_full["timestamp"] = df_full["timestamp"].astype("datetime64[ns, UTC]")
ids = set(df_full["id_adv"].unique())

logger.debug("All data loaded")


def get_data(id_adv):
    # if id_adv in m1_ids:
    #     return df_m1.loc[df_m1["id_adv"]==id_adv].drop_duplicates(subset="timestamp")
    # elif id_adv in m2_ids:
    #     return df_m2.loc[df_m2["id_adv"]==id_adv].drop_duplicates(subset="timestamp")
    # elif id_adv in m3_ids:
    #     return df_m3.loc[df_m3["id_adv"]==id_adv].drop_duplicates(subset="timestamp")
    # else:
    #     raise ValueError(f"id not found in any file: {id_adv}")
    return df_full.loc[df_full["id_adv"] == id_adv]


def calc_res_mat(row):
    df_data = get_data(row["id"])
    return calculate_residence_matrix(df_data, row["loose"])


def calc_sigma(row):
    df_data = get_data(row["id"])
    return calculate_sigma(df_data)


def runner_res_mat():
    residence_matrices = {}
    df_ids["matrix"] = df_ids.parallel_apply(calc_res_mat, axis=1)

    for i, row in tqdm(df_ids.iterrows()):
        residence_matrices[row["id"]] = row["matrix"]
    json.dump(
        residence_matrices,
        open(f"/home/est_posgrado_albert.akuno/new_sigma_m/final_residence_matrices_{PERIOD}_{PART}.json", "w"),
        indent=4,
    )
    return


def runner_sigma():
    sigma_values = {}
    df_ids["sigma"] = df_ids.parallel_apply(calc_sigma, axis=1)

    for i, row in tqdm(df_ids.iterrows()):
        sigma_values[row["id"]] = row["sigma"]
    json.dump(
        sigma_values,
        open(f"/home/est_posgrado_albert.akuno/new_sigma_m/sigma_m_{PERIOD}_{PART}.json", "w"),
        indent=4,
    )
    return


if __name__ == "__main__":
    if args.exp == "sigma":
        runner_sigma()
    elif args.exp == "resmat":
        runner_res_mat()