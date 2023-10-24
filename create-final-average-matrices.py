#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:56:17 2023

@author: albertakuno
"""

import json
import logging
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

logging.basicConfig(filename="run.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser(description="Calculate final average residence matrix.")
parser.add_argument(
    "--period", type=str, choices=["First", "Second", "Third"], help="Time period to run."
)
parser.add_argument(
    "--part", type=str, choices=["First", "Second"], help="Part in time period to run."
)

args = parser.parse_args()

PERIOD = args.period
PART = args.part

df_res = pd.read_csv(
    f"/Volumes/F/Hermosillo_AGEBs_data/final_time_periods/combined/{PERIOD}Period_{PART}Part_comb.csv",
    sep=";",
    header=0,
    names=["id", "ageb_crit2", "ageb_crit1", "loose"],
)
matrices_dict = json.load(
    open(
        f"/Volumes/F/Hermosillo_AGEBs_data/final_residence_matrices/final_residence_matrices_{PERIOD}_{PART}.json",
        "r",
    )
)

# final_matrix = np.zeros((582, 583))
# import pdb;pdb.set_trace()
final_matrix = np.zeros((582, 582))
for idx, group in tqdm(df_res.groupby("loose")):
    if idx == -1:
        print(group.shape)
        continue
    for ids in tqdm(group["id"], leave=False):
        temp = np.array(matrices_dict[ids])
        temp = temp / temp.sum() if temp.sum() != 0 else temp
        final_matrix[idx, :] += temp
    final_matrix[idx, :] /= group.shape[0]


agebs_to_remove = [i for i, row in enumerate(final_matrix) if row.sum() == 0]
final = np.delete(final_matrix, agebs_to_remove, 0)
final = np.delete(final, agebs_to_remove, 1)
final = final / final.sum(axis=1, keepdims=1)

json.dump(
        {"full_residence_matrix": final_matrix.tolist(), "truncated_residence_matrix": final.tolist(), "agebs_to_remove": agebs_to_remove},
    open(f"/Volumes/F/Hermosillo_AGEBs_data/avg_res_mat/avg_res_mat_{PERIOD}_{PART}.json", "w"),
    indent=4,
)