"""
"""

from __future__ import print_function
from __future__ import division

import os
import argparse
import glob

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./")

parser.add_argument("--fes_pmfs_file_matching", type=str, default="symmetric_uf_ntrajs_*.pkl symmetric_b_ntrajs_*.pkl symmetric_s_ntrajs_*.pkl asymmetric_uf_ntrajs_*.pkl asymmetric_ur_ntrajs_*.pkl asymmetric_b_ntrajs_*.pkl")
parser.add_argument("--num_fe_file", type=str, default="fe_symmetric_numerical.pkl")
parser.add_argument("--exact_pmf_file", type=str, default="pmf_symmetric_exact.pkl")

parser.add_argument( "--system_type", type=str, default="symmetric")

parser.add_argument("--data_estimator_pairs", type=str, default="s_u s_b s_s f_u r_u fr_b")

args = parser.parse_args()

fes_pmfs_file_matching = args.fes_pmfs_file_matching.split()
print("fes_pmfs_file_matching", fes_pmfs_file_matching)

data_estimator_pairs = args.data_estimator_pairs.split()
print("data_estimator_pairs", data_estimator_pairs)

if len(fes_pmfs_file_matching) != len(data_estimator_pairs):
    raise ValueError("len of fes_pmfs_file_matching not the same as data_estimator_pair")

fes_pmfs_files = {}
for label, matching in zip(data_estimator_pairs, fes_pmfs_file_matching):
    fes_pmfs_files[label] = glob.glob(os.path.join(args.data_dir, matching))


number_of_files = [len(files) for files in fes_pmfs_files.values()]
if np.unique(number_of_files).shape[0] != 1:
    raise ValueError("different labels do not have the same number of files")
