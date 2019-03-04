"""
"""

from __future__ import print_function
from __future__ import division

import os
import argparse
import glob
import pickle

import numpy as np

from _fe_pmf_plot_utils import first_to_zero, min_to_zero
from _fe_pmf_plot_utils import reverse_data_order, replicate_data
from _fe_pmf_plot_utils import put_first_of_fe_to_zero, put_argmin_of_pmf_to_target

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./")

parser.add_argument("--fes_pmfs_file_matching", type=str, default="symmetric_uf_ntrajs_*.pkl symmetric_b_ntrajs_*.pkl symmetric_s_ntrajs_*.pkl asymmetric_uf_ntrajs_*.pkl asymmetric_ur_ntrajs_*.pkl asymmetric_b_ntrajs_*.pkl")
parser.add_argument("--num_fe_file", type=str, default="fe_symmetric_numerical.pkl")
parser.add_argument("--exact_pmf_file", type=str, default="pmf_symmetric_exact.pkl")

parser.add_argument( "--system_type", type=str, default="symmetric")

parser.add_argument("--data_estimator_pairs", type=str, default="s_u s_b s_s f_u r_u fr_b")

parser.add_argument( "--pmf_bin_truncate", type=int, default=2)

args = parser.parse_args()


def _rmse(main_estimates, reference, truncate):
    begin = truncate
    end = len(reference) - truncate
    squared_deviations = [(estimates[begin : end] - reference[begin : end])**2
                          for estimates in main_estimates.values()]
    squared_deviations = np.array(squared_deviations)
    return squared_deviations.mean()


def _rmse_std_error(pull_data, reference, truncate):
    bootstrap_keys = [bt for bt in pull_data if bt.startswith("bootstrap_")]
    bootstrap_estimates = [_rmse(pull_data[bootstrap_key], reference, truncate) for bootstrap_key in bootstrap_keys]
    bootstrap_estimates = np.array(bootstrap_estimates)
    return bootstrap_estimates.std(axis=0)


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

num_fe_file = os.path.join(args.data_dir, args.num_fe_file)
print("num_fe_file", num_fe_file)

exact_pmf_file = os.path.join(args.data_dir, args.exact_pmf_file)
print("exact_pmf_file", exact_pmf_file)

fe_num = pickle.load(open(num_fe_file, "r"))
fe_num["fe"] = first_to_zero(fe_num["fe"])

pmf_exact = pickle.load(open(exact_pmf_file , "r"))
pmf_exact["pmf"] = min_to_zero(pmf_exact["pmf"])

fe_rmse  = {}
pmf_rmse = {}
for label in data_estimator_pairs:
    fe_rmse[label] = []
    pmf_rmse[label] = []

    for fes_pmfs_file in fes_pmfs_files[label]:
        data = pickle.load(open(fes_pmfs_file, "r"))

        if data["free_energies"]["ntrajs_per_block"] != data["pmfs"]["ntrajs_per_block"]:
            raise ValueError("ntrajs_per_block are not consistent in" + fes_pmfs_file)
        ntrajs = data["free_energies"]["ntrajs_per_block"]

        # reverse order
        if label == "r_u":
            data = reverse_data_order(data)

        # replicate data
        if label in ["f_u", "r_u", "fr_b"]:
            data = replicate_data(data, args.system_type)

        # put first of fes to zero
        data = put_first_of_fe_to_zero(data)

        # put argmin of pmf to pmf_exact["pmf"]
        data = put_argmin_of_pmf_to_target(data, pmf_exact["pmf"])

        fe_r = _rmse(data["free_energies"]["main_estimates"], fe_num["fe"], truncate=0)
        fe_r_error = _rmse_std_error(data["free_energies"], fe_num["fe"], truncate=0)
        fe_rmse[label].append((ntrajs, fe_r, fe_r_error))

        pmf_r = _rmse(data["pmfs"]["main_estimates"], pmf_exact["pmf"], truncate=args.pmf_bin_truncate)
        pmf_r_error = _rmse_std_error(data["pmfs"], pmf_exact["pmf"], truncate=args.pmf_bin_truncate)
        pmf_rmse[label].append((ntrajs, pmf_r, pmf_r_error))

    fe_rmse[label].sort(key=lambda item: item[0])
    pmf_rmse[label].sort(key=lambda item: item[0])