"""
"""
from __future__ import print_function

import argparse
import pickle
import os

import numpy as np

from utils import bin_centers

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="simulation")

parser.add_argument("--free_energies_pmfs_files", type=str, default="fe1 fe2")
parser.add_argument("--num_fe_file", type=str, default="num")
parser.add_argument("--exact_pmf_file", type=str, default="exact")

parser.add_argument("--data_estimator_pairs", type=str, default="s_u s_b s_s f_u r_u fr_b")

args = parser.parse_args()

free_energies_pmfs_files = [os.path.join(args.data_dir, f) for f in args.free_energies_pmfs_files.split()]
print("free_energies_pmfs_files", free_energies_pmfs_files)

num_fe_file = os.path.join(args.data_dir, args.num_fe_file)
print("num_fe_file", num_fe_file)

exact_pmf_file = os.path.join(args.data_dir, args.exact_pmf_file)
print("exact_pmf_file", exact_pmf_file)

data_estimator_pairs = args.data_estimator_pairs.split()
if len(data_estimator_pairs) != len(free_energies_pmfs_files):
    raise ValueError("data_estimator_pairs and free_energies_pmfs_files not consistent")

free_energies = {}
pmfs = {}
for file, label in zip(free_energies_pmfs_files, data_estimator_pairs):
    data = pickle.load(open(file, "r"))

    fe_x = data["free_energies"]["lambdas"]
    fe_ys = np.array(data["free_energies"]["main_estimates"].values())
    fe_y = fe_ys.mean(axis=0)
    fe_error = fe_ys.std(axis=0)
    free_energies[label] = {"x":fe_x, "y":fe_y, "error":fe_error}

    pmf_x = bin_centers(data["pmfs"]["pmf_bin_edges"])
    pmf_ys = np.array(data["pmfs"]["main_estimates"].values())
    pmf_y = pmf_ys.mean(axis=0)
    pmf_error = pmf_ys.std(axis=0)
    pmfs[label] = {"x":pmf_x, "y":pmf_y, "error":pmf_error}
    