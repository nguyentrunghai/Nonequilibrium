"""
"""

from __future__ import print_function
from __future__ import division

#import seaborn as sns
#sns.set()

import argparse
import pickle
import os

import numpy as np

from utils import bin_centers
from _plots import plot_lines

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="./")

parser.add_argument("--free_energies_pmfs_files", type=str, default="symmetric_uf_ntrajs_200.pkl symmetric_b_ntrajs_200.pkl symmetric_s_ntrajs_200.pkl asymmetric_uf_ntrajs_400.pkl asymmetric_ur_ntrajs_400.pkl asymmetric_b_ntrajs_400.pkl")
parser.add_argument("--num_fe_file", type=str, default="fe_symmetric_numerical.pkl")
parser.add_argument("--exact_pmf_file", type=str, default="pmf_symmetric_exact.pkl")

parser.add_argument("--data_estimator_pairs", type=str, default="s_u s_b s_s f_u r_u fr_b")

parser.add_argument("--fe_xlabel", type=str, default="$\lambda$")
parser.add_argument("--fe_ylabel", type=str, default="$\Delta F_{\lambda}$")

parser.add_argument("--pmf_xlabel", type=str, default="$z$")
parser.add_argument("--pmf_ylabel", type=str, default="$\Phi(z)$")
# for symmetric data plot the pmf from pmf[bin_ind_to_start_to_plot] to pmf[len - bin_ind_to_start_to_plot]
# for asymmetric data plot pmf from pmf[bin_ind_to_start_to_plot] to pmf[len]
parser.add_argument("--bin_ind_to_start_to_plot", type=int, default=1)

parser.add_argument("--legend_ncol", type=int, default=3)

parser.add_argument("--xlimits_fe", type=str, default="None")
parser.add_argument("--ylimits_fe", type=str, default="None")

parser.add_argument("--xlimits_pmf", type=str, default="None")
parser.add_argument("--ylimits_pmf", type=str, default="None")

parser.add_argument("--fe_out", type=str, default="fe_plots.pdf")
parser.add_argument("--pmf_out", type=str, default="pmf_plots.pdf")

args = parser.parse_args()

def _first_to_zero(values):
    return values - values[0]


def _min_to_zero(values):
    argmin = np.argmin(values)
    return values - values[argmin]

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
for file_name, label in zip(free_energies_pmfs_files, data_estimator_pairs):
    data = pickle.load(open(file_name, "r"))

    fe_x = data["free_energies"]["lambdas"]
    fe_ys = np.array(data["free_energies"]["main_estimates"].values())
