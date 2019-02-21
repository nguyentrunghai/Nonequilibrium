"""
"""
from __future__ import print_function

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

parser.add_argument("--xlabel", type=str, default="$\lambda$")
parser.add_argument("--ylabel", type=str, default="$\Delta F_{\lambda}$")

parser.add_argument("--fe_out", type=str, default="fe_plots.pdf")
parser.add_argument("--pmf_out", type=str, default="pmf_plots.pdf")

args = parser.parse_args()


def _first_to_zero(values):
    return values - values[0]


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

    if label == "r_u":
        fe_x = fe_x[::-1]
        fe_y = fe_y[::-1]
        fe_error = fe_error[::-1]

    fe_y = _first_to_zero(fe_y)
    free_energies[label] = {"x":fe_x, "y":fe_y, "error":fe_error}

    pmf_x = bin_centers(data["pmfs"]["pmf_bin_edges"])
    pmf_ys = np.array(data["pmfs"]["main_estimates"].values())
    pmf_y = pmf_ys.mean(axis=0)
    pmf_error = pmf_ys.std(axis=0)
    pmfs[label] = {"x":pmf_x, "y":pmf_y, "error":pmf_error}

fe_num = pickle.load(open(num_fe_file, "r"))
pmf_exact = pickle.load(open(exact_pmf_file , "r"))

# plot free energies
xs = []
ys = []
yerrs = []
for label in data_estimator_pairs:
    xs.append(free_energies[label]["x"])
    ys.append(free_energies[label]["y"])
    yerrs.append(free_energies[label]["error"])

xs.append(fe_num["lambdas"])
ys.append(fe_num["fe"])
yerrs.append(None)

MARKERS = ["<", ">", "^", "v", "s", "d", "."]

plot_lines(xs, ys, yerrs=yerrs,
           xlabel=args.xlabel, ylabel=args.ylabel,
           out=args.fe_out,
           legends=data_estimator_pairs + ["num"],
           markers=MARKERS,
           legend_pos="best",
           legend_fontsize=7,
           xlimits=None,
           ylimits=None,
           lw=1.0,
           markersize=4,
           alpha=1.,
           n_xtics=8,
           n_ytics=8)