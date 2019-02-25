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

parser.add_argument("--fe_xlabel", type=str, default="$\lambda$")
parser.add_argument("--fe_ylabel", type=str, default="$\Delta F_{\lambda}$")

parser.add_argument("--pmf_xlabel", type=str, default="$z$")
parser.add_argument("--pmf_ylabel", type=str, default="$\Phi(z)$")
# for symmetric data plot the pmf from pmf[bin_ind_to_start_to_plot] to pmf[len - bin_ind_to_start_to_plot]
# for asymmetric data plot pmf from pmf[bin_ind_to_start_to_plot] to pmf[len]
parser.add_argument("--bin_ind_to_start_to_plot", type=int, default=1)

parser.add_argument("--legend_ncol", type=int, default=3)

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

    pmf_y = _min_to_zero(pmf_y)
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
           xlabel=args.fe_xlabel, ylabel=args.fe_ylabel,
           out=args.fe_out,
           legends=data_estimator_pairs + ["num"],
           legend_pos="best",
           legend_ncol=args.legend_ncol,
           legend_fontsize=8,
           markers=MARKERS,
           xlimits=None,
           ylimits=None,
           lw=1.0,
           markersize=4,
           alpha=1.,
           n_xtics=8,
           n_ytics=8)


# plot pmfs
start_pmf_ind = args.bin_ind_to_start_to_plot

xs = []
ys = []
yerrs = []
for label in data_estimator_pairs:
    if label in ["s_u", "s_b", "s_s"]:
        end_pmf_ind = len(pmfs[label]["x"]) - start_pmf_ind
    else:
        end_pmf_ind = len(pmfs[label]["x"])
    xs.append(pmfs[label]["x"][start_pmf_ind : end_pmf_ind])
    ys.append(pmfs[label]["y"][start_pmf_ind : end_pmf_ind])
    yerrs.append(pmfs[label]["error"][start_pmf_ind : end_pmf_ind])

end_pmf_ind = len(pmf_exact["pmf"]) - start_pmf_ind
xs.append(bin_centers(pmf_exact["pmf_bin_edges"])[start_pmf_ind : end_pmf_ind])
ys.append(pmf_exact["pmf"][start_pmf_ind : end_pmf_ind])
yerrs.append(None)

plot_lines(xs, ys, yerrs=yerrs,
           xlabel=args.pmf_xlabel, ylabel=args.pmf_ylabel,
           out=args.pmf_out,
           legends=data_estimator_pairs + ["exact"],
           legend_ncol=args.legend_ncol,
           legend_pos="best",
           legend_fontsize=8,
           markers=MARKERS,
           xlimits=None,
           ylimits=None,
           lw=1.0,
           markersize=4,
           alpha=1.,
           n_xtics=8,
           n_ytics=8)
