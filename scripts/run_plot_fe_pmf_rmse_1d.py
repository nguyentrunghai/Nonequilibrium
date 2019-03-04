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

from _fe_pmf_plot_utils import first_to_zero, min_to_zero, reverse_data_order, replicate_data
from _fe_pmf_plot_utils import put_first_of_fe_to_zero, put_argmin_of_pmf_to_target

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="./")

parser.add_argument("--free_energies_pmfs_files", type=str, default="symmetric_uf_ntrajs_200.pkl symmetric_b_ntrajs_200.pkl symmetric_s_ntrajs_200.pkl asymmetric_uf_ntrajs_400.pkl asymmetric_ur_ntrajs_400.pkl asymmetric_b_ntrajs_400.pkl")
parser.add_argument("--num_fe_file", type=str, default="fe_symmetric_numerical.pkl")
parser.add_argument("--exact_pmf_file", type=str, default="pmf_symmetric_exact.pkl")

parser.add_argument( "--system_type", type=str, default="symmetric")

parser.add_argument("--data_estimator_pairs", type=str, default="s_u s_b s_s f_u r_u fr_b")

parser.add_argument("--fe_xlabel", type=str, default="$\lambda$")
parser.add_argument("--fe_ylabel", type=str, default="RMSE[$\Delta F_{\lambda}$]")

parser.add_argument("--pmf_xlabel", type=str, default="$z$")
parser.add_argument("--pmf_ylabel", type=str, default="RMSE[$\Phi(z)$]")

parser.add_argument("--bin_ind_to_start_to_plot", type=int, default=1)

parser.add_argument("--legend_ncol_fe", type=int, default=1)
parser.add_argument("--legend_ncol_pmf", type=int, default=1)

parser.add_argument("--xlimits_fe", type=str, default="None")
parser.add_argument("--ylimits_fe", type=str, default="-0.1 1.1")

parser.add_argument("--xlimits_pmf", type=str, default="None")
parser.add_argument("--ylimits_pmf", type=str, default="-0.1 1.5")

parser.add_argument("--fe_out", type=str, default="fe_rmse.pdf")
parser.add_argument("--pmf_out", type=str, default="pmf_rmse.pdf")

args = parser.parse_args()


def _rmse(main_estimates, reference):
    squared_deviations = [(estimates - reference)**2 for estimates in main_estimates.values()]
    squared_deviations = np.array(squared_deviations)
    return squared_deviations.mean(axis=0)


def _rmse_std_error(pull_data, reference):
    bootstrap_keys = [bt for bt in pull_data if bt.startswith("bootstrap_")]
    bootstrap_estimates = [_rmse(pull_data[bootstrap_key], reference) for bootstrap_key in bootstrap_keys]
    bootstrap_estimates = np.array(bootstrap_estimates)
    return bootstrap_estimates.std(axis=0)


free_energies_pmfs_files = [os.path.join(args.data_dir, f) for f in args.free_energies_pmfs_files.split()]
print("free_energies_pmfs_files", free_energies_pmfs_files)

num_fe_file = os.path.join(args.data_dir, args.num_fe_file)
print("num_fe_file", num_fe_file)

exact_pmf_file = os.path.join(args.data_dir, args.exact_pmf_file)
print("exact_pmf_file", exact_pmf_file)

fe_num = pickle.load(open(num_fe_file, "r"))
fe_num["fe"] = first_to_zero(fe_num["fe"])

pmf_exact = pickle.load(open(exact_pmf_file , "r"))
pmf_exact["pmf"] = min_to_zero(pmf_exact["pmf"])

data_estimator_pairs = args.data_estimator_pairs.split()
if len(data_estimator_pairs) != len(free_energies_pmfs_files):
    raise ValueError("data_estimator_pairs and free_energies_pmfs_files not consistent")

free_energies = {}
pmfs = {}

all_data = {}
for file_name, label in zip(free_energies_pmfs_files, data_estimator_pairs):
    data = pickle.load(open(file_name, "r"))

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

    all_data[label] = data

fe_rmse = {}
fe_rmse_std_error = {}
pmf_rmse = {}
pmf_rmse_std_error = {}
for label in all_data:
    fe_rmse[label] = _rmse(all_data[label]["free_energies"]["main_estimates"], fe_num["fe"])
    fe_rmse_std_error[label] = _rmse_std_error(all_data[label]["free_energies"], fe_num["fe"])

    pmf_rmse[label] = _rmse(all_data[label]["pmfs"]["main_estimates"], pmf_exact["pmf"])
    pmf_rmse_std_error[label] = _rmse_std_error(all_data[label]["pmfs"], pmf_exact["pmf"])

# plot fe rmse
xs = []
ys = []
yerrs = []
for label in data_estimator_pairs:
    xs.append(all_data[label]["free_energies"]["lambdas"])
    ys.append(fe_rmse[label])
    yerrs.append(fe_rmse_std_error[label] / 2)

if args.xlimits_fe.lower() != "none":
    xlimits_fe = [float(s) for s in args.xlimits_fe.split()]
else:
    xlimits_fe = None

if args.ylimits_fe.lower() != "none":
    ylimits_fe = [float(s) for s in args.ylimits_fe.split()]
else:
    ylimits_fe = None

MARKERS = ["<", ">", "^", "v", "s", "d", "."]

plot_lines(xs, ys, yerrs=yerrs,
           xlabel=args.fe_xlabel, ylabel=args.fe_ylabel,
           out=args.fe_out,
           legends=data_estimator_pairs,
           legend_pos="best",
           legend_ncol=args.legend_ncol_fe,
           legend_fontsize=8,
           markers=MARKERS,
           xlimits=xlimits_fe,
           ylimits=ylimits_fe,
           lw=1.0,
           markersize=4,
           alpha=1.,
           n_xtics=8,
           n_ytics=8)


# plot pmf rmse
start_pmf_ind = args.bin_ind_to_start_to_plot

xs = []
ys = []
yerrs = []
for label in data_estimator_pairs:
    x = bin_centers(all_data[label]["pmfs"]["pmf_bin_edges"])

    end_pmf_ind = len(x) - start_pmf_ind
    xs.append(x[start_pmf_ind : end_pmf_ind])
    ys.append(pmf_rmse[label][start_pmf_ind : end_pmf_ind])
    yerrs.append(pmf_rmse_std_error[label][start_pmf_ind : end_pmf_ind] / 2)

if args.xlimits_pmf.lower() != "none":
    xlimits_pmf = [float(s) for s in args.xlimits_pmf.split()]
else:
    xlimits_pmf = None

if args.ylimits_pmf.lower() != "none":
    ylimits_pmf = [float(s) for s in args.ylimits_pmf.split()]
else:
    ylimits_pmf = None

plot_lines(xs, ys, yerrs=yerrs,
           xlabel=args.pmf_xlabel, ylabel=args.pmf_ylabel,
           out=args.pmf_out,
           legends=data_estimator_pairs,
           legend_ncol=args.legend_ncol_pmf,
           legend_pos="best",
           legend_fontsize=8,
           markers=MARKERS,
           xlimits=xlimits_pmf,
           ylimits=ylimits_pmf,
           lw=1.0,
           markersize=4,
           alpha=1.,
           n_xtics=8,
           n_ytics=8)