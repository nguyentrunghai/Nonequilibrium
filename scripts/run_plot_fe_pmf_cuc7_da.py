"""
TODO:
for f_u, r_u and fr_b, do not replicate
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

parser.add_argument("--free_energies_pmfs_files", type=str, default="file1 file2")
parser.add_argument("--us_fe_file", type=str, default="us_fe.pkl")
parser.add_argument("--exact_pmf_file", type=str, default="pmf_exact.pkl")

parser.add_argument( "--system_type", type=str, default="symmetric")

parser.add_argument("--data_estimator_pairs", type=str, default="s_u s_b s_s f_u r_u fr_b")

parser.add_argument("--fe_xlabel", type=str, default="$\lambda$")
parser.add_argument("--fe_ylabel", type=str, default="$\Delta F_{\lambda}$")

parser.add_argument("--pmf_xlabel", type=str, default="$z$")
parser.add_argument("--pmf_ylabel", type=str, default="$\Phi(z)$")
# for symmetric data plot the pmf from pmf[bin_ind_to_start_to_plot] to pmf[len - bin_ind_to_start_to_plot]
# for asymmetric data plot pmf from pmf[bin_ind_to_start_to_plot] to pmf[len]
parser.add_argument("--bin_ind_to_start_to_plot", type=int, default=0)

parser.add_argument( "--want_right_replicate_for_asym", action="store_true", default=False)

parser.add_argument("--legend_ncol_fe", type=int, default=1)
parser.add_argument("--legend_ncol_pmf", type=int, default=1)

parser.add_argument("--xlimits_fe", type=str, default="None")
parser.add_argument("--ylimits_fe", type=str, default="None")

parser.add_argument("--xlimits_pmf", type=str, default="None")
parser.add_argument("--ylimits_pmf", type=str, default="None")

parser.add_argument("--fe_out", type=str, default="fe.pdf")
parser.add_argument("--pmf_out", type=str, default="pmf.pdf")

args = parser.parse_args()


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

for label in data_estimator_pairs:
    if label not in ["s_u", "s_b", "s_s", "f_u", "r_u", "fr_b"]:
        raise ValueError("Unrecognized label: " + label)

free_energies = {}
pmfs = {}
for file, label in zip(free_energies_pmfs_files, data_estimator_pairs):
    data = pickle.load(open(file, "r"))

    # reverse order
    if label == "r_u":
        data = reverse_data_order(data)

    # replicate data
    if args.want_right_replicate_for_asym:
        if label in ["f_u", "r_u", "fr_b"]:
            print("Right replicat for", label)
            data = replicate_data(data, args.system_type)

    # put first of fes to zero
    data = put_first_of_fe_to_zero(data)

    # put argmin of pmf to pmf_exact["pmf"]
    data = put_argmin_of_pmf_to_target(data, pmf_exact["pmf"])

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


# plot free energies
xs = []
ys = []
yerrs = []
for label in data_estimator_pairs:
    xs.append(free_energies[label]["x"])
    ys.append(free_energies[label]["y"])
    yerrs.append(free_energies[label]["error"] / 2)  # error bars are one std

xs.append(fe_num["lambdas"])
ys.append(fe_num["fe"])
yerrs.append(None)

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
           legends=data_estimator_pairs + ["num"],
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


# plot pmfs
start_pmf_ind = args.bin_ind_to_start_to_plot

xs = []
ys = []
yerrs = []
for label in data_estimator_pairs:
    end_pmf_ind = len(pmfs[label]["x"]) - start_pmf_ind

    if label in ["f_u", "r_u", "fr_b"] and (not args.want_right_replicate_for_asym) and args.system_type == "symmetric":
        end_pmf_ind = len(pmfs[label]["x"])

    xs.append(pmfs[label]["x"][start_pmf_ind : end_pmf_ind])
    ys.append(pmfs[label]["y"][start_pmf_ind : end_pmf_ind])
    yerrs.append(pmfs[label]["error"][start_pmf_ind : end_pmf_ind] / 2)  # error bars are one std


end_pmf_ind = len(pmf_exact["pmf"]) - start_pmf_ind
xs.append(bin_centers(pmf_exact["pmf_bin_edges"])[start_pmf_ind : end_pmf_ind])
ys.append(pmf_exact["pmf"][start_pmf_ind : end_pmf_ind])
yerrs.append(None)

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
           legends=data_estimator_pairs + ["exact"],
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
