
from __future__ import print_function

import os
import pickle
import argparse

import numpy as np

from _plots import plot_lines

from utils import bin_centers

parser = argparse.ArgumentParser()

parser.add_argument("--pull_fe_pmf_dir", type=str, default="/home/tnguye46/nonequilibrium/ALA_Namd/pull_fe_pmf/20A_per_2ns/full_traj")

parser.add_argument("--estimators", type=str, default="u b s1 s2")

parser.add_argument("--fe_nsamples",    type=int, default=40)

args = parser.parse_args()


def _transform_pmf(pmf_to_be_transformed, pmf_target):
    """
    transfrom such that transformed_pmf[argmin] == pmf_target[argmin]
    """
    assert pmf_to_be_transformed.ndim == pmf_target.ndim == 1, "pmf_to_be_transformed and pmf_target must be 1d"
    assert pmf_to_be_transformed.shape == pmf_target.shape, "pmf_to_be_transformed and pmf_target must have the same shape"
    argmin = np.argmin(pmf_target)
    d = pmf_target[argmin] - pmf_to_be_transformed[argmin]
    transformed_pmf = pmf_to_be_transformed + d
    return transformed_pmf


def _transform_pmfs(pmfs, estimators):
    pmf_target = pmfs["us"]["mean"]
    for e in estimators:
        pmfs[e]["mean"] = _transform_pmf(pmfs[e]["mean"], pmf_target)

        for r in pmfs[e]["pmfs"]:
            pmfs[e]["pmfs"][r] = _transform_pmf(pmfs[e]["pmfs"][r], pmf_target)
    return None


def _rmse(list_of_est, true_val):
    """
    """
    if isinstance(list_of_est, list):
        list_of_est = np.array(list_of_est)
    assert list_of_est.ndim ==  true_val.ndim + 1, "list_of_est must have one more dimension than true_val"
    assert list_of_est.shape[1:] == true_val.shape, "list_of_est.shape[1:] == true_val.shape" 

    rmse = list_of_est - true_val[np.newaxis, :]
    rmse = rmse * rmse
    rmse = rmse.mean(axis=0)
    rmse = np.sqrt(rmse)
    return rmse


def equal_stride_subsample(data, nsamples):
    assert nsamples < len(data), "nsamples must less than len of data"
    new_index = np.linspace(0, len(data)-1, nsamples, dtype=int)
    return data[new_index]


estimators = args.estimators.split()

fes_files = {estim : os.path.join(args.pull_fe_pmf_dir, "fe_"+estim+".pkl") for estim in estimators}
print(fes_files)
fes = { est : pickle.load( open(fes_files[est], "r") ) for est in fes_files }

pmfs_files = {estim : os.path.join(args.pull_fe_pmf_dir, "pmf_"+estim+".pkl") for estim in estimators}
pmfs = { est : pickle.load( open(pmfs_files[est], "r") ) for est in pmfs_files }


#----------
MARKERS = ["v", "^", "<", ">", "o"]

# plot pulling and us free energies
xlabel = "$\lambda$ (nm)"
ylabel = "$\Delta F_{\lambda}$ (RT)"
legends = estimators
out = "pull_fes.pdf"

xs = [ equal_stride_subsample( fes[esti]["lambdas"], args.fe_nsamples) for esti in estimators]
ys = [ equal_stride_subsample( fes[esti]["mean"], args.fe_nsamples) for esti in estimators]
yerrs = [ equal_stride_subsample( fes[esti]["std"], args.fe_nsamples) for esti in estimators]

plot_lines(xs, ys, yerrs=yerrs, 
            xlabel=xlabel, ylabel=ylabel, out=out, 
            legends=legends,
            legend_pos="best",
            legend_fontsize=8,
            lw=1.0,
            alpha=1.0,
            n_xtics=6,
            n_ytics=10)


# plot pmfs
#_transform_pmfs(pmfs, estimators)

xlabel = "$d$ (nm)"
ylabel = "$\Phi (d)$ (RT)"
legends = estimators
out = "pmfs.pdf"

xs = [ bin_centers( pmfs[est]["pmf_bin_edges"] ) for est in legends ]
ys = [ pmfs[est]["mean"] for est in legends ]
yerrs = [ pmfs[est]["std"] for est in legends ]

plot_lines(xs, ys, yerrs=yerrs,
            xlabel=xlabel, ylabel=ylabel, out=out,
            legends=legends,
            legend_pos="best",
            legend_fontsize=8,
            lw=1.0,
            alpha=1.0,
            n_xtics=6,
            n_ytics=10)

