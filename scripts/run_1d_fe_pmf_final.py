"""
estimate free energy differences and PMF using unidirectional estimator for pulling simulation data
"""

from __future__ import print_function

import argparse
import pickle

import numpy as np
import netCDF4 as nc

from _IO import load_1d_sim_results


from models_1d import U0_sym, U0_asym, V, U_sym, U_asym, numerical_df_t
from utils import equal_spaced_bins, bin_centers

from utils import right_wrap, left_wrap
from utils import right_replicate_fe, left_replicate_fe
from utils import right_replicate_bin_edges, left_replicate_bin_edges
from utils import right_replicate_pmf, left_replicate_pmf

from _fe_pmf import pull_fe_pmf


parser = argparse.ArgumentParser()

parser.add_argument( "--pulling_data_nc_file",           type=str,
                     default="/home/tnguye46/nonequilibrium/1d/asym/simulation/m1.5_to_p1.5_backto_m1.5/work_500rep_1000traj.nc")

parser.add_argument( "--other_pulling_data_nc_files",    type=str,
                     default="/home/tnguye46/nonequilibrium/1d/asym/simulation/m1.5_to_p1.5/work_500rep_1000traj.nc")   # to determine pmf bin

parser.add_argument( "--pmf_nbins",                      type=int, default=24)

parser.add_argument( "--system_type",    type=str,   default="asymmetric")       #  # symmetric or asymmetric
# for symmetric systems, asymmetric protocol means pulling only one half of the pmf
parser.add_argument( "--protocol_type",  type=str,   default="symmetric")   # symmetric or asymmetric

parser.add_argument( "--symmetric_center",   type=float, default=0) # some number or -999 (means None)

# right wrap z around pmf_bin_symm_center,
# if z > pmf_bin_symm_center, do this z = 2*pmf_bin_symm_center - z
parser.add_argument( "--right_wrap", action="store_true", default=False)

# left wrap z around pmf_bin_symm_center,
# if z < pmf_bin_symm_center, do this z = 2*pmf_bin_symm_center - z
parser.add_argument( "--left_wrap", action="store_true", default=False)

parser.add_argument( "--estimators",  type=str, default="uf ur b s1 s2")

parser.add_argument( "--fe_out_prefix",   type=str, default="fe")
parser.add_argument( "--pmf_out_prefix",  type=str, default="pmf")

args = parser.parse_args()

assert args.system_type in ["symmetric", "asymmetric"], "unknown system_type"
assert args.protocol_type in ["symmetric", "asymmetric"], "unknown protocol_type"
assert not (args.right_wrap and left_wrap), "cannot be both right_wrap and left_wrap"


print("pulling_data_nc_file", args.pulling_data_nc_file)
print("other_pulling_data_nc_files", args.other_pulling_data_nc_files)
print("pmf_nbins", args.pmf_nbins)

print("estimators", args.estimators)

print("system_type", args.system_type)
print("protocol_type", args.protocol_type)

print("right_wrap", args.right_wrap)
print("left_wrap", args.left_wrap)


def _pmf_bin_edges(pulling_files, nbins, symmetric_center):
    trajs = []
    for f in pulling_files:
        data = nc.Dataset(f, "r")
        zF_t = data.variables["zF_t"]
        zR_t = data.variables["zR_t"]
        trajs.append(zF_t)
        trajs.append(zR_t)
    return equal_spaced_bins(trajs, nbins, symmetric_center=symmetric_center)



def num_fe(pulling_data, system_type):
    if system_type == "symmetric":
        U = U_sym
    elif system_type == "asymmetric":
        U = U_asym

    ks = pulling_data["ks"]
    dt = pulling_data["dt"]
    lambda_F = pulling_data["lambda_F"]

    num_df_t = numerical_df_t(U, ks, lambda_F, limit=5.)

    free_energies = {}
    free_energies["lambdas"] = lambda_F
    free_energies["pulling_times"] = np.arange(len(lambda_F))*dt
    free_energies["fes"] = num_df_t

    return free_energies


def _exact_pmf(system_type, pmf_bin_edges):
    if system_type == "symmetric":
        U0 = U0_sym
    elif system_type == "asymmetric":
        U0 = U0_asym

    centers = bin_centers(pmf_bin_edges)
    exact_pmf = U0(centers)

    pmfs = {}
    pmfs["pmf_bin_edges"] = pmf_bin_edges        
    pmfs["pmfs"] = exact_pmf

    return pmfs


def _replicate_fe(fes, side):
    assert side in ["left", "right"], "side must be either left or right"
    if side == "right":
        replicate_func = right_replicate_fe
    else:
        replicate_func = left_replicate_fe

    fes["mean"] = replicate_func(fes["mean"])
    fes["std"] = replicate_func(fes["std"])
    fes["lambdas"] = replicate_func(fes["lambdas"])

    for repeat in fes["fes"]:
        fes["fes"][repeat] = replicate_func(fes["fes"][repeat])
    return None


def _replicate_pmf(ps, side):
    assert side in ["left", "right"], "side must be either left or right"

    if side == "right":
        replicate_edges = right_replicate_bin_edges
        replicate_pmf = right_replicate_pmf
    else:
        replicate_edges = left_replicate_bin_edges
        replicate_pmf = left_replicate_pmf

    ps["pmf_bin_edges"] = replicate_edges(ps["pmf_bin_edges"])
    ps["mean"] = replicate_pmf(ps["mean"])
    ps["std"] = replicate_pmf(ps["std"])

    for repeat in ps["pmfs"]:
        ps["pmfs"][repeat] = replicate_pmf(ps["pmfs"][repeat])
    return None


# -------------
if args.symmetric_center == -999:
    symmetric_center = None
else:
    symmetric_center = args.symmetric_center

if args.system_type == "symmetric":
    assert symmetric_center is not None, "symmetric_center must be not None"
print("symmetric_center", symmetric_center)

pulling_data = load_1d_sim_results(args.pulling_data_nc_file)
estimators = args.estimators.split()

# when system is symm and protocol is asymm, wrap z
if args.system_type == "symmetric" and args.protocol_type == "asymmetric":
    if args.right_wrap:
        pulling_data["zF_t"] = right_wrap(pulling_data["zF_t"], symmetric_center)
        pulling_data["zR_t"] = right_wrap(pulling_data["zR_t"], symmetric_center)

    if args.left_wrap:
        pulling_data["zF_t"] = left_wrap(pulling_data["zF_t"], symmetric_center)
        pulling_data["zR_t"] = left_wrap(pulling_data["zR_t"], symmetric_center)


pulling_files = args.other_pulling_data_nc_files.split() + [args.pulling_data_nc_file]


pmf_bin_edges = _pmf_bin_edges(pulling_files, args.pmf_nbins, symmetric_center)
print("pmf_bin_edges", pmf_bin_edges)

num_free_energies = num_fe(pulling_data, args.system_type)
exact_pmf = _exact_pmf(args.system_type, pmf_bin_edges)

pickle.dump(num_free_energies, open(args.fe_out_prefix + "_numerical" + ".pkl", "w"))
pickle.dump(exact_pmf, open(args.pmf_out_prefix + "_exact" + ".pkl", "w"))

# when both system and protocol are symmetric, use symmetrize_pmf=True, in the s1, s2 estimators
if args.system_type == "symmetric" and args.protocol_type == "symmetric":
    symmetrize_pmf = True
else:
    symmetrize_pmf = True

print("symmetrize_pmf", symmetrize_pmf)


for estimator in estimators:

    if args.system_type == "symmetric" and args.protocol_type == "asymmetric":

        if args.right_wrap:
            half_pmf_bin_edges = pmf_bin_edges[ : args.pmf_nbins/2 + 1 ]
        elif args.left_wrap:
            half_pmf_bin_edges = pmf_bin_edges[args.pmf_nbins/2 : ]
        print("half_pmf_bin_edges", half_pmf_bin_edges)

        fes, ps = pull_fe_pmf(estimator, pulling_data, half_pmf_bin_edges, symmetrize_pmf, V)

        print("replicat final results")
        if args.right_wrap:
            _replicate_fe(fes, "right")
            _replicate_pmf(ps, "right")
        else:
            _replicate_fe(fes, "left")
            _replicate_pmf(ps, "left")

    else:
        fes, ps = pull_fe_pmf(estimator, pulling_data, pmf_bin_edges, symmetrize_pmf, V)

    pickle.dump(fes, open(args.fe_out_prefix + "_" + estimator + ".pkl", "w"))
    pickle.dump(ps, open(args.pmf_out_prefix + "_" + estimator + ".pkl", "w"))

print("DONE")

