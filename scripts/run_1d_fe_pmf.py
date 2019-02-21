"""
TODO:
if lambda_F[timeseries_indices] gives the same elements (but in opposite direction) as
lambda_R[timeseries_indices]

TODO: Check whether we need symmetrize_pmf
"""
from __future__ import print_function
from __future__ import division

import argparse
import pickle

import netCDF4 as nc
import numpy as np

from _IO import load_1d_sim_results
from utils import stride_lambda_indices
from utils import equal_spaced_bins
from utils import bin_centers
from utils import right_wrap, left_wrap

from models_1d import U0_sym, U0_asym, V, U_sym, U_asym, numerical_df_t

from _fe_pmf import unidirectional_fe, unidirectional_pmf
from _fe_pmf import bidirectional_fe, bidirectional_pmf
from _fe_pmf import symmetric_fe, symmetric_pmf

parser = argparse.ArgumentParser()

parser.add_argument( "--pulling_data_nc_file", type=str, default="../../simulation/m1.5_to_p1.5/work_100rep_200traj.nc")

# to determine pmf bin
parser.add_argument( "--other_pulling_data_nc_files", type=str, default="../../simulation/m1.5_to_0/work_100rep_400traj.nc")

# number of data points to take for free energy calculations
parser.add_argument( "--nfe_points", type=int, default=21)

# number of bins for the PMF
parser.add_argument( "--pmf_nbins", type=int, default=20)

# some number or -999 (means None)
parser.add_argument( "--symmetric_center",   type=float, default=0)

# which side to wrap z, left, right or none
parser.add_argument( "--side_to_wrap_z",   type=str, default="none")

# symmetric or asymmetric
parser.add_argument( "--system_type", type=str, default="symmetric")

# for symmetric systems, asymmetric protocol means pulling only one half of the pmf
# symmetric or asymmetric
parser.add_argument( "--protocol_type", type=str, default="symmetric")

# include uf, ur, b, s
parser.add_argument( "--estimators",  type=str, default="uf b s")

# number of blocks of trajectories
parser.add_argument( "--nblocks",  type=int, default=100)

# numbers of trajectories
parser.add_argument( "--ntrajs_per_block",  type=str, default="50 100 150 200")

# number of bootstrap samples
parser.add_argument( "--nbootstraps",  type=int, default=2)

args = parser.parse_args()


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

    free_energy = {}
    free_energy["lambdas"] = lambda_F
    free_energy["pulling_times"] = np.arange(len(lambda_F))*dt
    free_energy["fes"] = num_df_t

    return free_energy


def _exact_pmf(system_type, pmf_bin_edges):
    if system_type == "symmetric":
        U0 = U0_sym
    elif system_type == "asymmetric":
        U0 = U0_asym

    centers = bin_centers(pmf_bin_edges)
    exact_pmf = U0(centers)

    pmf = {}
    pmf["pmf_bin_edges"] = pmf_bin_edges
    pmf["pmfs"] = exact_pmf

    return pmf


if args.system_type not in ["symmetric", "asymmetric"]:
    raise ValueError("unrecognized system_type: " + args.system_type)

if args.protocol_type not in ["symmetric", "asymmetric"]:
    raise ValueError("unrecognized protocol_type: " + args.protocol_type)

if args.side_to_wrap_z not in ["none", "left", "right"]:
    raise ValueError("Unrecognized side_to_wrap_z: " + args.side_to_wrap_z)

estimators = args.estimators.split()
if not all(e in ["uf", "ur", "b", "s"] for e in estimators):
    raise ValueError("unrecognized estimator(s)")

pulling_data = load_1d_sim_results(args.pulling_data_nc_file)

timeseries_indices_F, timeseries_indices_R = stride_lambda_indices(pulling_data["lambda_F"],
                                                                   pulling_data["lambda_R"],
                                                                   args.nfe_points)

ntrajs_list = [int(s) for s in args.ntrajs_per_block.split()]
total_ntrajs_in_data = pulling_data["wF_t"].shape[0]
if np.any(np.array(ntrajs_list) > total_ntrajs_in_data):
    raise ValueError("Number of trajectories requested is larger than available in data (%d)"%total_ntrajs_in_data)

print("timeseries_indices_F", timeseries_indices_F)
print("timeseries_indices_R", timeseries_indices_R)
print("Reduced lambda_F", pulling_data["lambda_F"][timeseries_indices_F])
print("Reduced lambda_R", pulling_data["lambda_R"][timeseries_indices_R])

if args.symmetric_center == -999:
    symmetric_center = None
else:
    symmetric_center = args.symmetric_center

print("symmetric_center", symmetric_center)

pulling_files = [args.pulling_data_nc_file] + [f for f in args.other_pulling_data_nc_files.split()]
print("Data Files to consider for calculating for PMF bins:")
print("\n".join(pulling_files))

pmf_bin_edges = _pmf_bin_edges(pulling_files, args.pmf_nbins, symmetric_center)

# this means pulling is done only half way through
if args.system_type == "symmetric" and args.protocol_type == "asymmetric":
    print("Halve pmf_bin_edges")
    half_len = pmf_bin_edges.shape[0] // 2 + 1
    pmf_bin_edges = pmf_bin_edges[:half_len]

print("pmf_bin_edges", pmf_bin_edges)

if args.system_type == "symmetric" and args.protocol_type == "asymmetric":
    if symmetric_center is None:
        raise ValueError("symmetric_center is None")

    if args.side_to_wrap_z == "right":
        pulling_data["zF_t"] = right_wrap(pulling_data["zF_t"], symmetric_center)
        pulling_data["zR_t"] = right_wrap(pulling_data["zR_t"], symmetric_center)

    elif args.side_to_wrap_z == "left":
        pulling_data["zF_t"] = left_wrap(pulling_data["zF_t"], symmetric_center)
        pulling_data["zR_t"] = left_wrap(pulling_data["zR_t"], symmetric_center)

    else:
        raise ValueError("Really, no wrap?")

# when both system and protocol are symmetric, use symmetrize_pmf=True, in the s1, s2 estimators
if args.system_type == "symmetric" and args.protocol_type == "symmetric":
    symmetrize_pmf = True
else:
    symmetrize_pmf = False

num_free_energies = num_fe(pulling_data, args.system_type)
exact_pmf = _exact_pmf(args.system_type, pmf_bin_edges)

pickle.dump(num_free_energies, open("fe_" + args.protocol_type + "_numerical" + ".pkl", "w"))
pickle.dump(exact_pmf, open("pmf_" + args.protocol_type + "_exact" + ".pkl", "w"))


for ntrajs in ntrajs_list:
    for estimator in estimators:

        if estimator == "uf" or estimator == "ur":
            if estimator == "uf":
                which_data = "F"
                timeseries_indices = timeseries_indices_F
            else:
                which_data = "R"
                timeseries_indices = timeseries_indices_R

            free_energies = unidirectional_fe(pulling_data, args.nblocks, ntrajs,
                                              timeseries_indices,
                                              which_data,
                                              nbootstraps=args.nbootstraps)

            pmfs = unidirectional_pmf(pulling_data, args.nblocks, ntrajs,
                                      which_data,
                                      pmf_bin_edges, V,
                                      nbootstraps=args.nbootstraps)

        elif estimator == "b":
            free_energies = bidirectional_fe(pulling_data, args.nblocks, ntrajs,
                                             timeseries_indices_F,
                                             nbootstraps=args.nbootstraps)

            pmfs = bidirectional_pmf(pulling_data, args.nblocks, ntrajs,
                                     pmf_bin_edges, V,
                                     nbootstraps=args.nbootstraps)

        elif estimator == "s":
            free_energies = symmetric_fe(pulling_data, args.nblocks, ntrajs,
                                         timeseries_indices_F,
                                         nbootstraps=args.nbootstraps)

            pmfs = symmetric_pmf(pulling_data, args.nblocks, ntrajs,
                  pmf_bin_edges, V,
                  symmetrize_pmf,
                  nbootstraps=args.nbootstraps)

        out_file = args.protocol_type + "_" + estimator + "_ntrajs_%d.pkl"%ntrajs
        print("Saving " + out_file)
        pickle.dump({"free_energies" : free_energies, "pmfs" : pmfs}, open(out_file, "w"))


