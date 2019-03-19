"""
"""
from __future__ import print_function
from __future__ import division

import argparse
import pickle
import copy

import numpy as np

from _IO import load_1d_sim_results
from utils import right_wrap, left_wrap
from utils import closest_sub_array, indices_F_to_R

from _fe_pmf import unidirectional_fe, unidirectional_pmf
from _fe_pmf import bidirectional_fe, bidirectional_pmf
from _fe_pmf import symmetric_fe, symmetric_pmf

parser = argparse.ArgumentParser()

parser.add_argument( "--pulling_data_nc_file", type=str, default="work.nc")
# to read lambdas from umbrella sampling
parser.add_argument("--us_fe_file", type=str, default="us_fe.pkl")

# lower and upper edges are taken from pmf_min_max/
parser.add_argument( "--pmf_lower_edge",            type=float, default=1)
parser.add_argument( "--pmf_upper_edge",            type=float, default=2)
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
parser.add_argument( "--nbootstraps",  type=int, default=10)

args = parser.parse_args()


def _pmf_bin_edges(lower, upper, nbins, symmetric_center):

    if symmetric_center is not None:
        assert lower < symmetric_center < upper, "symmetric_center is not in between [min, max]"

        left_interval = symmetric_center - lower
        right_interval = upper - symmetric_center

        interval = np.max([left_interval, right_interval])

        lower = symmetric_center - interval
        upper = symmetric_center + interval

    bin_edges = np.linspace(lower, upper, nbins + 1)

    return bin_edges

def _time_series_indices(lambda_F, lambda_R, us_fe_file, system_type, protocol_type):
    us_lambdas = pickle.load(open(us_fe_file, "r"))["lambdas"]

    if (system_type == "symmetric") and (protocol_type == "asymmetric"):
        # cut us_lambdas half
        us_lambdas = us_lambdas[: us_lambdas.shape[0]//2]

    if (system_type == "asymmetric") and (protocol_type == "symmetric"):
        # double us_lambdas
        us_lambdas = np.concatenate([us_lambdas[:-1], us_lambdas[::-1]])

    indices_F = closest_sub_array(lambda_F, us_lambdas, threshold=1e-3)
    print("indices_F", indices_F)
    print("lambda_F[indices_F]", lambda_F[indices_F])
    print("us_lambdas", us_lambdas)

    indices_R = indices_F_to_R(indices_F, lambda_F, lambda_R)
    return indices_F, indices_R


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

indices_F, indices_R = _time_series_indices(pulling_data["lambda_F"], pulling_data["lambda_R"],
                                            args.us_fe_file,
                                            args.system_type,
                                            args.protocol_type)

ntrajs_list = [int(s) for s in args.ntrajs_per_block.split()]
total_ntrajs_in_data = pulling_data["wF_t"].shape[0]
if np.any(np.array(ntrajs_list) > total_ntrajs_in_data):
    raise ValueError("Number of trajectories requested is larger than available in data (%d)"%total_ntrajs_in_data)


if args.symmetric_center == -999:
    symmetric_center = None
else:
    symmetric_center = args.symmetric_center

print("symmetric_center", symmetric_center)

pmf_bin_edges = _pmf_bin_edges(args.pmf_lower_edge, args.pmf_lower_edge, args.pmf_nbins, symmetric_center)

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

num_free_energies = num_fe(pulling_data, args.system_type, timeseries_indices_F)
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
        else:
            raise ValueError("Unrecognized estimator")

        out_file = args.protocol_type + "_" + estimator + "_ntrajs_%d.pkl"%ntrajs
        print("Saving " + out_file)
        pickle.dump({"free_energies" : free_energies, "pmfs" : pmfs}, open(out_file, "w"))


