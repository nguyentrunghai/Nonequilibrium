"""
estimate free energy differences and PMF using unidirectional estimator for pulling simulation data
"""

from __future__ import print_function

import argparse
import pickle

import numpy as np

from _IO import load_1d_sim_results


from models_1d import U0_sym, U0_asym, V, U_sym, U_asym, numerical_df_t
from utils import equal_spaced_bins, bin_centers
from _fe_pmf import pull_fe_pmf


parser = argparse.ArgumentParser()

parser.add_argument( "--pulling_data_nc_file",      type=str, default="1d_simulation_results_100repeats.nc")

parser.add_argument( "--other_pulling_data_nc_files",      type=str, default="")   # to determine pmf bin
parser.add_argument( "--pmf_nbins",                 type=int, default=25)


parser.add_argument( "--system_type",    type=str,   default="asymmetric")       #  # symmetric or asymmetric
# for symmetric system, asymmetric protocol means pulling only one half of the pmf
parser.add_argument( "--protocol_type",  type=str,   default="asymmetric")   # symmetric or asymmetric

parser.add_argument( "--estimators",                type=str, default="u b s1 s2")

parser.add_argument( "--fe_out_prefix",             type=str, default="fe")
parser.add_argument( "--pmf_out_prefix",            type=str, default="pmf")

args = parser.parse_args()


print("pulling_data_nc_file", args.pulling_data_nc_file)
print("pmf_nbins", args.pmf_nbins)
print("estimators", args.estimators)
print("system_type", args.system_type)


def _pmf_bin_edges(pulling_files, nbins):
    trajs = []
    for f in pulling_files:
        data = nc.Dataset(f, "r")
        zF_t = data.variables["zF_t"]
        zR_t = data.variables["zR_t"]
        trajs.append(zF_t)
        trajs.append(zR_t)
    return equal_spaced_bins(trajs, nbins, symmetric_center=0)


#def _pmf_bin_edges(pulling_data, nbins):
#    zF_t = pulling_data["zF_t"]
#    zR_t = pulling_data["zR_t"]
#    return equal_spaced_bins([zF_t, zR_t], nbins, symmetric_center=0)


def num_fe(pulling_data, is_system_symmetric):
    
    if is_system_symmetric:
        U = U_sym
    else:
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


def _exact_pmf(is_system_symmetric, pmf_bin_edges):

    if is_system_symmetric:
        U0 = U0_sym
    else:
        U0 = U0_asym

    centers = bin_centers(pmf_bin_edges)
    exact_pmf = U0(centers)

    pmfs = {}
    pmfs["pmf_bin_edges"] = pmf_bin_edges        
    pmfs["pmfs"] = exact_pmf

    return pmfs

# -------------

pulling_data = load_1d_sim_results(args.pulling_data_nc_file)
estimators = args.estimators.split()

pulling_files = args.other_pulling_data_nc_files.split() + [args.pulling_data_nc_file]
pmf_bin_edges = _pmf_bin_edges(pulling_files, args.pmf_nbins)
print("pmf_bin_edges", pmf_bin_edges)

num_free_energies = num_fe(pulling_data, args.is_system_symmetric)
exact_pmf = _exact_pmf(args.is_system_symmetric, pmf_bin_edges)

pickle.dump(num_free_energies, open(args.fe_out_prefix + "_numerical" + ".pkl", "w"))
pickle.dump(exact_pmf, open(args.pmf_out_prefix + "_exact" + ".pkl", "w"))

for estimator in estimators:
    fes, ps = pull_fe_pmf(estimator, pulling_data, pmf_bin_edges, args.is_system_symmetric, V)

    pickle.dump(fes, open(args.fe_out_prefix + "_" + estimator + ".pkl", "w"))
    pickle.dump(ps, open(args.pmf_out_prefix + "_" + estimator + ".pkl", "w"))

print("DONE")

