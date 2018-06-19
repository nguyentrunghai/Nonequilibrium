"""
estimate free energy differences and PMF using MBAR estimator for umbrella sampling simulation data
"""
from __future__ import print_function

import glob
import pickle
import argparse

import numpy as np
import netCDF4 as nc

import pymbar

from utils import get_bin_indexes, equal_spaced_bins

from _IO import load_1d_sim_results


parser = argparse.ArgumentParser()

parser.add_argument( "--glob_matching_4_us_data_files", type=str, default="repeat_*/us_energies_trajectories.nc" )
parser.add_argument( "--pulling_data_nc_file",   type=str, default="smd_work_data_4800.nc")   # to define common bins

parser.add_argument( "--pmf_nbins",                     type=int, default=25)
parser.add_argument( "--is_system_symmetric",       action="store_true", default=False)

parser.add_argument( "--fe_out_prefix",                 type=str, default="fe_us")
parser.add_argument( "--pmf_out_prefix",                type=str, default="pmf_us")

args = parser.parse_args()

print("pulling_data_nc_file", args.pulling_data_nc_file)
print("glob_matching_4_us_data_files", args.glob_matching_4_us_data_files)
print("pmf_nbins", args.pmf_nbins)
print("is_system_symmetric", args.is_system_symmetric)


def _pmf_bin_edges(pulling_data, nc_us_result_files, nbins, is_system_symmetric):
    """ """
    zF_t = pulling_data["zF_t"]
    zR_t = pulling_data["zR_t"]

    min_max = []
    for loc in nc_us_result_files:
        print("reading ", loc)
        nc_handle = nc.Dataset(loc, "r")
        min_max.append( nc_handle.variables["restrained_coordinates_kn"][:].min() )
        min_max.append( nc_handle.variables["restrained_coordinates_kn"][:].max() )
        nc_handle.close()
    min_max = np.array(min_max)

    if is_system_symmetric:
        return equal_spaced_bins([zF_t, zR_t, min_max], nbins, symmetric_center=0)
    else:
        return equal_spaced_bins([zF_t, zR_t, min_max], nbins, symmetric_center=None)


def _enlarge_u_matrix(u0_kn, u_kln):
    assert u0_kn.shape[0] == u_kln.shape[0], "u0_kn and u_kln must have the same number of sampled states"
    assert u0_kn.shape[-1] == u_kln.shape[-1], "u0_kn and u_kln must have the same N"

    K = u_kln.shape[0]
    N = u_kln.shape[-1]

    N_k = np.ones([K+1], dtype=int) * N
    N_k[0] = 0

    large_u_kln = np.zeros( [ K, K+1, N_k.max() ], dtype=float )
    large_u_kln[:, 1:, :] = u_kln
    large_u_kln[:, 0, :] = u0_kn

    return large_u_kln, N_k


#------------------
nc_us_result_files = glob.glob(args.glob_matching_4_us_data_files)
print("\n".join(nc_us_result_files))

"""
nc_handles = { loc : nc.Dataset(loc, "r") for loc in nc_us_result_files }

list_of_lambdas = {loc : nc_handles[loc].variables["list_of_lambdas"] for loc in nc_us_result_files}

u_kln = {loc : nc_handles[loc].variables["u_kln"] for loc in nc_us_result_files}

u0_kn = {loc : nc_handles[loc].variables["u0_kn"] for loc in nc_us_result_files}

restrained_coordinates_kn = {loc : nc_handles[loc].variables["restrained_coordinates_kn"] for loc in nc_us_result_files}
"""

pulling_data = load_1d_sim_results(args.pulling_data_nc_file)

pmf_bin_edges = _pmf_bin_edges(pulling_data, nc_us_result_files, args.pmf_nbins, args.is_system_symmetric)

print("pmf_bin_edges", pmf_bin_edges)
nbins = len(pmf_bin_edges) - 1

free_energies = {}
free_energies["lambdas"] = nc.Dataset(nc_us_result_files[0], "r").variables["list_of_lambdas"][:]
free_energies["fes"] = {} 

pmfs = {}
pmfs["pmf_bin_edges"] = pmf_bin_edges
pmfs["pmfs"] = {}

for loc in nc_us_result_files:

    # cal fe
    print("cal fe for ", loc)

    u_kln = nc.Dataset(loc, "r").variables["u_kln"][:]
    K = u_kln.shape[0]
    N = u_kln.shape[-1]
    N_k = np.ones([K], dtype=int) * N

    mbar = pymbar.MBAR( u_kln, N_k, verbose=True )
    free_energies["fes"][loc] = np.array(mbar.f_k)


    # calculate PMF
    print("cal pmf for ", loc)
    u0_kn = nc.Dataset(loc, "r").variables["u0_kn"][:]
    
    l_u_kln, l_N_k = _enlarge_u_matrix( u0_kn, u_kln ) 
    mbar = pymbar.MBAR( l_u_kln, l_N_k, verbose=True )

    _u0_kn = l_u_kln[:, 0, :]
    phi = nc.Dataset(loc, "r").variables["restrained_coordinates_kn"][:]
    
    bin_indexes, bin_edges = get_bin_indexes(phi, pmf_bin_edges)
    if np.any(bin_indexes == -1):
        raise Exception("some phi value does not have a bin to be put into")

    pmf_i, _ = mbar.computePMF(_u0_kn, bin_indexes, nbins)

    pmfs["pmfs"][loc] = pmf_i

free_energies["mean"] = np.stack( free_energies["fes"].values() ).mean(axis=0)
free_energies["std"] = np.stack( free_energies["fes"].values() ).std(axis=0)

pmfs["mean"] = np.stack( pmfs["pmfs"].values() ).mean(axis=0)
pmfs["std"] = np.stack( pmfs["pmfs"].values() ).std(axis=0)

"""
for handle in nc_handles.values():
    handle.close()
"""

pickle.dump(free_energies, open(args.fe_out_prefix + ".pkl", "w"))
pickle.dump(pmfs, open(args.pmf_out_prefix + ".pkl", "w"))

