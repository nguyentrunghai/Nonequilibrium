"""
make u_kln from namd simulation data.
u_kln is needed for MBAR
"""
from __future__ import print_function

import os
import argparse
import pickle

import numpy as np
import netCDF4 as nc
import pymbar

from _namd_us import load_namd_potential, load_biased_coordinates, cal_u_kln
from _IO import save_to_nc

parser = argparse.ArgumentParser()

parser.add_argument( "--window_centers_file",   type=str,
                     default="window_centers.dat")

parser.add_argument("--namd_sim_dir", type=str,
                    default="us_md")

parser.add_argument("--namd_logfile", type=str, default="logfile")
parser.add_argument("--force_file", type=str, default="da.force")

parser.add_argument( "--nr_last_samples",   type=int, default=1000)

parser.add_argument( "--force_constant",  type=float, default=7.2)   # kcal/mol/A^2

parser.add_argument("--symmetrized_states", action="store_true", default=False)

parser.add_argument("--u_kln_out", type=str, default="u_kln.nc")

parser.add_argument("--fe_out", type=str, default="fe.pkl")

args = parser.parse_args()

KB = 0.0019872041   # kcal/mol/K
TEMPERATURE = 300.
BETA = 1/KB/TEMPERATURE

us_centers = np.loadtxt(args.window_centers_file)
K = us_centers.shape[0]

coordinates = []
unbiased_potentials = []

for k in range(K):
    namd_log = os.path.join(args.namd_sim_dir, "%d"%k, args.namd_logfile)
    force_file = os.path.join(args.namd_sim_dir, "%d"%k, args.force_file)

    print("loading", namd_log)
    pot_e = load_namd_potential(namd_log)
    pot_e = pot_e[- args.nr_last_samples : ]
    unbiased_potentials.append( pot_e )

    print("loading", force_file)
    coord = load_biased_coordinates(force_file)
    coord = coord[- args.nr_last_samples : ]
    coordinates.append(coord)

coordinates = np.array(coordinates)
unbiased_potentials = np.array(unbiased_potentials)

if args.symmetrized_states:
    print("symmetrized states")
    us_centers = np.concatenate((us_centers, -us_centers))
    coordinates = np.concatenate((coordinates, -coordinates), axis=0)
    unbiased_potentials = np.concatenate((unbiased_potentials, unbiased_potentials), axis=0)

u_kln = cal_u_kln(args.force_constant, us_centers, coordinates, unbiased_potentials)
u_kln *= BETA  # kcal/mol to kT
us_centers /= 10.  # Angstrom to nm

nc_handle = nc.Dataset(args.u_kln_out, "w", format="NETCDF4")
save_to_nc({"u_kln":u_kln, "us_centers":us_centers}, nc_handle)
nc_handle.close()

K = u_kln.shape[0]
N = u_kln.shape[-1]
N_k = np.array([N]*K, dtype=int)

mbar = pymbar.MBAR(u_kln, N_k, verbose=True)
fe = mbar.f_k

if args.symmetrized_states:
    out_data = {"fe": fe[ :K/2 ], "lambdas": us_centers[ :K/2 ]}
else:
    out_data = {"fe":fe, "lambdas":us_centers}

pickle.dump(out_data, open(args.fe_out, "w"))

print("DONE!")
