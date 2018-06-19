"""

"""

from __future__ import print_function

import argparse
import glob

import numpy as np
import netCDF4 as nc


parser = argparse.ArgumentParser()

parser.add_argument("--pull_nc_file_matching", type=str,
                    default="*/pull_data.nc")

parser.add_argument("--us_time_series_file_matching", type=str,
                    default="window_*.dat")
args = parser.parse_args()

pull_data_files = glob.glob(args.pull_nc_file_matching)

mins = []
maxs = []

for nc_file in pull_data_files:
    print("loading "+ nc_file)
    data = nc.Dataset(nc_file)

    mins.append(data.variables["zF_t"][:].min())
    maxs.append(data.variables["zF_t"][:].max())

    mins.append(data.variables["zR_t"][:].min())
    maxs.append(data.variables["zR_t"][:].max())

us_time_series_files = glob.glob(args.us_time_series_file_matching)

for time_s_file in us_time_series_files:
    print("loading " + time_s_file)
    data = np.loadtxt(time_s_file)[:, 1]
    data /= 10.   # Angstrom to nm

    mins.append(data.min())
    maxs.append(data.max())

print("\n\n")
print("nm")
print( "Min = %15.10f,  Max = %15.10f" % (np.min(mins), np.max(maxs)) )

print("\n\n")
print("Angstrom")
print( "Min = %15.10f,  Max = %15.10f" % (np.min(mins)*10., np.max(maxs)*10.) )