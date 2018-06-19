"""
"""

from __future__ import print_function

import os
import argparse
import numpy as np
import netCDF4 as nc

from _IO import save_to_nc
from _work import extract_multiple_nc

parser = argparse.ArgumentParser()

parser.add_argument("--forward_pull_dir", type=str, default="1nm_per_ns")
parser.add_argument("--backward_pull_dir", type=str, default="None")

parser.add_argument("--range", type=str,  default="0 1")
parser.add_argument("--exclude", type=str,  default=" ")

parser.add_argument("--ntrajs_per_block", type=int,  default=10)

parser.add_argument("--time_stride", type=int,  default=100)

parser.add_argument("--out", type=str,  default="pull_data.nc")
args = parser.parse_args()

NC_DATA_FILE = "out.nc"
RUN_PREFIX = "conf_"
DT = 0.001
TIME_STRIDE = args.time_stride

#------------------

start = int(args.range.split()[0])
end = int(args.range.split()[1])
exclude = [int(s) for s in args.exclude.split()]
print("exclude", exclude)

indices_to_collect = [i for i in range(start, end) if i not in exclude]

forward_data_files = [os.path.join(args.forward_pull_dir, RUN_PREFIX + "%d"%i, NC_DATA_FILE)
                      for i in indices_to_collect]
for file_name in forward_data_files:
    if not os.path.exists(file_name):
        raise Exception(file_name + " does not exist.")

if args.backward_pull_dir.lower() != "none":
    backward_data_files = [os.path.join(args.backward_pull_dir, RUN_PREFIX + "%d"%i, NC_DATA_FILE)
                           for i in indices_to_collect]
else:
    backward_data_files = []

for file_name in backward_data_files:
    if not os.path.exists(file_name):
        raise Exception(file_name + " does not exist.")


nrepeats = len(forward_data_files) / args.ntrajs_per_block
ntotal_trajs = nrepeats * args.ntrajs_per_block
forward_data_files = forward_data_files[:ntotal_trajs]
backward_data_files = backward_data_files[:ntotal_trajs]

out_nc_handle = nc.Dataset(args.out, "w", format="NETCDF4")

k, lambda_t, z_t, w_t, indices = extract_multiple_nc(forward_data_files, TIME_STRIDE, nrepeats, args.ntrajs_per_block)
data = {"dt" : np.array([DT], dtype=float),
        "pulling_times" : DT*indices,
        "ks" : np.array([k]),
        "lambda_F" : lambda_t,
        "wF_t" : w_t,
        "zF_t" : z_t}
save_to_nc(data, out_nc_handle)


if len(backward_data_files) == 0:
    data = {"lambda_R": lambda_t[::-1], "wR_t": w_t[::-1, ::-1, :], "zR_t": z_t[::-1, ::-1, :]}
else:
    k, lambda_t, z_t, w_t, indices = extract_multiple_nc(backward_data_files, TIME_STRIDE, nrepeats, args.ntrajs_per_block)
    data = {"lambda_R": lambda_t, "wR_t": w_t, "zR_t": z_t}

save_to_nc(data, out_nc_handle)
out_nc_handle.close()

print("DONE")

