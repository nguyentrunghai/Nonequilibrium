"""
run 1d pulling simulations
"""
from __future__ import print_function

import argparse

import numpy as np
import netCDF4 as nc

from models_1d import switching, symmetrize_lambda

from _IO import save_to_nc

parser = argparse.ArgumentParser()

parser.add_argument( "--system_type",    type=str,   default="asymmetric")   # symmetric or asymmetric
parser.add_argument( "--protocol_type",  type=str,   default="asymmetric")   # symmetric or asymmetric

parser.add_argument( "--repeats",               type=int,   default=10)
parser.add_argument( "--trajs_per_repeat",      type=int,   default=1000)

parser.add_argument( "--steps_per_trajectory",              type=int,   default=750)
parser.add_argument( "--equilibration_steps",               type=int,   default=250)

parser.add_argument( "--dt",                                type=float, default=0.001)
parser.add_argument( "--ks",                                type=float, default=15.)
parser.add_argument( "--initital_lambda",                   type=float, default=-1.5)
parser.add_argument( "--final_lambda",                      type=float, default=1.5)

parser.add_argument( "--out",                       type=str,   default="work.nc")


args = parser.parse_args()


assert args.system_type in ["symmetric", "asymmetric"], "unknown system_type"
assert args.protocol_type in ["symmetric", "asymmetric"], "unknown protocol_type"


if args.system_type == "symmetric":
    from models_1d import U_sym as U
    from models_1d import dU_dx_sym as dU_dx

elif args.system_type == "asymmetric":
    from models_1d import U_asym as U
    from models_1d import dU_dx_asym as dU_dx


lambda_F = np.linspace(args.initital_lambda, args.final_lambda,      args.steps_per_trajectory)
lambda_R = np.linspace(args.final_lambda,    args.initital_lambda,   args.steps_per_trajectory)

if args.protocol_type == "symmetric":
    lambda_F = symmetrize_lambda(lambda_F)
    lambda_R = lambda_F

ntimesteps = lambda_F.shape[0]

zF_t = np.zeros([args.repeats, args.trajs_per_repeat, ntimesteps], dtype=float)
wF_t = np.zeros([args.repeats, args.trajs_per_repeat, ntimesteps], dtype=float)

zR_t = np.zeros([args.repeats, args.trajs_per_repeat, ntimesteps], dtype=float)
wR_t = np.zeros([args.repeats, args.trajs_per_repeat, ntimesteps], dtype=float)

for repeat in range(args.repeats):
    print("Repeat ", repeat)
    zF_t[repeat, :, :], wF_t[repeat, :, :] = switching(args.ks, lambda_F, args.equilibration_steps,
                                                       args.trajs_per_repeat, args.dt, U, dU_dx)
    zR_t[repeat, :, :], wR_t[repeat, :, :] = switching(args.ks, lambda_R, args.equilibration_steps,
                                                       args.trajs_per_repeat, args.dt, U, dU_dx)

data = {}
data["lambda_F"] = lambda_F
data["lambda_R"] = lambda_R
data["ks"] = np.array([args.ks], dtype=float)
data["dt"] = np.array([args.dt], dtype=float)

data["zF_t"] = zF_t
data["wF_t"] = wF_t 

data["zR_t"] = zR_t
data["wR_t"] = wR_t 

nc_handle   = nc.Dataset(args.out, "w", format="NETCDF4")
save_to_nc(data, nc_handle)
nc_handle.close()

