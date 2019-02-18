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

# symmetric or asymmetric
parser.add_argument( "--system_type",    type=str,   default="asymmetric")

# for asymmetric systems, this option will make the protocol symmetric
# by reversing the motion of lambda back to the initial position
parser.add_argument( "--whether_symmetrize_protocol",  action="store_true", default=False)

# number of repeats
parser.add_argument( "--repeats",               type=int,   default=10)
# number of trajectories per repeat
parser.add_argument( "--trajs_per_repeat",      type=int,   default=1000)

# number of simulation steps per trajectory
# this should be chosen carefully such that the middle value to be exactly
# (initital_lambda + final_lambda) / 2
parser.add_argument( "--steps_per_trajectory",              type=int,   default=751)
# number of steps to be discarded before collecting data
parser.add_argument( "--equilibration_steps",               type=int,   default=250)

# time step
parser.add_argument( "--dt",                                type=float, default=0.001)
# harmonic force constant
parser.add_argument( "--ks",                                type=float, default=15.)
# initial value of the protocol parameter lambda
parser.add_argument( "--initital_lambda",                   type=float, default=-1.5)
# final value of the protocol parameter lambda
# note that if --whether_symmetrize_protocol is used, the values of lambda will go
# from initital_lambda to final_lambda and back to initital_lambda
parser.add_argument( "--final_lambda",                      type=float, default=1.5)

# name of netcdf file to store the output data
parser.add_argument( "--out",                       type=str,   default="work.nc")

args = parser.parse_args()


assert args.system_type in ["symmetric", "asymmetric"], "unknown system_type"

if args.system_type == "symmetric":
    from models_1d import U_sym as U
    from models_1d import dU_dx_sym as dU_dx

elif args.system_type == "asymmetric":
    from models_1d import U_asym as U
    from models_1d import dU_dx_asym as dU_dx


lambda_F = np.linspace(args.initital_lambda, args.final_lambda,      args.steps_per_trajectory)
lambda_R = np.linspace(args.final_lambda,    args.initital_lambda,   args.steps_per_trajectory)

if args.whether_symmetrize_protocol:
    lambda_F = symmetrize_lambda(lambda_F)
    lambda_R = lambda_F

ntimesteps = lambda_F.shape[0]

#zF_t = np.zeros([args.repeats, args.trajs_per_repeat, ntimesteps], dtype=float)
zF_t = np.zeros([args.repeats * args.trajs_per_repeat, ntimesteps], dtype=float)
#wF_t = np.zeros([args.repeats, args.trajs_per_repeat, ntimesteps], dtype=float)
wF_t = np.zeros([args.repeats * args.trajs_per_repeat, ntimesteps], dtype=float)

#zR_t = np.zeros([args.repeats, args.trajs_per_repeat, ntimesteps], dtype=float)
zR_t = np.zeros([args.repeats * args.trajs_per_repeat, ntimesteps], dtype=float)
#wR_t = np.zeros([args.repeats, args.trajs_per_repeat, ntimesteps], dtype=float)
wR_t = np.zeros([args.repeats * args.trajs_per_repeat, ntimesteps], dtype=float)

for repeat in range(args.repeats):
    print("Repeat ", repeat)
    #zF_t[repeat, :, :], wF_t[repeat, :, :] = switching(args.ks, lambda_F, args.equilibration_steps,
    #                                                   args.trajs_per_repeat, args.dt, U, dU_dx)
    lower = repeat * args.trajs_per_repeat
    upper = (repeat + 1) * args.trajs_per_repeat

    zF_t[lower : upper, :], wF_t[lower : upper, :] = switching(args.ks, lambda_F, args.equilibration_steps,
                                                               args.trajs_per_repeat, args.dt, U, dU_dx)

    #zR_t[repeat, :, :], wR_t[repeat, :, :] = switching(args.ks, lambda_R, args.equilibration_steps,
    #                                                   args.trajs_per_repeat, args.dt, U, dU_dx)
    zR_t[lower : upper, :], wR_t[lower : upper, :] = switching(args.ks, lambda_R, args.equilibration_steps,
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

