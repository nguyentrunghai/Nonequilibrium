
from __future__ import print_function

import argparse

from md import MD

parser = argparse.ArgumentParser()
parser.add_argument( "--system",                        type=str, default="ALA")
parser.add_argument( "--main_force_field",              type=str, default="Amber12SB")
parser.add_argument( "--initial_pdb",                   type=str, default="in.pdb")

parser.add_argument( "--guest_index",                   type=int, default=0)
parser.add_argument( "--host_index",                    type=int, default=1)

parser.add_argument( "--initial_lambda",                type=float, default=0.)
parser.add_argument( "--lambda_k",                      type=float, default=1500.)

parser.add_argument( "--ref_pdb_for_host_restraint",    type=str,
                     default="pre_coord/cuc/min_aligned_x_moved_to_origin.pdb")

parser.add_argument( "--host_restraint_k",              type=float, default=20000.)

parser.add_argument( "--equilibrate_steps",       type=int, default=1000)

parser.add_argument( "--steps_per_iteration",       type=int, default=1000)
parser.add_argument( "--niterations",               type=int, default=10)

parser.add_argument( "--out_prefix",                type=str, default="conf_")

args = parser.parse_args()

assert args.guest_index != args.host_index, "guest and host cannot be the same"

RESTRAIN_CARBONS_ONLY = True
DT = 0.001    # ps
TEMPERATURE = 300. # K
THERMOSTAT = "Nose"

print("system = ", args.system)
print("main_force_field = ", args.main_force_field)
print("initial_pdb = ", args.initial_pdb)
print("guest_index = ", args.guest_index)
print("host_index = ", args.host_index)

print("initial_lambda = ", args.initial_lambda)
print("lambda_k = ", args.lambda_k)

print("ref_pdb_for_host_restraint = ", args.ref_pdb_for_host_restraint)
print("host_restraint_k = ", args.host_restraint_k)
print("restrain_host_carbons_only = ", RESTRAIN_CARBONS_ONLY)

print("dt = ", DT)
print("temperature = ", TEMPERATURE)

print("thermostat = ", THERMOSTAT)

print("equilibrate_steps = ", args.equilibrate_steps)
print("steps_per_iteration = ", args.steps_per_iteration)
print("niterations = ", args.niterations)
print("out_prefix = ", args.out_prefix)

molecule_indices = {"guest":args.guest_index, "host":args.host_index}

md = MD(system=args.system,
        main_force_field=args.main_force_field,
        initial_pdb=args.initial_pdb,
        molecule_indices=molecule_indices,
        initial_lambda=args.initial_lambda,
        lambda_k=args.lambda_k,
        ref_pdb_for_host_restraint=args.ref_pdb_for_host_restraint,
        host_restraint_k=args.host_restraint_k,
        restrain_host_carbons_only=RESTRAIN_CARBONS_ONLY,
        steps_per_iteration=args.steps_per_iteration,
        dt=DT,
        temperature=TEMPERATURE,
        thermostat=THERMOSTAT,
        out_prefix=args.out_prefix)

md.minimize()
md.equilibrate(steps=args.equilibrate_steps)
md.propagate(niterations=args.niterations)
print("DONE")


