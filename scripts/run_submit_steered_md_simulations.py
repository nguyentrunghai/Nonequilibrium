
from __future__ import print_function

import sys
import os
import argparse

import numpy as np

from _qsub import head_text_4_qsub_script
from steered_md import SMD

parser = argparse.ArgumentParser()

parser.add_argument( "--first_job_index",   type=int, default=0)
parser.add_argument( "--last_job_index",    type=int, default=1)

parser.add_argument( "--initial_pdb_dir",   type=str, default="pdbs")
parser.add_argument( "--in_pdb_prefix",     type=str, default="conf_")

parser.add_argument( "--initial_pdb",       type=str, default="conf.pdb") # needed for the else part
parser.add_argument( "--out_dir",           type=str, default="out") # needed for the else part

parser.add_argument( "--system",            type=str, default="ALA")     # CUC or ALA
parser.add_argument( "--main_force_field",  type=str, default="Amber12SB")    # Gaff or Amber12SB

parser.add_argument( "--guest_index",       type=int, default=0)
parser.add_argument( "--host_index",        type=int, default=1)

parser.add_argument( "--first_lambda",      type=float, default=1.3)    # 1.3 for ALA; -2 or 2 for CUC
parser.add_argument( "--last_lambda",       type=float, default=3.3)    # 3.3 for ALA; 2 or -2 for CUC
parser.add_argument( "--lambda_k",          type=float, default=1500.)

parser.add_argument( "--ref_pdb_for_host_restraint",    type=str, default="min_aligned_x_moved_to_origin.pdb")
parser.add_argument( "--host_restraint_k",  type=float, default=20000.)

parser.add_argument( "--steps",             type=int, default=2*10**6)

parser.add_argument( "--symmetric_process", action="store_true", default=False)

parser.add_argument( "--memory_request",         type=int, default=2048) # mb

parser.add_argument( "--submit",            action="store_true", default=False)

args = parser.parse_args()

RESTRAIN_CARBONS_ONLY = True
DT = 0.001    # ps
TEMPERATURE = 300. # K
THERMOSTAT = "Nose"
POSITION_SAVE_FREQ = 10000

OUT_NC = "out.nc"
FIRST_OUT_PDB = "first.pdb"
LAST_OUT_PDB = "last.pdb"


if args.submit:

    this_script = os.path.abspath(sys.argv[0])
    initial_pdb_dir = os.path.abspath(args.initial_pdb_dir)

    pdb_files = [os.path.join(initial_pdb_dir, args.in_pdb_prefix + "%d.pdb"%i)
                 for i in range(args.first_job_index, args.last_job_index)]

    for f in pdb_files:
        if not os.path.exists(f):
            raise Exception(f + " does not exist")
    pdb_files = [os.path.abspath(f) for f in pdb_files]

    for pdb_file in pdb_files:
        label = os.path.basename(pdb_file)[:-4]
        out_dir = os.path.abspath(label)

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        if args.symmetric_process:
            symmetric_process = " --symmetric_process "
        else:
            symmetric_process = " "

        qsub_file = os.path.join(out_dir, "smd_" + label + ".job")
        log_file = os.path.join(out_dir, "smd_" + label + ".log")

        qsub_script = head_text_4_qsub_script("cpu", log_file, mem=args.memory_request)

        qsub_script += '''
source /home/tnguye46/opt/module/python2.sh
source /home/tnguye46/opt/module/mmtk.sh
date

python ''' + this_script + \
        ''' --system ''' + args.system + \
        ''' --main_force_field ''' + args.main_force_field + \
        ''' --initial_pdb ''' + pdb_file + \
        ''' --guest_index %d ''' % args.guest_index + \
        ''' --host_index %d ''' % args.host_index + \
        ''' --first_lambda %0.10f ''' % args.first_lambda + \
        ''' --last_lambda %0.10f ''' % args.last_lambda + \
        ''' --lambda_k %0.10f''' % args.lambda_k + \
        ''' --ref_pdb_for_host_restraint ''' + args.ref_pdb_for_host_restraint + \
        ''' --host_restraint_k %0.10f ''' % args.host_restraint_k + \
        ''' --steps %d ''' % args.steps + \
        ''' --out_dir ''' + out_dir + \
        symmetric_process + "\ndate"

        open(qsub_file, "w").write(qsub_script)
        print("submitting " + label)
        os.system("qsub %s" % qsub_file)

else:
    molecule_indices = {"guest": args.guest_index, "host": args.host_index}

    out_nc_file = os.path.join(args.out_dir, OUT_NC)
    first_out_pdb = os.path.join(args.out_dir, FIRST_OUT_PDB)
    last_out_pdb = os.path.join(args.out_dir, LAST_OUT_PDB)

    print("system", args.system)
    print("main_force_field", args.main_force_field)
    print("initial_pdb", args.initial_pdb)
    print("molecule_indices", molecule_indices)
    print("first_lambda", args.first_lambda)
    print("last_lambda", args.last_lambda)
    print("lambda_k", args.lambda_k)
    print("ref_pdb_for_host_restraint", args.ref_pdb_for_host_restraint)
    print("host_restraint_k", args.host_restraint_k)
    print("restrain_carbons_only", RESTRAIN_CARBONS_ONLY)
    print("dt", DT)
    print("steps", args.steps)
    print("position_saving_frequency", POSITION_SAVE_FREQ)
    print("temperature", TEMPERATURE)
    print("thermostat", THERMOSTAT)
    print("out_nc_file", out_nc_file)
    print("symmetric_process", args.symmetric_process)

    smd = SMD(system=args.system,
              main_force_field=args.main_force_field,
              initial_pdb=args.initial_pdb,
              molecule_indices=molecule_indices,
              lambda_k=args.lambda_k,
              ref_pdb_for_host_restraint=args.ref_pdb_for_host_restraint,
              restrain_carbons_only=RESTRAIN_CARBONS_ONLY,
              dt=DT,
              temperature=TEMPERATURE,
              thermostat=THERMOSTAT,
              out_nc_file=out_nc_file)

    programmed_lambdas = np.linspace(args.first_lambda, args.last_lambda, args.steps)
    if args.symmetric_process:
        r_lambda = programmed_lambdas[::-1]
        programmed_lambdas = np.append(programmed_lambdas, r_lambda[1:])

    smd.write_pdb(first_out_pdb)
    smd.propagate(programmed_lambdas, position_save_freq=POSITION_SAVE_FREQ)
    smd.write_pdb(last_out_pdb)

    print("DONE")
