"""
submit US jobs to the queue
"""

from __future__ import print_function

import sys
import os
import argparse

import numpy as np

from _qsub import head_text_4_qsub_script

parser = argparse.ArgumentParser()

parser.add_argument( "--first_job_index",   type=int, default=0)
parser.add_argument( "--last_job_index",    type=int, default=1)

parser.add_argument( "--system",            type=str, default="CUC")          # CUC or ALA
parser.add_argument( "--main_force_field",  type=str, default="Gaff")    # Gaff or Amber12SB

parser.add_argument( "--in_pdb_dir",        type=str, default="pdbs")
parser.add_argument( "--in_pdb_prefix",     type=str, default="conf_")

parser.add_argument( "--initial_pdb",           type=str, default="init.pdb")

parser.add_argument( "--list_of_lambdas_file",  type=str, default="lambdas.dat")

parser.add_argument( "--lambda_k",              type=float, default=1500.)

parser.add_argument( "--guest_index",                   type=int, default=0)
parser.add_argument( "--host_index",                    type=int, default=1)
parser.add_argument( "--ref_pdb_for_host_restraint",    type=str, default="min_aligned_x_moved_to_origin.pdb")
parser.add_argument( "--host_restraint_k",              type=float, default=20000.)

parser.add_argument( "--steps_per_iteration",   type=int, default=1000)
parser.add_argument( "--equil_iterations",      type=int, default=50)
parser.add_argument( "--prod_iterations",       type=int, default=5000)

parser.add_argument( "--want_exchange",         action="store_true", default=False)

parser.add_argument( "--out_dir",               type=str, default="out")

parser.add_argument( "--memory_request",         type=int, default=2048) # mb

parser.add_argument( "--submit",                action="store_true", default=False)

args = parser.parse_args()

JOB_NAME_PREFIX = "repeat"
RESTRAIN_CARBONS_ONLY = True
TEMPERATURE = 300. # K
THERMOSTAT = "Nose"
DT = 0.001    # ps

MIN_PDB_PREFIX = "min"
EQUIL_PDB_PREFIX = "equil"
FINAL_PDB_PREFIX = "final"
NC_OUT = "us_energies_trajectories.nc"

if args.submit:

    this_script =  os.path.abspath(sys.argv[0])
    in_pdb_dir = os.path.abspath(args.in_pdb_dir)

    nr_submits = 0

    for job_index in range(args.first_job_index, args.last_job_index):

        in_pdb_file = os.path.join(in_pdb_dir, args.in_pdb_prefix + "%d.pdb" % job_index)
        if not os.path.exists(in_pdb_file):
            raise Exception(in_pdb_file + " does not exist")

        out_dir = JOB_NAME_PREFIX + "_%d"%job_index

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        out_dir = os.path.abspath(out_dir)

        want_exchange = " --want_exchange " if args.want_exchange else " "

        qsub_file = os.path.join(out_dir, JOB_NAME_PREFIX + "_%d.job"%job_index)
        log_file  = os.path.join(out_dir, JOB_NAME_PREFIX + "_%d.log"%job_index)

        qsub_script = head_text_4_qsub_script("cpu", log_file, mem=args.memory_request)

        qsub_script += '''
source /home/tnguye46/opt/module/python2.sh
source /home/tnguye46/opt/module/mmtk.sh

date
python ''' + this_script + \
        ''' --system ''' + args.system + \
        ''' --main_force_field ''' + args.main_force_field + \
        ''' --initial_pdb ''' + in_pdb_file + \
        ''' --list_of_lambdas_file ''' + args.list_of_lambdas_file + \
        ''' --lambda_k %0.10f ''' % args.lambda_k + \
        ''' --guest_index %d ''' % args.guest_index + \
        ''' --host_index %d ''' % args.host_index + \
        ''' --ref_pdb_for_host_restraint ''' + args.ref_pdb_for_host_restraint + \
        ''' --host_restraint_k %0.10f ''' % args.host_restraint_k + \
        ''' --steps_per_iteration %d ''' % args.steps_per_iteration + \
        ''' --equil_iterations %d ''' % args.equil_iterations + \
        ''' --prod_iterations %d ''' % args.prod_iterations + \
        ''' --out_dir ''' + out_dir + \
        want_exchange + \
        '''\ndate\n'''

        open(qsub_file, "w").write(qsub_script)
        print("submitting "+qsub_file)
        os.system("qsub %s" %qsub_file)
        nr_submits += 1
    print("Submitted %d jobs"%nr_submits)

else:
    from umbrella_sampling import UmbrellaSampling
    print("system", args.system)
    print("main_force_field", args.main_force_field)
    print("initial_pdb", args.initial_pdb)

    molecule_indices = {"guest":args.guest_index, "host":args.host_index}
    print("molecule_indices", molecule_indices)

    list_of_lambdas = np.loadtxt(args.list_of_lambdas_file)
    list_of_lambdas = list(list_of_lambdas)
    print("list_of_lambdas", list_of_lambdas)

    print("lambda_k", args.lambda_k)
    print("ref_pdb_for_host_restraint", args.ref_pdb_for_host_restraint)
    print("host_restraint_k", args.host_restraint_k)

    print("restrain_host_carbons_only", RESTRAIN_CARBONS_ONLY)
    print("temperature", TEMPERATURE)
    print("thermostat", THERMOSTAT)
    print("time_step", DT)

    print("steps_per_iteration", args.steps_per_iteration)
    print("equil_iterations", args.equil_iterations)
    print("prod_iterations", args.prod_iterations)
    print("want_exchange", str(args.want_exchange))
    print("out_dir", args.out_dir)

    out_nc_file      = os.path.join(args.out_dir, NC_OUT)
    min_pdb_prefix   = os.path.join(args.out_dir, MIN_PDB_PREFIX)
    equil_pdb_prefix = os.path.join(args.out_dir, EQUIL_PDB_PREFIX)
    final_pdb_prefix = os.path.join(args.out_dir, FINAL_PDB_PREFIX)

    sampler = UmbrellaSampling(
                                system=args.system,
                                main_force_field=args.main_force_field,
                                initial_pdbs=args.initial_pdb,
                                molecule_indices=molecule_indices,
                                list_of_lambdas=list_of_lambdas,
                                lambda_k=args.lambda_k,
                                ref_pdb_for_host_restraint=args.ref_pdb_for_host_restraint,
                                host_restraint_k=args.host_restraint_k,
                                restrain_host_carbons_only=RESTRAIN_CARBONS_ONLY,
                                time_step=DT,
                                temperature=TEMPERATURE,
                                thermostat=THERMOSTAT,
                                out_nc_file=out_nc_file)

    sampler.minimize_all_replicas()
    sampler.write_to_pdb(min_pdb_prefix)

    for iteration in range(args.equil_iterations):
        sampler.propagate_all_replicas(args.steps_per_iteration, want_exchange=False, save_2_nc=False)
    sampler.write_to_pdb(equil_pdb_prefix)

    for iteration in range(args.prod_iterations):
        sampler.propagate_all_replicas(args.steps_per_iteration, want_exchange=args.want_exchange, save_2_nc=True)
    sampler.write_to_pdb(final_pdb_prefix)
    sampler.close_nc()

    print("DONE!")


