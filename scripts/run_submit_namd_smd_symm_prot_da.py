"""

"""
from __future__ import print_function

import os
import argparse

from _namd import write_namd_conf_smd_da
from _qsub import head_text_4_qsub_script

parser = argparse.ArgumentParser()

parser.add_argument( "--first_job_index",   type=int, default=0)
parser.add_argument( "--last_job_index",    type=int, default=5)

parser.add_argument( "--initial_pdb_dir",   type=str, default="equilibrate_at_13A")
parser.add_argument( "--in_pdb_prefix",     type=str, default="conf_")

parser.add_argument( "--structure_file",  type=str, default="da.psf")
parser.add_argument( "--parm_file",       type=str, default="par_all27_prot_lipid_cmap.prm")

parser.add_argument( "--forward_tcl",       type=str, default="/home/tnguye46/nonequilibrium/ALA_Namd/tcl_force/forward_20A_per_20ps.tcl")
parser.add_argument( "--backward_tcl",      type=str, default="/home/tnguye46/nonequilibrium/ALA_Namd/tcl_force/backward_20A_per_20ps.tcl")

parser.add_argument( "--output_forward",     type=str, default="forward")
parser.add_argument( "--output_backward",    type=str, default="backward")

parser.add_argument( "--temperature",  type=float, default=300)

parser.add_argument( "--steps",          type=int, default=10000)

args = parser.parse_args()

pdb_files = [ (i, os.path.join(args.initial_pdb_dir, args.in_pdb_prefix + "%d.pdb"%i) )
             for i in range(args.first_job_index, args.last_job_index) ]

for i, f in pdb_files:
    if not os.path.exists(f):
        raise Exception(f + " does not exist")

assert os.path.exists(args.structure_file), args.structure_file + " does not exist."
assert os.path.exists(args.parm_file), args.parm_file + " does not exist."
assert os.path.exists(args.forward_tcl), args.forward_tcl + " does not exist."
assert os.path.exists(args.backward_tcl), args.backward_tcl + " does not exist."


for i, pdb_file in pdb_files:
    out_dir = "%d"%i

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_dir = os.path.abspath(out_dir)

    namd_forward = os.path.join(out_dir, args.output_forward + ".namd")

    write_namd_conf_smd_da(pdb_file, args.structure_file, args.parm_file, args.output_forward,
                           args.temperature, args.forward_tcl, args.steps,
                           namd_forward,
                           coordinates_res=None,
                           velocities_res=None)


    namd_backward = os.path.join(out_dir, args.output_backward + ".namd")
    coor_res = os.path.join(out_dir, args.output_forward + ".restart.coor")
    vel_res = os.path.join(out_dir, args.output_forward + ".restart.vel")

    write_namd_conf_smd_da(pdb_file, args.structure_file, args.parm_file, args.output_backward,
                           args.temperature, args.backward_tcl, args.steps,
                           namd_backward,
                           coordinates_res=coor_res,
                           velocities_res=vel_res)

    qsub_log = os.path.join(out_dir, "logfile")

    qsub_text = head_text_4_qsub_script("cpu", qsub_log, mem=2048)
    qsub_text += "\n"
    qsub_text += "module load namd/2.9\n"
    qsub_text += "\n"
    qsub_text += "cd " + out_dir + "\n"
    qsub_text += "\n"

    log_forward = os.path.join(out_dir, args.output_forward + ".log")
    qsub_text += "namd2 " + namd_forward + " > " + log_forward + "\n"
    qsub_text += "echo DONE FORWARD\n"
    qsub_text += "date\n\n"

    log_backward = os.path.join(out_dir, args.output_backward + ".log")
    qsub_text += "namd2 " + namd_backward + " > " + log_backward + "\n"
    qsub_text += "echo DONE BACKWARD\n"
    qsub_text += "date\n"

    qsub_script = os.path.join(out_dir, "smd.job")
    open(qsub_script, "w").write(qsub_text)
    print("submitting " + qsub_script)
    os.system("qsub %s" % qsub_script)


