"""

"""

from __future__ import print_function

import os
import argparse

import numpy as np

from _namd_us import write_tcl_us_cuc, write_namd_conf_us_cuc
from _qsub import head_text_4_qsub_script

parser = argparse.ArgumentParser()

parser.add_argument( "--window_centers_file",   type=str, default="window_centers.dat")
parser.add_argument( "--job_range",   type=str, default="0 None")

parser.add_argument( "--initial_pdb",  type=str, default="cuc7_fb.pdb")

parser.add_argument( "--prmtop", type=str, default="cuc7_fb.prmtop")
parser.add_argument( "--host_fixed_pdb",     type=str, default="cuc7_fb_fixed_host.pdb")

parser.add_argument( "--tcl_force_file",       type=str, default="harmonic_force.tcl")

parser.add_argument( "--force_constant",  type=float, default=7.2)   # kcal/mol/A^2

parser.add_argument( "--namd_output_name",     type=str, default="cuc7")

parser.add_argument( "--temperature",  type=float, default=300.)

parser.add_argument( "--min_steps",      type=int, default=1000)
parser.add_argument( "--steps",          type=int, default=500000) # default 1ns

args = parser.parse_args()

window_centers = np.loadtxt(args.window_centers_file)

job_begin = int(args.job_range.split()[0])
job_end = args.job_range.split()[1]
if job_end.lower() == "none":
    job_end = len(window_centers)
else:
    job_end = int(job_end)

assert job_end > job_begin, "job_end must be greater than job_begin"

for window_id in range(job_begin, job_end):
    window_center = window_centers[window_id]

    out_dir = "%d" % window_id
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_dir = os.path.abspath(out_dir)

    tcl_force_file = os.path.join(out_dir, args.tcl_force_file)
    print("Writing " + tcl_force_file)
    write_tcl_us_cuc(tcl_force_file, window_center, force_constant=args.force_constant,
                    force_out=args.namd_output_name+".force")

    namd_conf_file = os.path.join(out_dir, args.namd_output_name + ".namd")
    print("Writing "+namd_conf_file)
    write_namd_conf_us_cuc(namd_conf_file,
                          args.initial_pdb, args.prmtop, args.host_fixed_pdb,
                          args.namd_output_name,
                          args.temperature,
                          tcl_force_file,
                          args.min_steps,
                          args.steps)

    qsub_log = os.path.join(out_dir, "logfile")
    qsub_text = head_text_4_qsub_script("cpu", qsub_log, mem=2048)
    qsub_text += "\n"

    qsub_text += "module load namd/2.9\n"
    qsub_text += "\n"
    qsub_text += "cd " + out_dir + "\n"
    qsub_text += "\n"

    qsub_text += "date\n"
    qsub_text += "namd2 " + namd_conf_file + "\n"
    qsub_text += "echo DONE\n"
    qsub_text += "date\n"

    qsub_script = os.path.join(out_dir, "cuc7_us.job")
    open(qsub_script, "w").write(qsub_text)
    print("submitting " + qsub_script)
    os.system("qsub %s" % qsub_script)
