"""

"""

from __future__ import print_function

import os
import argparse

from _namd import write_namd_conf_smd_cuc

from _qsub import head_text_4_qsub_script

parser = argparse.ArgumentParser()

parser.add_argument( "--first_job_index",   type=int, default=0)
parser.add_argument( "--last_job_index",    type=int, default=10)

parser.add_argument( "--initial_pdb_dir",   type=str, default="at_x-20A")
parser.add_argument( "--in_pdb_prefix",     type=str, default="conf_")

parser.add_argument( "--prmtop",     type=str, default="cuc7_fb.prmtop")
parser.add_argument( "--host_fixed_pdb",     type=str, default="cuc7_fb_fixed_host.pdb")

parser.add_argument( "--tcl_file",       type=str, default="forward.tcl")

parser.add_argument( "--output_name",     type=str, default="test")

parser.add_argument( "--temperature",  type=float, default=300)

parser.add_argument( "--steps",          type=int, default=2000)

args = parser.parse_args()

pdb_files = [ (i, os.path.join(args.initial_pdb_dir, args.in_pdb_prefix + "%d.pdb"%i) )
             for i in range(args.first_job_index, args.last_job_index) ]

pdb_files = [ (i, os.path.abspath(f)) for i, f in pdb_files]

for i, f in pdb_files:
    if not os.path.exists(f):
        raise Exception(f + " does not exist")


for i, pdb_file in pdb_files:
    out_dir = "%d"%i

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_dir = os.path.abspath(out_dir)

    namd_conf = os.path.join(out_dir, args.output_name + ".namd")

    write_namd_conf_smd_cuc(pdb_file, args.prmtop, args.host_fixed_pdb,
                            args.output_name,
                            args.temperature,
                            args.tcl_file,
                            args.steps,
                            namd_conf)

    qsub_log = os.path.join(out_dir, "logfile")

    qsub_text = head_text_4_qsub_script("cpu", qsub_log, mem=2048)
    qsub_text += "\n"
    qsub_text += "module load namd/2.9\n"
    qsub_text += "\n"
    qsub_text += "cd " + out_dir + "\n"
    qsub_text += "\n"

    qsub_text += "namd2 " +  namd_conf + "\n"
    qsub_text += "echo DONE\n"
    qsub_text += "date\n\n"

    qsub_script = os.path.join(out_dir, "smd.job")
    open(qsub_script, "w").write(qsub_text)
    print("submitting " + qsub_script)
    os.system("qsub %s" % qsub_script)
