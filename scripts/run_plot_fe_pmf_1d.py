"""
"""
from __future__ import print_function

import argparse
import pickle
import os

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="simulation")

parser.add_argument("--free_energies_files", type=str, default="fe1 fe2")
parser.add_argument("--num_fe_file", type=str, default="num")

parser.add_argument("--pmfs_files", type=str, default="pmf1 pmf2")
parser.add_argument("--exact_pmf_file", type=str, default="exact")

parser.add_argument("--data_estimator_pairs", type=str, default="s_u s_b s_s f_u r_u fr_b")

args = parser.parse_args()

free_energies_files = [os.path.join(args.data_dir, f) for f in args.free_energies_files.split()]
num_fe_file = os.path.join(args.data_dir, args.num_fe_file)

pmfs_files = [os.path.join(args.data_dir, f) for f in args.pmfs_files.split()]
exact_pmf_file = os.path.join(args.data_dir, args.exact_pmf_file)