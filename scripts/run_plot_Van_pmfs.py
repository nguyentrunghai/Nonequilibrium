"""
"""

from __future__ import print_function
from __future__ import division

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./")
parser.add_argument("--data_files", type=str,
                    default="unidirectional.dat bidirectional.dat s1.dat pmf.dat us_pmf.dat")

parser.add_argument("--labels", type=str, default="u b s b+WHAM us")
parser.add_argument("--colors", type=str, default="blue green red cyan black")
parser.add_argument("--markers", type=str, default="< > ^ v .")

parser.add_argument("--units", type=str, default="kt kt kt kcal_per_mol kcal_per_mol")

parser.add_argument("--xlabel", type=str, default="$z$ (nm)")
parser.add_argument("--ylabel", type=str, default="$\Phi(z)$ (kcal/mol)")

parser.add_argument("--xlimits", type=str, default="None")
parser.add_argument("--ylimits", type=str, default="None")

parser.add_argument("--pmf_out", type=str, default="pmf.pdf")

args = parser.parse_args()