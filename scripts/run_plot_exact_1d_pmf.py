
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
from utils import bin_centers
from models_1d import U0_sym, U0_asym

from _plots import plot_lines

parser = argparse.ArgumentParser()

# symmetric or asymmetric
parser.add_argument( "--system_type", type=str, default="symmetric")

parser.add_argument("--left_most_edge", type=float, default=-1.5)
parser.add_argument("--right_most_edge", type=float, default=1.5)

# number of bins for the PMF
parser.add_argument("--pmf_nbins", type=int, default=40)

parser.add_argument("--xlabel", type=str, default="$z$")
parser.add_argument("--ylabel", type=str, default="$V(z)$")

parser.add_argument("--out", type=str, default="exact_pmf.pdf")

args = parser.parse_args()


def _exact_pmf(system_type, pmf_bin_edges):
    if system_type == "symmetric":
        U0 = U0_sym
    elif system_type == "asymmetric":
        U0 = U0_asym
    else:
        raise ValueError("Unknown system_type: " + system_type)

    centers = bin_centers(pmf_bin_edges)
    exact_pmf = U0(centers)

    return centers, exact_pmf


pmf_bin_edges = np.linspace(args.left_most_edge, args.right_most_edge, args.pmf_nbins + 1)
bin_centers, exact_pmf = _exact_pmf(args.system_type, pmf_bin_edges)

plot_lines([bin_centers], [pmf_bin_edges],
           xlabel=args.xlabel, ylabel=args.ylabel,
           out=args.out,
           lw=1.0)

print("DONE")


