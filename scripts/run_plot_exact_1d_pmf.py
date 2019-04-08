
from __future__ import division
from __future__ import print_function

import numpy as np
from utils import bin_centers
from models_1d import U0_sym, U0_asym


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
