"""
To extract lambda values from pulling simulations for US simulations
"""

import argparse

import numpy as np
import netCDF4 as nc

parser = argparse.ArgumentParser()

parser.add_argument("--pulling_nc_file", type=str, default="out.nc")
parser.add_argument("--nsamples", type=int, default=100)
parser.add_argument( "--is_system_or_protocol_symmetric", type=str, default="system")
parser.add_argument( "--out", type=str, default="lambdas.dat")

args = parser.parse_args()

assert args.is_system_or_protocol_symmetric in ["system", "protocol"]


def _equal_stride_subsample(data, nsamples):
    assert nsamples < len(data), "nsamples must less than len of data"
    new_index = np.linspace(0, len(data)-1, nsamples, dtype=int)
    return data[new_index]


def _load_lambdas(nc_file, is_system_or_protocol_symmetric):
    with nc.Dataset(nc_file, "r") as handle:
        lambdas = handle.variables["lambda_F"][:]

    if is_system_or_protocol_symmetric == "system":
        return lambdas

    elif is_system_or_protocol_symmetric == "protocol":
        half = lambdas.shape[0]/2
        return lambdas[:half+1]


def _extract_lambdas(nc_file, nsamples, is_system_or_protocol_symmetric):
    lambdas = _load_lambdas(nc_file, is_system_or_protocol_symmetric)
    extracted_lambdas = _equal_stride_subsample(lambdas, nsamples)
    return extracted_lambdas


if __name__ == "__main__":
    extracted_lambdas = _extract_lambdas(args.pulling_nc_file, args.nsamples, args.is_system_or_protocol_symmetric)
    extracted_lambdas *= 10 # nm to Angstrom

    with open(args.out, "w") as handle:
        handle.write("#       lambda (Angstrom)\n")
        for l in extracted_lambdas:
            handle.write("%30.20f\n" % (l))