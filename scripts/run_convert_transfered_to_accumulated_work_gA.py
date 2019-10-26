"""
to convert transfered to accumulated work for gA system
"""

import argparse

import netCDF4 as nc

from models_1d import V
from _IO import save_to_nc

parser = argparse.ArgumentParser()
parser.add_argument("--work_in_file", type=str, default="work.nc")

parser.add_argument("--work_out_file", type=str, default="accumulated_work.nc")
args = parser.parse_args()


def _convert_2_acc_work(w_t, z_t, lambda_t, k):
    """
    use Eq. (31) in Hummer and Szabo 2005
    :param w_t: 2d array of shape (ntrajs, times)
    :param z_t: 2d array of shape (ntrajs, times)
    :param lambda_t: 1d array of shape (times,)
    :param k: float, pulling force constant
    :return: acc_work
    """
    v_t = V(z_t, k, lambda_t)       # v_t has shape (ntrajs, times)
    v_0 = v_t[:, [0]]               # v_0 has shape (ntrajs, 1)
    acc_w_t = w_t + v_t - v_0
    return acc_w_t


if args.work_in_file == args.work_out_file:
    raise ValueError("in and out files are the same")

with nc.Dataset(args.work_in_file, "r") as handle:
    data = {key: handle.variables[key][:] for key in handle.variables.keys()}

ks = data["ks"][0]   # kcal/mol/A^2quit

data["wF_t"] = _convert_2_acc_work(data["wF_t"], data["zF_t"], data["lambda_F"], ks)
data["wR_t"] = _convert_2_acc_work(data["wR_t"], data["zR_t"], data["lambda_R"], ks)

with nc.Dataset(args.work_out_file, "w", format="NETCDF4") as handle:
    save_to_nc(data, handle)

print("DONE")
