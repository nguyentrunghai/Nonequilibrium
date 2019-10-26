"""
to convert transfered to accumulated work for gA system
"""

import argparse

from models_1d import V
from _IO import save_to_nc

parser = argparse.ArgumentParser()
parser.add_argument("--work_in_file", type=str, default="work.nc")

parser.add_argument("--force_constant", type=float, default=100)

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

