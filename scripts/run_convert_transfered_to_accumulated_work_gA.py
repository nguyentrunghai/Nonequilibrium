"""
to convert transfered to accumulated work for gA system
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--work_in_file", type=str, default="work.nc")

parser.add_argument("--force_constant", type=float, default=100)

parser.add_argument("--work_out_file", type=str, default="accumulated_work.nc")
args = parser.parse_args()

