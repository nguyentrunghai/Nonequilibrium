"""
calculate pmf for gA
"""

from __future__ import print_function

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--work_data_file", type=str, default="work.nc")

args = parser.parse_args()