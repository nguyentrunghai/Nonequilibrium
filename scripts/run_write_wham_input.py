"""

"""

import argparse
import os

import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument( "--us_dir",   type=str, default="us_md")

parser.add_argument( "--window_centers_file",   type=str, default="window_centers.dat")

parser.add_argument( "--force_constant",  type=float, default=7.2)   # kcal/mol/A^2

parser.add_argument( "--us_force_file",   type=str, default="da.force")

parser.add_argument( "--nr_last_samples",   type=int, default=2000)

parser.add_argument( "--time_series_prefix",  type=str, default="window_")   # kcal/mol/A^2

parser.add_argument( "--meta_file",  type=str, default="wham_meta.dat")

parser.add_argument( "--reflect_thro_zero",  action="store_true", default=False)

args = parser.parse_args()


def write_time_series(us_force_file, nr_last_samples, out, reflect_thro_zero=False):
    """
    :param us_force_file:  str, name of file that stores us harmonic coordinates forces
    :param nr_last_samples: int, the last nr_last_samples samples will be taken from us_force_file
    :param out: str, name of output file
    :param reflect_thro_zero: bool
    :return: None
    """
    data = np.loadtxt(us_force_file)
    assert data.shape[0] >= nr_last_samples, "number of lines in " + us_force_file + "must be greater than nr_last_samples"
    data = data[-nr_last_samples : , 1]

    if reflect_thro_zero:
        data = - data

    with open(out, "w") as handle:
        for i, coor in enumerate(data):
            handle.write( "%10d %20.10f\n" % (i, coor) )
    return data


window_centers = np.loadtxt(args.window_centers_file)
nwidows = window_centers.shape[0]

meta_str = ""
mins = []
maxs = []
for window in range(nwidows):
    us_force_file = os.path.join(args.us_dir, "%d"%window, args.us_force_file)
    print "loading " + us_force_file

    time_series_out = args.time_series_prefix + "%d.dat"%window
    tmp = write_time_series(us_force_file, args.nr_last_samples, time_series_out, reflect_thro_zero=False)

    mins.append(tmp.min())
    maxs.append(tmp.max())

    meta_str += time_series_out + " %20.10f  %20.10f\n" %(window_centers[window], args.force_constant)

    if args.reflect_thro_zero:
        time_series_out = args.time_series_prefix + "replicated_%d.dat" % window
        tmp = write_time_series(us_force_file, args.nr_last_samples, time_series_out, reflect_thro_zero=True)
        mins.append(tmp.min())
        maxs.append(tmp.max())
        meta_str += time_series_out + " %20.10f  %20.10f\n" % (-window_centers[window], args.force_constant)

open(args.meta_file, "w").write(meta_str)

print "DONE"
