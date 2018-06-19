"""

"""

from __future__ import print_function

import os
import pickle
import argparse

import numpy as np

from _plots import plot_lines

parser = argparse.ArgumentParser()

parser.add_argument("--pull_fe_dir", type=str,
                    default="pull_fe_pmf")
parser.add_argument("--pull_estimators", type=str, default="u b s1 s2")

parser.add_argument("--us_fe_file", type=str,
                    default="fe.pkl")

parser.add_argument("--nsamples", type=int, default=20)

parser.add_argument("--is_protocol_symmetric", action="store_true", default=False)

parser.add_argument("--xlimits", type=str, default="None")
parser.add_argument("--ylimits_fe", type=str, default="None")
parser.add_argument("--ylimits_rmse", type=str, default="None")

parser.add_argument("--xlabel", type=str, default="$\lambda$ (nm)")
parser.add_argument("--ylabel_fe", type=str, default="$\Delta F_{\lambda}$ (kcal/mol)")
parser.add_argument("--ylabel_rmse", type=str, default="RMSE[$\Delta F_{\lambda}$] (kcal/mol)")

parser.add_argument("--fe_out", type=str, default="fe.pdf")
parser.add_argument("--rmse_out", type=str, default="fe_rmse.pdf")

args = parser.parse_args()


KB = 0.0019872041   # kcal/mol/K
TEMPERATURE = 300.

FE_FILE_PREFIX = "fe_"

MARKERS = ["<", ">", "^", "v", "o"]

def _shift_min_to_zero(array):
    return array - array.min()


def _equal_stride_subsample(data, nsamples):
    assert nsamples < len(data), "nsamples must less than len of data"
    new_index = np.linspace(0, len(data)-1, nsamples, dtype=int)
    return data[new_index]


def find_index_monotonic(where_to_find, what_to_find):
    indices = []
    for what in what_to_find:
        d = np.abs(where_to_find - what)
        ind = np.argmin(d)
        indices.append(ind)
    return np.array(indices, dtype=int)


def point_of_1st_change_slope(data):
    for i in range(data.shape[0]-1):
        if data[i+1] <= data[i]:
            break
    return i+1


def find_index_point_symm(where_to_find, what_to_find):
    break_point_where = point_of_1st_change_slope(where_to_find)
    break_point_what  = point_of_1st_change_slope(what_to_find)

    indices_1 = find_index_monotonic(where_to_find[:break_point_where],
                                     what_to_find[:break_point_what])

    indices_2 = find_index_monotonic(where_to_find[break_point_where:],
                                     what_to_find[break_point_what:])
    indices_2 += break_point_where

    return np.concatenate((indices_1, indices_2))


def rmsd(list_of_data, ref_data):
    ref_data = _shift_min_to_zero(ref_data)

    for i in range(len(list_of_data)):
        assert list_of_data[i].shape == ref_data.shape, "data and ref_data do not have the same shape"
        list_of_data[i] = _shift_min_to_zero(list_of_data[i])

    err = ( np.array(list_of_data) - ref_data[None, :] )**2
    err = np.sqrt( err.mean(axis=0) )

    return err


def equal_stride_subsample(data, nsamples):
    assert nsamples < len(data), "nsamples must less than len of data"
    new_index = np.linspace(0, len(data)-1, nsamples, dtype=int)
    return data[new_index]


us_result = pickle.load( open(args.us_fe_file, "r") )
us_lambdas = us_result["lambdas"]
us_fe = us_result["fe"]


pull_estimators = args.pull_estimators.split()
fes_files = {est : os.path.join(args.pull_fe_dir, FE_FILE_PREFIX+est+".pkl")
             for est in pull_estimators}

fes = { est : pickle.load( open(fes_files[est], "r") ) for est in pull_estimators }

pull_lambdas = fes[pull_estimators[0]]["lambdas"]


if args.is_protocol_symmetric:
    us_lambdas = np.concatenate( (us_lambdas, us_lambdas[::-1][1:]) )
    us_fe = np.concatenate( (us_fe, us_fe[::-1][1:]))

    indices = find_index_point_symm(pull_lambdas, us_lambdas)
else:
    indices = find_index_monotonic(pull_lambdas, us_lambdas)


test_match = []
for i, ind in enumerate(indices):
    test_match.append( np.abs(us_lambdas[i] - pull_lambdas[ind]) )

print("Test matching, max =", np.max(test_match), " min =", np.min(test_match))

# plot fe
xs = [fes[est]["lambdas"][indices] for est in pull_estimators]
xs.append(us_lambdas)
xs = [equal_stride_subsample(x, args.nsamples) for x in xs]

ys = [_shift_min_to_zero(fes[est]["mean"][indices]) for est in pull_estimators]
ys.append(_shift_min_to_zero(us_fe))
ys = [y*KB*TEMPERATURE for y in ys]  # kT to kcal/mol
ys = [equal_stride_subsample(y, args.nsamples) for y in ys]

yerrs = [fes[est]["std"][indices] for est in pull_estimators]
yerrs = [yerr*KB*TEMPERATURE for yerr in yerrs] # kT to kcal/mol
yerrs = [equal_stride_subsample(yerr, args.nsamples) for yerr in yerrs]
yerrs = [yerr/2. for yerr in yerrs] # 1 std
yerrs.append(None)

if args.xlimits.lower() != "none":
    xlimits = [float(s) for s in args.xlimits.split()]
else:
    xlimits = None
print("xlimits = ", xlimits)

if args.ylimits_fe.lower() != "none":
    ylimits_fe = [float(s) for s in args.ylimits_fe.split()]
else:
    ylimits_fe = None
print("ylimits_fe = ", ylimits_fe)


plot_lines(xs, ys, yerrs=yerrs,
           xlabel=args.xlabel, ylabel=args.ylabel_fe,
           out=args.fe_out,
           legends=pull_estimators + ["us"],
           legend_pos="best",
           legend_fontsize=8,
           xlimits=xlimits,
           ylimits=ylimits_fe,
           markers=MARKERS,
           lw=1.0,
           markersize=5,
           alpha=0.5,
           n_xtics=8,
           n_ytics=8)


# rmsd with respect to US
xs = []
ys = []

for est in pull_estimators:
    list_of_fes = fes[est]["fes"].values()
    list_of_fes = [fe[indices] for fe in list_of_fes]
    err = rmsd(list_of_fes, us_fe)
    ys.append(err)

    xs.append(fes[est]["lambdas"][indices])

ys = [y*KB*TEMPERATURE for y in ys]

xs = [equal_stride_subsample(x, args.nsamples) for x in xs]
ys = [equal_stride_subsample(y, args.nsamples) for y in ys]

if args.ylimits_rmse.lower() != "none":
    ylimits_rmse = [float(s) for s in args.ylimits_rmse.split()]
else:
    ylimits_rmse = None
print("ylimits_rmse = ", ylimits_rmse)

plot_lines(xs, ys,
           xlabel=args.xlabel, ylabel=args.ylabel_rmse,
           out=args.rmse_out,
           legends=pull_estimators,
           legend_pos="best",
           legend_fontsize=8,
           markers=MARKERS,
           xlimits=xlimits,
           ylimits=ylimits_rmse,
           lw=1.0,
           markersize=5,
           alpha=0.5,
           n_xtics=8,
           n_ytics=8)

