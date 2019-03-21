"""
"""
from __future__ import print_function
from __future__ import division

import numpy as np


def first_to_zero(values):
    return values - values[0]


def min_to_zero(values):
    argmin = np.argmin(values)
    return values - values[argmin]


def argmin_to_target(to_be_transformed, target):
    """
    transfrom such that to_be_transformed[argmin] == target[argmin]
    where argmin = np.argmin(target)
    """
    assert to_be_transformed.ndim == target.ndim == 1, "to_be_transformed and target must be 1d"
    #assert to_be_transformed.shape == target.shape, "pmf_to_be_transformed and pmf_target must have the same shape"

    argmin = np.argmin(target)
    if argmin >= to_be_transformed.shape[0]:
        raise IndexError("argmin >= to_be_transformed.shape[0]")

    d = target[argmin] - to_be_transformed[argmin]
    transformed = to_be_transformed + d

    return transformed


def _replicate(first_half, method, exclude_last_in_first_half=True):
    if method not in ["as_is", "to_the_right_of_zero"]:
        raise ValueError("Unrecognized method")

    if exclude_last_in_first_half:
        second_half = first_half[:-1]
    else:
        second_half = first_half

    second_half = second_half[::-1]
    if method == "as_is":
        return np.hstack([first_half, second_half])
    else:
        return np.hstack([first_half, -second_half])


def reverse_data_order(data):
    """Only need to do for free energies"""
    data["free_energies"]["lambdas"] = data["free_energies"]["lambdas"][::-1]

    for block in data["free_energies"]["main_estimates"]:
        data["free_energies"]["main_estimates"][block] = data["free_energies"]["main_estimates"][block][::-1]

    bootstrap_keys = [bt for bt in data["free_energies"] if bt.startswith("bootstrap_")]
    for bootstrap_key in bootstrap_keys:
        for block in data["free_energies"][bootstrap_key]:
            data["free_energies"][bootstrap_key][block] = data["free_energies"][bootstrap_key][block][::-1]

    return data


def replicate_data_1d(data, system_type):
    """when using the function, we assume that the protocol is asymmetric"""
    if system_type == "symmetric":
        data["free_energies"]["lambdas"] = _replicate(data["free_energies"]["lambdas"], method="to_the_right_of_zero",
                                                      exclude_last_in_first_half=True)
        data["pmfs"]["pmf_bin_edges"] = _replicate(data["pmfs"]["pmf_bin_edges"], method="to_the_right_of_zero",
                                                   exclude_last_in_first_half=True)

    elif system_type == "asymmetric":
        data["free_energies"]["lambdas"] = _replicate(data["free_energies"]["lambdas"], method="as_is",
                                                      exclude_last_in_first_half=True)
        # for asymmetric systems, the pmf cover the full landscape,
        # so we don't need to replicate
    else:
        raise ValueError("Unrecognized system_type")

    for block in data["free_energies"]["main_estimates"]:
        data["free_energies"]["main_estimates"][block] = _replicate(data["free_energies"]["main_estimates"][block],
                                                                    method="as_is", exclude_last_in_first_half=True)

    # only replicate pmf fot symmetric system but not asymmetric one
    if system_type == "symmetric":
        for block in data["pmfs"]["main_estimates"]:
            data["pmfs"]["main_estimates"][block] = _replicate(data["pmfs"]["main_estimates"][block], method="as_is",
                                                                 exclude_last_in_first_half=False)

    bootstrap_keys = [bt for bt in data["free_energies"] if bt.startswith("bootstrap_")]
    for bootstrap_key in bootstrap_keys:
        for block in data["free_energies"][bootstrap_key]:
            data["free_energies"][bootstrap_key][block] = _replicate(data["free_energies"][bootstrap_key][block],
                                                                     method="as_is", exclude_last_in_first_half=True)

    # only replicate pmf fot symmetric system but not asymmetric one
    if system_type == "symmetric":
        bootstrap_keys = [bt for bt in data["pmfs"] if bt.startswith("bootstrap_")]
        for bootstrap_key in bootstrap_keys:
            for block in data["pmfs"][bootstrap_key]:
                data["pmfs"][bootstrap_key][block] = _replicate(data["pmfs"][bootstrap_key][block], method="as_is",
                                                                  exclude_last_in_first_half=False)
    return data


def replicate_data_cuc7_da(data, system_type):
    """when using the function, we assume that the protocol is asymmetric"""
    if system_type == "symmetric":
        data["free_energies"]["lambdas"] = _replicate(data["free_energies"]["lambdas"], method="to_the_right_of_zero",
                                                      exclude_last_in_first_half=False)
        data["pmfs"]["pmf_bin_edges"] = _replicate(data["pmfs"]["pmf_bin_edges"], method="to_the_right_of_zero",
                                                   exclude_last_in_first_half=True)

    elif system_type == "asymmetric":
        data["free_energies"]["lambdas"] = _replicate(data["free_energies"]["lambdas"], method="as_is",
                                                      exclude_last_in_first_half=True)
        # for asymmetric systems, the pmf cover the full landscape,
        # so we don't need to replicate
    else:
        raise ValueError("Unrecognized system_type")

    for block in data["free_energies"]["main_estimates"]:
        if system_type == "symmetric":
            data["free_energies"]["main_estimates"][block] = _replicate(data["free_energies"]["main_estimates"][block],
                                                                    method="as_is", exclude_last_in_first_half=False)
        elif system_type == "asymmetric":
            data["free_energies"]["main_estimates"][block] = _replicate(data["free_energies"]["main_estimates"][block],
                                                                        method="as_is",
                                                                        exclude_last_in_first_half=True)

    # only replicate pmf fot symmetric system but not asymmetric one
    if system_type == "symmetric":
        for block in data["pmfs"]["main_estimates"]:
            data["pmfs"]["main_estimates"][block] = _replicate(data["pmfs"]["main_estimates"][block], method="as_is",
                                                                 exclude_last_in_first_half=False)

    bootstrap_keys = [bt for bt in data["free_energies"] if bt.startswith("bootstrap_")]
    for bootstrap_key in bootstrap_keys:
        for block in data["free_energies"][bootstrap_key]:
            if system_type == "symmetric":
                data["free_energies"][bootstrap_key][block] = _replicate(data["free_energies"][bootstrap_key][block],
                                                                     method="as_is", exclude_last_in_first_half=False)
            elif system_type == "asymmetric":
                data["free_energies"][bootstrap_key][block] = _replicate(data["free_energies"][bootstrap_key][block],
                                                                         method="as_is", exclude_last_in_first_half=True)

    # only replicate pmf fot symmetric system but not asymmetric one
    if system_type == "symmetric":
        bootstrap_keys = [bt for bt in data["pmfs"] if bt.startswith("bootstrap_")]
        for bootstrap_key in bootstrap_keys:
            for block in data["pmfs"][bootstrap_key]:
                data["pmfs"][bootstrap_key][block] = _replicate(data["pmfs"][bootstrap_key][block], method="as_is",
                                                                  exclude_last_in_first_half=False)
    return data


def put_first_of_fe_to_zero(data):
    for block in data["free_energies"]["main_estimates"]:
        data["free_energies"]["main_estimates"][block] = first_to_zero(data["free_energies"]["main_estimates"][block])

    bootstrap_keys = [bt for bt in data["free_energies"] if bt.startswith("bootstrap_")]
    for bootstrap_key in bootstrap_keys:
        for block in data["free_energies"][bootstrap_key]:
            data["free_energies"][bootstrap_key][block] = first_to_zero(data["free_energies"][bootstrap_key][block])
    return data


def put_argmin_of_pmf_to_target(data, target):
    for block in data["pmfs"]["main_estimates"]:
        data["pmfs"]["main_estimates"][block] = argmin_to_target(data["pmfs"]["main_estimates"][block], target)

    bootstrap_keys = [bt for bt in data["pmfs"] if bt.startswith("bootstrap_")]
    for bootstrap_key in bootstrap_keys:
        for block in data["pmfs"][bootstrap_key]:
            data["pmfs"][bootstrap_key][block] = argmin_to_target(data["pmfs"][bootstrap_key][block], target)

    return data
