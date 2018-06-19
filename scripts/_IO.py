
import netCDF4 as nc


def save_to_nc(data, nc_handle, exclude=None):
    """
    data is a dict
    """
    if exclude is None:
        exclude = []

    keys = [key for key in data.keys() if key not in exclude]

    # create dimensions
    for key in keys:
        for dim in data[key].shape:
            dim_name = "%d"%dim
            if dim_name not in nc_handle.dimensions.keys():
                nc_handle.createDimension( dim_name, dim)

    # create variables
    for key in keys:
        if data[key].dtype == int:
            store_format = "i8"
        elif data[key].dtype == float:
            store_format = "f8"
        else:
            raise RuntimeError("unsupported dtype %s"%data[key].dtype)

        dimensions = tuple([ "%d"%dim for dim in data[key].shape ])
        nc_handle.createVariable(key, store_format, dimensions)

    # save data
    for key in keys:
        nc_handle.variables[key][:] = data[key]

    return nc_handle


def load_1d_sim_results(nc_file):
    """
    :param nc_file: str
    :return: data, dict mapping key to float or ndarray
    """
    nc_handle = nc.Dataset(nc_file, "r")

    data = {}
    data["ks"]          = nc_handle.variables["ks"][0]
    data["dt"]          = nc_handle.variables["dt"][0]
    data["lambda_F"]    = nc_handle.variables["lambda_F"][:]
    data["lambda_R"]    = nc_handle.variables["lambda_R"][:]

    data["wF_t"]    = nc_handle.variables["wF_t"][:]
    data["wR_t"]    = nc_handle.variables["wR_t"][:]

    data["zF_t"]    = nc_handle.variables["zF_t"][:]
    data["zR_t"]    = nc_handle.variables["zR_t"][:]

    if "pulling_times" in nc_handle.variables.keys():
        data["pulling_times"] = nc_handle.variables["pulling_times"][:]

    nc_handle.close()
    return data

