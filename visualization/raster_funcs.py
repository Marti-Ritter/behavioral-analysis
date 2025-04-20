import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation


def bool_func_intersection(func_list):
    def inner(*args, **kwargs):
        return all([func(*args, **kwargs) for func in func_list])

    return inner


def bool_raster_from_functions(func_list, x_lim=None, y_lim=None, resolution=1):
    x_positions = np.arange(x_lim[0], x_lim[1] + resolution, resolution)
    y_positions = np.arange(y_lim[0], y_lim[1] + resolution, resolution)
    bool_raster = np.zeros((len(y_positions), len(x_positions)), dtype=bool)

    combined_bool_func = bool_func_intersection(func_list)

    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            bool_raster[j, i] = combined_bool_func(x, y)
    return bool_raster


def clean_raster(raster_array, dilation_size=3, dilation_iterations=1, erosion_size=3, erosion_iterations=1):
    """
    Clean a boolean raster by dilating and eroding it. The dilation and erosion are done using a square structuring
    element with the specified size.

    :param raster_array: A boolean raster
    :type raster_array: np.ndarray
    :param dilation_size: The size of the dilation structuring element
    :type dilation_size: int
    :param dilation_iterations: The number of times to dilate the raster
    :type dilation_iterations: int
    :param erosion_size: The size of the erosion structuring element
    :type erosion_size: int
    :param erosion_iterations: The number of times to erode the raster
    :type erosion_iterations: int
    :return: The cleaned boolean raster
    :rtype: np.ndarray
    """
    dilation_struct = np.ones((dilation_size, dilation_size))
    erosion_struct = np.ones((erosion_size, erosion_size))
    return binary_erosion(binary_dilation(raster_array, dilation_struct, iterations=dilation_iterations),
                          erosion_struct, iterations=erosion_iterations)
