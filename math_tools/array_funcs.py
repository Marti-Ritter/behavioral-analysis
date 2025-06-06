"""
Utility functions for numpy arrays
"""

import numpy as np


def get_closest_value(array, value, direction='up'):
    """
    Get the value in the array that is closest to the given value

    :param array: The array to search
    :type array: np.ndarray
    :param value: The value to search for
    :type value: float
    :param direction: The direction to search in. Can be 'up', 'down', or 'closest'
    :type direction: str
    :return: The value in the array that is closest to the given value
    :rtype: float
    """
    assert direction in ['up', 'down', 'closest'], "direction must be 'up', 'down', or 'closest'"
    array = np.array(array)  # Make sure the array is a numpy array

    if direction == 'up':
        return array[array >= value].min()
    elif direction == 'down':
        return array[array <= value].max()
    else:
        return array[np.argmin(np.abs(array - value))]


def get_most_recent_index(known_indices, reference_index):
    """
    Get the index in the known_indices that is closest to the reference_index. Convenience function for
    get_closest_value

    :param known_indices: Known indices
    :type known_indices: np.ndarray
    :param reference_index: The reference index
    :type reference_index: int
    :return: The index in the known_indices that is closest to the reference_index
    :rtype: int
    """
    known_indices = np.array(known_indices)  # Make sure the array is a numpy array
    assert reference_index > known_indices.min(), "reference_index must be greater than the minimum of known_indices"
    return get_closest_value(known_indices, reference_index, direction='down')


def np_mse(array1, array2):
    """
    Calculate the mean squared error between two arrays of the same shape using numpy functions only.
    Equivalent to sklearn.metrics.mean_squared_error(array1, array2).
    Taken from https://stackoverflow.com/a/47374870.

    :param array1: Array 1
    :type array1: numpy.ndarray
    :param array2: Array 2
    :type array2: numpy.ndarray
    :return: Mean squared error between array1 and array2
    :rtype: float
    """
    return np.square(np.subtract(array1, array2)).mean()


def np_axis_choice(a, axis=0, p=None, random_state=None):
    """
    This function selects a single value across a given axis from an array. The selection is made according to the
    probabilities in p. If p is None, then a uniform distribution is assumed. The result is an array with the same
    shape as a, except that the axis along which the selection was made is removed. p must have the same length as the
    size of the axis along which the selection is made and must sum to 1.

    :param a: Array from which the selection is made
    :type a: numpy.ndarray
    :param axis: Axis along which the selection is made
    :type axis: int
    :param p: Probabilities of selection
    :type p: numpy.ndarray
    :param random_state: Random state
    :type random_state: int
    :return: Array with the same shape as a, except that the axis along which the selection was made is removed
    :rtype: numpy.ndarray
    """
    if random_state is not None:
        np.random.seed(random_state)

    return np.apply_along_axis(np.random.choice, axis, a, p=p)


def np_sample_along_first_axis(arr, p=None, random_state=None):
    """
    A faster version of np_axis_choice for the first axis. This function selects a single value across the first axis
    from an array. The selection is made according to the probabilities in p. If p is None, then a uniform distribution
    is assumed. The result is an array with the same shape as a, except that the first axis is removed. p must have the
    same length as the size of the first axis and must sum to 1.

    :param arr: Array from which the selection is made
    :type arr: np.ndarray
    :param p: Probabilities of selection along the first axis
    :type p: np.ndarray
    :param random_state: Random state
    :type random_state: int
    :return: Array with the same shape as a, except that the first axis is removed
    :rtype: np.ndarray
    """

    x, *y = arr.shape

    if random_state is not None:
        np.random.seed(random_state)

    if p is None:
        p = np.ones(x) / x
    else:
        p = np.array(p)
        assert len(p) == x, "Length of p must be equal to the size of axis x"
        p /= p.sum()

    random_indices = np.random.choice(np.arange(x), size=y, p=p)
    index_arrays = np.meshgrid(*[np.arange(dim_size) for dim_size in y], indexing='ij')
    index_arrays.insert(0, random_indices)
    sampled_array = arr[tuple(index_arrays)]
    return sampled_array


def np_update(array1, array2, overwrite=False, filter_func=None):
    """
    A numpy implementation of pandas DataFrame.update and Series.update with a similar signature. The function updates
    array1 with the values from array2 in locations where array2 is not NaN. If overwrite is True, then array1 is
    updated with the values from array2 in all locations, otherwise only in locations where array1 is NaN.
    Both array1 and array2 must be numpy arrays of the same shape.
    If filter_func is not None, then it must be a function that takes the two arrays as input and returns a boolean
    array of the same shape as array1 and array2. The function is then only applied to locations where the boolean array
    is True.

    :param array1: A numpy array to be updated
    :type array1: numpy.ndarray
    :param array2: A numpy array with the values to update array1 with
    :type array2: numpy.ndarray
    :param overwrite: A boolean indicating whether to overwrite array1 with the values from array2 in all locations or
    only in locations where array1 is NaN (default, False)
    :type overwrite: bool
    :param filter_func: A function that takes the two arrays as input and returns a boolean array of the same shape as
    array1 and array2. The function is then only applied to locations where the boolean array is True.
    :type filter_func: function
    :return: The updated array1
    :rtype: numpy.ndarray
    """

    if filter_func is None:
        def _nan_filter(x, y):
            return np.isnan(x) & ~np.isnan(y) if overwrite else np.isnan(x)
        filter_func = _nan_filter

    filter_array = filter_func(array1, array2)
    array1[filter_array] = array2[filter_array]
    return array1


def truncate_array_with_mask(input_array, mask, fill_value=np.nan):
    """
    Truncate the image with the mask. The mask is a boolean mask. The fill_value is the value that will be used to
    fill the image where the mask is False. The fill_value is NaN by default. This is more or less a convenience
    function for np.where(mask, img, np.full_like(img.astype(float), fill_value)), just to make the code more readable.

    :param input_array: An image
    :type input_array: np.ndarray
    :param mask: A boolean mask
    :type mask: np.ndarray
    :param fill_value: The value to fill the image with where the mask is False
    :type fill_value: float or int
    :return: The truncated image
    :rtype: np.ndarray
    """
    return np.where(mask, input_array, np.full_like(input_array.astype(float), fill_value))


def get_middle_index(array_length):
    """
    Just a definition for the "middle" of an array. This is relative to index 0, and results in a length 11 array
    returning 5:
    0 1 2 3 4 5 6 7 8 9 10 (length 11)
    I I I I I X I I I I I (5 is the middle index, because it's at the center of an odd length array)

    For an even length array, this will be the left "middle" index.
    0 1 2 3 4 5 6 7 8 9 (length 10)
    I I I I X I I I I I (4 is the middle index)

    This definition is necessary for the alignment of e.g. VAME motif arrays.

    :param array_length: The length of an array
    :type array_length: int
    :return: The index that represents "middle" in the context of this function
    :type: int
    """
    return np.ceil(array_length / 2).astype(int) - 1  # minus 1 due to zero-indexing
