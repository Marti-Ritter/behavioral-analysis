import numpy as np


def get_linear_transform(input_x, input_y):
    """Get the linear fit between two input sequences

    :param input_x: sequence of values to fit to
    :type input_x: tuple or list
    :param input_y: sequence of values to fit to x
    :type input_y: tuple or list
    :return: tuple of parameters for the linear fit
    :rtype: tuple
    """
    return np.poly1d(np.polyfit(input_x, input_y, 1))


def invert_matrix(input_matrix):
    return np.linalg.pinv(np.r_[input_matrix, [np.zeros(3)]])[:2, ]


def multiply_by_matrix(input_array, input_matrix):
    return np.matmul(input_array, input_matrix)[:input_array.shape[0]]


def apply_transform_matrix(input_array, transform_matrix):
    assert (input_array.shape[-1] == transform_matrix.shape[
        1]), "The last dimension of the input array must match the number of columns in the transform matrix."
    if input_array.ndim == 2:
        return np.apply_along_axis(lambda x: np.matmul(x, transform_matrix) + transform_matrix[-1,], 1, input_array)
    else:
        return multiply_by_matrix(input_array, transform_matrix)


def apply_2d_affine_transform(input_array, transform_matrix):
    """
    A function that applies an affine transformation matrix to a 2D array of points. The matrix can describe any
    affine transformation, including translation, rotation, scaling.

    :param input_array:
    :type input_array:
    :param transform_matrix:
    :type transform_matrix:
    :return:
    :rtype:
    """
    input_array = np.asarray(input_array)  # Ensure points is a numpy array
    # Add a row of ones at the end of points array
    input_array = np.concatenate([input_array, np.ones((input_array.shape[0], 1))], axis=1)
    transformed_points = np.dot(input_array, transform_matrix.T)  # Apply the transformation matrix
    return transformed_points


def apply_inverse_transform(input_transform, input_value):
    """

    :param input_transform:
    :type input_transform:
    :param input_value:
    :type input_value:
    :return:
    :rtype:
    """
    return (input_transform - input_value).roots


def inverse_linear_transform(input_transform, input_value):
    """

    :param input_transform:
    :type input_transform:
    :param input_value:
    :type input_value:
    :return:
    :rtype:
    """
    return apply_inverse_transform(input_transform, input_value)[0]
