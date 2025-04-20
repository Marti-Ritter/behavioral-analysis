import numpy as np


def cart2pol(x, y):
    """Transform a cartesian coordinate pair to polar coordinates

    :param x: x-coordinate
    :type x: float
    :param y: y-coordinate
    :type y: float
    :return: tuple of rho and phi, describing length and angle of the polar vector
    :rtype: tuple
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """Transform a pair of polar coordinates to cartesian coordinates

    :param rho: rho coordinate (Length)
    :type rho: float
    :param phi: phi coordinate (angle, radians)
    :type phi: float
    :return: tuple of xy coordinates
    :rtype: tuple
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def normalize_angle(input_angle, use_radians=False, return_sign=False):
    maximum_angle = 2 * np.pi if use_radians else 360
    phi = -input_angle % maximum_angle
    sign = -1
    # used to calculate sign
    if not ((0 <= phi <= maximum_angle / 2) or (-(maximum_angle / 2) >= phi >= -maximum_angle)):
        sign = 1
    if phi > maximum_angle / 2:
        result = maximum_angle - phi
    else:
        result = phi
    return ((result * sign), sign) if return_sign else result * sign


def delta_angle(angle_minuend, angle_subtrahend, use_radians=False, return_sign=False):
    # adapted from https://stackoverflow.com/a/36000994
    return normalize_angle(angle_minuend - angle_subtrahend, use_radians=use_radians, return_sign=return_sign)


def angular_diff(input_series, use_radians=True):
    output_series = input_series.copy()
    arguments = zip(input_series, input_series.shift(1), [use_radians] * len(input_series))
    output_series[:] = list(map(lambda x: delta_angle(x[0], x[1], use_radians=x[2]), arguments))
    return output_series


def calculate_change_over_time(input_series, expected_fps=30.00003):
    return input_series.diff() / (input_series.index.to_series().diff() / expected_fps)


def rotate_points_np_old(p, origin=(0, 0), angle=0):
    r = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((r @ (p.T - o.T) + o.T).T)


def rotate_2d_point_groups_np(points, origins=(0, 0), angles=0):
    """
    Rotate a group of 2D points around a given origin by a given angle in radians. Can be used to rotate multiple groups
    of points at once by passing points, origins and angles as arrays of the same length.
    Adapted from https://stackoverflow.com/a/58781388.

    :param points: Array of points to rotate, shape (2,) or (n_points, 2) or (n_groups, n_points, 2)
    :type points: np.ndarray
    :param origins: Array of origins to rotate around, shape (2,) or (n_groups, 2)
    :type origins: np.ndarray
    :param angles: Array of angles to rotate by, shape (1,) or (n_groups,)
    :type angles: np.ndarray or float
    :return: Rotated points. Shape (2,) or (n_points, 2) or (n_groups, n_points, 2)
    :rtype: np.ndarray
    """

    if np.array(points).ndim < 3:
        points = np.transpose(np.atleast_3d(points), axes=[2, 0, 1])
    points_transposed = np.transpose(points, axes=[0, 2, 1])
    origins = np.atleast_3d(origins)
    angles = np.atleast_1d(angles)

    rotation_matrix = np.transpose(np.array([[np.cos(angles), -np.sin(angles)],
                                             [np.sin(angles), np.cos(angles)]]), axes=[2, 0, 1])

    return np.squeeze(np.transpose(rotation_matrix @ (points_transposed - origins) + origins, axes=[0, 2, 1]))
