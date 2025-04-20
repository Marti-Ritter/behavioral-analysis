import numpy as np
import pandas as pd

from ..math_tools.angle_funcs import cart2pol, pol2cart, angular_diff, calculate_change_over_time


def calculate_angular_speed(input_series, use_radians=True, expected_fps=30.00003):
    return angular_diff(input_series, use_radians=use_radians) / (input_series.index.to_series().diff() / expected_fps)


def calculate_speed_1d(input_series, expected_fps=30.00003):
    return calculate_change_over_time(input_series, expected_fps=expected_fps)


def calculate_acceleration_1d(input_series, expected_fps=30.00003):
    return calculate_change_over_time(calculate_change_over_time(input_series, expected_fps=expected_fps),
                                      expected_fps=expected_fps)


def calculate_speed_2d(x_series, y_series, expected_fps=30.00003):
    x_speed = calculate_speed_1d(x_series, expected_fps=expected_fps)
    y_speed = calculate_speed_1d(y_series, expected_fps=expected_fps)
    return pd.Series(np.linalg.norm([x_speed, y_speed], axis=0), index=x_series.index)


def np_scalar_to_polar(x_array, y_array):
    return np.vectorize(cart2pol)(x_array, y_array)


def np_polar_to_scalar(rho_array, phi_array):
    return np.vectorize(pol2cart)(rho_array, phi_array)


def calculate_velocity_2d(x_series, y_series, expected_fps=30.00003):
    x_speed = calculate_speed_1d(x_series, expected_fps=expected_fps)
    y_speed = calculate_speed_1d(y_series, expected_fps=expected_fps)
    magnitude, direction = np_scalar_to_polar(x_speed, y_speed)
    return magnitude, direction


def calculate_acceleration_2d(x_series, y_series, expected_fps=30.00003):
    x_acceleration = calculate_acceleration_1d(x_series, expected_fps=expected_fps)
    y_acceleration = calculate_acceleration_1d(y_series, expected_fps=expected_fps)
    magnitude, direction = np_scalar_to_polar(x_acceleration, y_acceleration)
    return magnitude, direction


def calculate_acceleration_2d_old(x_series, y_series, expected_fps=30.00003):
    return calculate_change_over_time(calculate_speed_2d(x_series, y_series, expected_fps=expected_fps),
                                      expected_fps=expected_fps)
