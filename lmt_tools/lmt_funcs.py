import cv2
import numpy as np


def create_circle_polygon(center, radius, rotation_degree=0, start_degree=0, end_degree=360, step_size=1):
    """
    Create a polygon that approximates a circle. The polygon is defined by a set of points that are equally spaced
    along the circumference of the circle. The number of points is determined by the step_size parameter.

    :param center: The center of the circle
    :type center: tuple of int
    :param radius: The radius of the circle
    :type radius: int
    :param rotation_degree: The rotation of the circle in degrees
    :type rotation_degree: int
    :param start_degree: The starting degree of the circle
    :type start_degree: int
    :param end_degree: The ending degree of the circle
    :type end_degree: int
    :param step_size: The step size between points on the circle
    :type step_size: int
    :return: A polygon that approximates a circle
    :rtype: numpy.ndarray
    """
    polygon = cv2.ellipse2Poly(center, (radius, radius), rotation_degree, start_degree, end_degree, step_size)

    return polygon


def create_boolean_polygon_mask(polygon_points, mask_shape):
    """
    Generate a boolean mask from a polygon. The mask is the same shape as the mask_shape parameter. The polygon is
    defined by a list of points.

    :param polygon_points: The points that define the polygon
    :type polygon_points: list of (int, int)
    :param mask_shape: The shape of the mask
    :type mask_shape: tuple of int
    :return: A boolean mask
    :rtype: np.ndarray
    """
    mask = np.zeros(mask_shape)
    mask = cv2.fillPoly(mask, np.array([polygon_points]).astype(np.int32), color=1).astype(bool)
    return mask


def create_lmt_receptive_field_mask(frame_shape=(512, 424)):
    """
    Create a boolean mask that represents the receptive field of the LMT. The mask is the same shape as the frame_shape

    :param frame_shape: The shape of the frame
    :type frame_shape: tuple of int
    :return: A boolean mask
    :rtype: np.ndarray
    """
    mask_shape = np.array(frame_shape).astype(np.int32)
    center = (mask_shape/2).astype(np.int32)
    radius = 256
    receptive_field_polygon = create_circle_polygon(center, radius)
    receptive_mask = create_boolean_polygon_mask(receptive_field_polygon, mask_shape[::-1])
    return receptive_mask


def create_lmt_default_cage_wall_mask(frame_shape=(512, 424)):
    """
    Create a boolean mask that represents the upper limit of the wall of the cage of the LMT. The mask is the same shape
    as the frame_shape

    :param frame_shape: The shape of the frame
    :type frame_shape: tuple of int
    :return: A boolean mask
    :rtype: np.ndarray
    """
    # as defined in the LMT code (LiveMouseTracker.java), ~line 3332
    default_corners = [[84, 33], [428, 33], [428, 383], [84, 383]]
    mask_shape = np.array(frame_shape).astype(np.int32)
    cage_mask = create_boolean_polygon_mask(default_corners, mask_shape[::-1])
    return cage_mask


def create_lmt_default_cage_floor_mask(frame_shape=(512, 424)):
    """
    Create a boolean mask that represents the floor of the cage of the LMT. The mask is the same shape as the
    frame_shape

    :param frame_shape: The shape of the frame
    :type frame_shape: tuple of int
    :return: A boolean mask
    :rtype: np.ndarray
    """
    # as defined in the LMT code (LiveMouseTracker.java), ~line 3347
    default_corners = [[114, 63], [398, 63], [398, 353], [114, 353]]
    mask_shape = np.array(frame_shape).astype(np.int32)
    cage_mask = create_boolean_polygon_mask(default_corners, mask_shape[::-1])
    return cage_mask
