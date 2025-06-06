import numpy as np


def get_roi_from_frame(frame, roi_bbox, ensure_inside_frame=True):
    if ensure_inside_frame:
        x, y, w, h = ensure_bbox_inside_frame((frame.shape[1], frame.shape[0]), roi_bbox)
    else:
        x, y, w, h = roi_bbox
    return frame[y:y + h, x:x + w]


def get_padded_roi_from_frame(frame, roi_bbox, **pad_kwargs):
    """
    Get a ROI from a frame. If the ROI is partially outside the frame, the ROI is padded with zeros so that it fits the
    size of the given ROI bounding box.

    :param frame: The frame to extract the ROI from.
    :type frame: np.ndarray
    :param roi_bbox: The bounding box of the ROI.
    :type roi_bbox: tuple[int]
    :param pad_kwargs: Keyword arguments for the np.pad function.
    :type pad_kwargs: dict[str, Any]
    :return: The ROI extracted from the frame.
    :rtype: np.ndarray
    """
    height, width = frame.shape[:2]
    xo, yo, wo, ho = roi_bbox

    left_pad = max(0, -xo)
    right_pad = max(0, xo + wo - width)
    top_pad = max(0, -yo)
    bottom_pad = max(0, yo + ho - height)

    if xo < 0:
        wo = wo + xo
        xo = 0
    if yo < 0:
        ho = ho + yo
        yo = 0

    pad_format = ((top_pad, bottom_pad), (left_pad, right_pad)) + ((0, 0),) * (frame.ndim - 2)
    return np.pad(frame[yo:yo + ho, xo:xo + wo], pad_format, **pad_kwargs)


def is_gray(frame):
    # from https://stackoverflow.com/a/58791118
    if len(frame.shape) < 3: return True
    if frame.shape[2]  == 1: return True
    b,g,r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    if (b==g).all() and (b==r).all(): return True
    return False


def ensure_bbox_inside_frame(frame_size, bbox):
    """
    Transform a bounding box so that it is entirely inside the frame limits.

    :param frame_size: A tuple containing the width and height of the video frame.
    :type frame_size: tuple(int, int)
    :param bbox: A tuple containing the x-coordinate, y-coordinate, width, and height of the bounding box.
    :type bbox: tuple(int, int, int, int)
    :return: A tuple containing the x-coordinate, y-coordinate, width, and height of the transformed bounding box.
    :rtype: tuple(int, int, int, int)
    """

    x_min, y_min, width, height = bbox
    frame_width, frame_height = frame_size

    # Calculate the x_max and y_max coordinates of the bounding box
    x_max = x_min + width
    y_max = y_min + height

    # If the bounding box is already inside the frame limits, return it
    if x_min >= 0 and y_min >= 0 and x_max <= frame_width and y_max <= frame_height:
        return bbox

    # Calculate the amount to shift the bounding box in the x and y directions
    x_shift = 0
    y_shift = 0
    if x_min < 0:
        x_shift = -x_min
    elif x_max > frame_width:
        x_shift = frame_width - x_max

    if y_min < 0:
        y_shift = -y_min
    elif y_max > frame_height:
        y_shift = frame_height - y_max

    # Transform the bounding box with the calculated shift
    new_x_min = x_min + x_shift
    new_y_min = y_min + y_shift

    # Correct width and height if necessary
    width = min(width, frame_width - new_x_min)
    height = min(height, frame_height - new_y_min)

    return new_x_min, new_y_min, width, height
