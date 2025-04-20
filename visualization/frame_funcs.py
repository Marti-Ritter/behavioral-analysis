import numpy as np

from ..math_tools.geometry_funcs import ensure_bbox_inside_frame


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
