from zlib import decompress

import numpy as np
from shapely.ops import transform
from ..utility.files.xml_tools import etree_to_simplified_dict
from ..visualization.raster_funcs import clean_raster
from ..visualization.shapely_funcs import bool_raster_to_polygon


class LmtDetection:
    """
    A class that represents an LMT detection. The class is initialized with an LMT detection xml dictionary or a string
    representation of an LMT detection XML element. This class is useful as an annotation to see the data structure
    expected from LMT detection XML elements.
    """
    isLookingDown = None
    isLookingUp = None
    isRearing = None
    mass_x = None
    mass_y = None
    mass_z = None
    t = None

    def __init__(self, lmt_xml_dict_or_string):
        xml_dict = etree_to_simplified_dict(lmt_xml_dict_or_string)
        for k, v in xml_dict["DATA"].items():
            self.__setattr__(k, v)
        self.roi = LmtRoiMask(xml_dict["ROI"])


class LmtRoiMask:
    """
    A class that represents an LMT ROI mask. The class is initialized with an LMT ROI dictionary. The LMT ROI dictionary
    is a dictionary representation of an LMT ROI XML element. This is more used as annotation for the data structure
    expected from LMT ROI XML elements. The mask attribute is a shapely polygon that represents the ROI mask with the
    correct position. The position is determined by the boundsX and boundsY values.
    """
    classname = None
    id = None
    name = None
    selected = None
    readOnly = None
    color = None
    stroke = None
    opacity = None
    showName = None
    z = None
    t = None
    c = None
    boundsX = 0
    boundsY = 0
    boundsW = 0
    boundsH = 0
    boolMaskData = ""
    _mask = None
    _raster = None
    _clean_raster = None

    def __init__(self, roi_dict):
        for k, v in roi_dict.items():
            self.__setattr__(k, v)

    @property
    def mask(self):
        """
        The mask attribute is a shapely polygon that represents the ROI mask with the correct position. The position is
        determined by the boundsX and boundsY values, as well as the boundsW and boundsH values. This is used so that
        the mask is only created once and only when accessed.
        :return: A shapely polygon that represents the ROI mask with the correct position
        :rtype: shapely.geometry.Polygon
        """
        if self._mask is None:
            self._mask = create_polygon_from_roi_dict(self.__dict__)
        return self._mask

    @property
    def raster(self):
        """
        The raster attribute is a boolean raster that represents the ROI mask in a (424, 512) frame. The raster is of
        shape (424, 512). The raster is translated to the correct position using the boundsX and boundsY values. Its
        values are extracted from the boolMaskData value. This is used so that the raster is only created once and only
        when accessed.
        :return: A boolean raster that represents the ROI mask in a (424, 512) frame
        :rtype: np.ndarray
        """
        if self._raster is None:
            self._raster = create_frame_raster_from_roi_dict(self.__dict__)
        return self._raster

    @property
    def clean_raster(self):
        """
        The clean_raster attribute is a cleaned version of the raster attribute. The clean_raster is created by
        dilating and eroding the raster attribute. This is used so that the clean_raster is only created once and only
        when accessed.
        :return: A cleaned version of the raster attribute
        :rtype: np.ndarray
        """
        if self._clean_raster is None:
            self._clean_raster = create_clean_frame_raster_from_roi_dict(self.__dict__)
        return self._clean_raster


def decompress_raster_hex_string(lmt_raster_hex_string):
    """
    Decompress the LMT raster hex string. The LMT raster hex string is a string of hex values separated by colons.
    The leading zeroes in the values were stripped off. The hex values are decompressed using zlib.

    :param lmt_raster_hex_string: The LMT raster hex string
    :type lmt_raster_hex_string: str
    :return: The decompressed hex string
    :rtype: str
    """
    uncompressed = " ".join([v.zfill(2) for v in lmt_raster_hex_string.split(":")])
    uncompressed = decompress(bytearray.fromhex(uncompressed))
    return uncompressed


def raster_hex_string_to_bool_raster(lmt_raster_hex_string, raster_shape):
    """
    Translate the LMT raster hex string to a boolean raster. Uses the decompress_lmt_raster_hex_string function to
    decompress the hex string. The decompressed hex string is converted to a numpy array of booleans with the specified
    shape.

    :param lmt_raster_hex_string: The LMT raster hex string
    :type lmt_raster_hex_string: str
    :param raster_shape: The shape of the raster
    :type raster_shape: tuple of int
    :return: A boolean raster
    :rtype: np.ndarray
    """
    uncompressed = decompress_raster_hex_string(lmt_raster_hex_string)
    bool_raster = np.array(tuple(uncompressed), dtype=bool).reshape(*raster_shape)
    return bool_raster


def create_frame_raster_from_roi_dict(roi_dict, frame_shape=(424, 512)):
    """
    Create a boolean raster from an LMT ROI dictionary. The raster is of shape frame_shape. The raster is translated
    to the correct position using the boundsX and boundsY values. Its values are extracted from the boolMaskData value.

    :param roi_dict: The LMT ROI dictionary
    :type roi_dict: dict
    :param frame_shape: The shape of the frame
    :type frame_shape: tuple of int
    :return: A boolean raster
    :rtype: np.ndarray
    """
    x, y = roi_dict["boundsX"], roi_dict["boundsY"]
    mask_shape = (roi_dict["boundsH"], roi_dict["boundsW"])
    mask = raster_hex_string_to_bool_raster(roi_dict["boolMaskData"], mask_shape)
    frame = np.zeros(frame_shape, dtype=bool)
    frame[y:y + mask_shape[0], x:x + mask_shape[1]] = mask
    return frame


def create_clean_frame_raster_from_roi_dict(roi_dict, frame_shape=(424, 512), **clean_frame_raster_kwargs):
    """
    Create a cleaned boolean raster from an LMT ROI dictionary. The raster is of shape frame_shape. The raster is
    cleaned by dilating and eroding it. See the clean_frame_raster function for more information on the cleaning
    process.

    :param roi_dict: The LMT ROI dictionary
    :type roi_dict: dict
    :param frame_shape: The shape of the frame
    :type frame_shape: tuple of int
    :param clean_frame_raster_kwargs: Keyword arguments passed to the clean_frame_raster function
    :return: A cleaned boolean raster
    :rtype: np.ndarray
    """
    frame_raster = create_frame_raster_from_roi_dict(roi_dict, frame_shape)
    return clean_raster(frame_raster, **clean_frame_raster_kwargs)


def create_polygon_from_roi_dict(roi_dict):
    """
    Create a boolean mask from an LMT ROI dictionary. The mask is of shape (boundsH, boundsW). The mask is translated
    to the correct position using the boundsX and boundsY values. Its values are extracted from the boolMaskData value.

    :param roi_dict: The LMT ROI dictionary
    :type roi_dict: dict
    :return: A boolean mask
    :rtype: np.ndarray
    """
    x, y = roi_dict["boundsX"], roi_dict["boundsY"]
    mask_shape = (roi_dict["boundsH"], roi_dict["boundsW"])
    mask = bool_raster_to_polygon(
        raster_hex_string_to_bool_raster(roi_dict["boolMaskData"], mask_shape))
    mask = transform(lambda _x, _y, z=None: (_x + x, _y + y), mask)
    return mask
