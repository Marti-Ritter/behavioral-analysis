from json import JSONEncoder

import numpy as np
from rasterio.features import shapes
from shapely import from_geojson, MultiPolygon
from shapely.affinity import translate, scale

from .raster_funcs import bool_raster_from_functions


def polygon_from_bool_functions(func_list, x_lim=None, y_lim=None, resolution=1):
    bool_raster = bool_raster_from_functions(func_list, x_lim=x_lim, y_lim=y_lim, resolution=resolution)

    raster_polygon = bool_raster_to_polygon(bool_raster)
    raster_polygon = translate(raster_polygon, xoff=x_lim[0] / resolution, yoff=y_lim[0] / resolution)
    raster_polygon = scale(raster_polygon, xfact=resolution, yfact=resolution, origin=(0, 0))
    return raster_polygon


def bool_raster_to_polygon(bool_raster):
    """
    Convert a boolean raster to a shapely polygon. The boolean raster is converted to a set of shapely polygons using
    the rasterio.features.shapes function. The polygons are then combined into a single polygon using the shapely
    MultiPolygon class. The holes are subtracted from the polygon using the difference method of the MultiPolygon class.

    :param bool_raster: A boolean raster that represents a polygon
    :type bool_raster: np.ndarray
    :return: A shapely MultiPolygon that represents the polygon
    :rtype: shapely.geometry.MultiPolygon
    """
    raster_shapes = shapes(bool_raster.astype(np.uint8))
    encoder = JSONEncoder()
    polygons, holes = [], []
    for shape_dict, shape_value in raster_shapes:
        _p = from_geojson(encoder.encode(shape_dict))
        if shape_value:
            polygons.append(_p)
        else:
            holes.append(_p)
    return MultiPolygon(polygons).difference(MultiPolygon(holes))
