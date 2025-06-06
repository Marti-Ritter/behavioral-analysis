import numpy as np
from .angle_funcs import rotate_points_np_old, rotate_2d_point_groups_np
from .matrix_funcs import apply_transform_matrix
from shapely.affinity import scale, rotate
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


# from https://stackoverflow.com/a/40795835
def check_bbox_intersection(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    max_x1, max_y1, max_x2, max_y2 = x1 + w1, y1 + h1, x2 + w2, y2 + h2
    return not (max_x1 <= x2 or x1 >= max_x2 or max_y1 <= y2 or y1 >= max_y2)


def create_ellipsis(center_point, orientation, axis0, axis1):
    circle = Point(center_point).buffer(1)
    ells = scale(circle, axis0, axis1)
    return rotate(ells, orientation, use_radians=True)


def create_box(center_point, direction, axis0, axis1):
    """
    Creates a box polygon with the given center point, direction, and axis lengths.

    :param center_point:
    :type center_point:
    :param direction:
    :type direction:
    :param axis0:
    :type axis0:
    :param axis1:
    :type axis1:
    :return:
    :rtype:
    """
    return Polygon(get_box_points(center_point, direction, axis0, axis1))


def get_box_points(center_point, direction, axis0, axis1):
    x, y = center_point
    box_corners = np.array([
        [x + axis0, y + axis1],
        [x - axis0, y + axis1],
        [x - axis0, y - axis1],
        [x + axis0, y - axis1]
    ])
    return rotate_points_np_old(box_corners, origin=center_point, angle=direction)


def get_box_corners_np(center_points, directions, axis0_array, axis1_array):
    x, y = center_points.T
    box_corners = np.transpose(np.array([
        [x + axis0_array, y + axis1_array],
        [x - axis0_array, y + axis1_array],
        [x - axis0_array, y - axis1_array],
        [x + axis0_array, y - axis1_array]
    ]), axes=[2, 0, 1])
    return rotate_2d_point_groups_np(points=box_corners, origins=center_points, angles=directions)


def get_points_bbox(input_points):
    """
    Calculates the bounding box of a list or array of points (x, y) in the form (x_min, y_min, x_width, y_width).

    :param input_points: A list or array of points (x, y).
    :type input_points: list or np.ndarray
    :return: X and Y minimum values, and X and Y width values.
    :rtype: tuple[float, float, float, float]
    """
    point_array = np.array(input_points)
    minima = np.min(point_array, axis=0)
    return np.hstack([minima, np.max(point_array, axis=0) - minima])


def get_2d_point_groups_bboxes(input_points):
    """
    Calculates the bounding box of a list or array of points (x, y) in the form (x_min, y_min, x_width, y_width).
    Can handle multiple groups of points in the shape (group, point, x/y).

    :param input_points: A list or array of points (x, y). Can be multiple groups of points in the shape
        (group, point, x/y).
    :type input_points: list or np.ndarray
    :return: X and Y minimum values, and X and Y width values.
    :rtype: tuple[float, float, float, float]
    """

    point_array = np.array(input_points)
    minima = np.min(point_array, axis=point_array.ndim - 2)
    return np.hstack([minima, np.max(point_array, axis=point_array.ndim - 2) - minima])


def shapely_bbox_to_matplotlib_bbox(shapely_bbox):
    return shapely_bbox[0], shapely_bbox[1], shapely_bbox[2] - shapely_bbox[0], shapely_bbox[3] - shapely_bbox[1]


def matplotlib_bbox_to_shapely_bbox(matplotlib_bbox):
    return matplotlib_bbox[0], matplotlib_bbox[1], matplotlib_bbox[0] + matplotlib_bbox[2], matplotlib_bbox[1] + \
           matplotlib_bbox[3]


def scale_and_pad_bbox(input_bbox, x_fact=1, y_fact=1, x_pad=0, y_pad=0):
    x, y, width, height = input_bbox
    new_width, new_height = width * x_fact + x_pad, height * y_fact + y_pad
    center_x, center_y = x + width / 2, y + height / 2
    return center_x - new_width / 2, center_y - new_height / 2, new_width, new_height


def bbox_to_polygon(input_bbox):
    x, y, width, height = input_bbox
    return Polygon([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])


def cumulative_bbox(list_of_bbox):
    o_x, o_y, o_max_x, o_max_y = matplotlib_bbox_to_shapely_bbox(list_of_bbox.pop())
    for bbox in list_of_bbox:
        x, y, max_x, max_y = matplotlib_bbox_to_shapely_bbox(bbox)
        o_x, o_y, o_max_x, o_max_y = min(x, o_x), min(y, o_y), max(max_x, o_max_x), max(max_y, o_max_y)
    return shapely_bbox_to_matplotlib_bbox((o_x, o_y, o_max_x, o_max_y))


def polygon_unions(list_of_shapely_polygons):
    polygon_union = unary_union(list_of_shapely_polygons)
    geom_type = polygon_union.geom_type

    if polygon_union.is_empty:
        return []

    if geom_type in ["MultiPolygon", "GeometryCollection"]:
        return list(polygon_union)
    else:
        return [polygon_union]


def summarize_bboxes(list_of_bbox):
    bbox_list = list_of_bbox
    previous_bbox_list = []

    while bbox_list != previous_bbox_list:
        previous_bbox_list = bbox_list
        intersection_sets = []

        for bbox in bbox_list:
            for intersection_set in intersection_sets:
                if any([check_bbox_intersection(bbox, set_bbox) for set_bbox in intersection_set]):
                    intersection_set.append(bbox)
                    break
            else:
                intersection_sets.append([bbox])
        bbox_list = [cumulative_bbox(intersection_set) for intersection_set in intersection_sets]
    return bbox_list


def get_bbox(center_point, direction, axis0, axis1, calculation_mode="box"):
    if calculation_mode == "ellipsis":
        return shapely_bbox_to_matplotlib_bbox(create_ellipsis(center_point, direction, axis0, axis1).bounds)
    elif calculation_mode == "box":
        return get_points_bbox(get_box_points(center_point, direction, axis0, axis1))
    else:
        raise ValueError(f"Unknown calculation mode: {calculation_mode}")


def get_bbox_np(center_points, directions, axis0_array, axis1_array):
    box_corners_array = get_box_corners_np(center_points=center_points,
                                           directions=directions,
                                           axis0_array=axis0_array,
                                           axis1_array=axis1_array)
    return get_2d_point_groups_bboxes(box_corners_array)


def get_alignment_index(alignment_array, array_to_align):
    """
    Returns the index of the values in array_to_align in the sorted alignment_array.
    Default index for values not in alignment_array is 0. Due to this there is a check to ensure that all values in
    array_to_align are in alignment_array.

    :param alignment_array: Array to align to.
    :type alignment_array: np.ndarray
    :param array_to_align: Array to align.
    :type array_to_align: np.ndarray
    :return: Index of the values in array_to_align in the sorted alignment_array.
    :rtype: np.ndarray
    """

    if not set(array_to_align).issubset(set(alignment_array)):
        raise ValueError(f"Array to align contains values not in alignment array: "
                         f"{set(array_to_align) - set(alignment_array)}")

    sort_idx = alignment_array.argsort()
    return sort_idx[np.searchsorted(alignment_array, array_to_align, sorter=sort_idx)]


def check_point_in_bbox(point, bbox):
    p_x, p_y = point
    b_x, b_y, b_w, b_h = bbox
    return b_x <= p_x <= (b_x + b_w) and b_y <= p_y <= (b_y + b_h)


def get_tags_in_bbox(tag_positions, bbox_list):
    tag_bbox = {}
    for tag, position in tag_positions.items():
        tag_bbox[tag] = []
        for bbox in bbox_list:
            if check_point_in_bbox(position, bbox):
                tag_bbox[tag].append(bbox)
    return tag_bbox


def get_rois_from_bbox_list(list_of_bbox, mode="points"):
    """
    n=10
    point_based: 100 µs ± 6.69 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    polygon_based: 617 µs ± 20.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    n=5000
    point_based: 4.58 s ± 608 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    polygon_based: 1.46 s ± 109 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    --> presumably polygon overhead. Disappears for large numbers, recursions in point_based take over.

    :param list_of_bbox:
    :type list_of_bbox:
    :param mode:
    :type mode:
    :return:
    :rtype:
    """
    if mode == "points":
        return summarize_bboxes(list_of_bbox)
    elif mode == "polygons":
        return [shapely_bbox_to_matplotlib_bbox(poly.bounds) for poly in
                polygon_unions([bbox_to_polygon(bbox) for bbox in list_of_bbox])]
    else:
        raise ValueError(f"Unknown calculation mode: {mode}")


def apply_transform_to_bbox(bbox, transform_matrix):
    x, y, w, h = bbox
    return (*apply_transform_matrix(np.array([x, y]), transform_matrix),
            *apply_transform_matrix(np.array([w, h]), transform_matrix))


def fit_bbox_to_limits(bbox, limits, raise_error_when_outside=True):
    if len(limits) == 2:
        limit_bbox = 0, 0, *limits
    elif len(limits) == 4:
        limit_bbox = limits
    else:
        raise ValueError(f"Len(limits) should be 2 or 4! But it is {len(limits)}.")

    intersection = get_bbox_intersection(bbox, limit_bbox)

    if intersection is None and raise_error_when_outside:
        raise ValueError("Bbox outside limits! No fitting possible.")
    return intersection


def frame_shape_to_bbox(frame_shape):
    return (0, 0, *frame_shape)


def get_bbox_intersection(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_o, y_o = max(x1, x2), max(y1, y2)
    w_o, h_o = min(x1 + w1, x2 + w2) - x_o, min(y1 + h1, y2 + h2) - y_o

    if w_o <= 0 or h_o <= 0:
        return None
    else:
        return x_o, y_o, w_o, h_o


def np_matrix_to_shapely_matrix(numpy_matrix):
    shape = numpy_matrix.shape
    if shape[1] == 3 and shape[0] in (2, 3):
        return np.concatenate((numpy_matrix[:2, :2].flatten(), numpy_matrix[:2, 2].flatten()))
    elif shape[1] == 4 and shape[0] in (3, 4):
        return np.concatenate((numpy_matrix[:3, :3].flatten(), numpy_matrix[:3, 3].flatten()))
    else:
        raise ValueError(f"Invalid shape: {shape}!")


def get_bbox_from_event_segment(start, end, trace_df_dict, involved_mice_list, transform_matrix=None, x_scaling=1,
                                y_scaling=1):
    frame_bboxes = []
    for i in range(start, end + 1):
        trace_entries_for_mice = [trace_df_dict[int(mouse)].loc[i, :] for mouse in involved_mice_list]
        mouse_bboxes = [get_bbox((t["x"], t["y"]), t["direction"], t["axis0"], t["axis1"]) for t in
                        trace_entries_for_mice if not t.isna().any()]
        if len(mouse_bboxes) > 0:
            frame_bboxes.append(summarize_bboxes(mouse_bboxes)[0])
    if len(frame_bboxes) > 0:
        bbox = summarize_bboxes(frame_bboxes)[0]
        bbox = scale_and_pad_bbox(bbox, x_fact=x_scaling, y_fact=y_scaling)
        if transform_matrix is not None:
            bbox = apply_transform_to_bbox(bbox, transform_matrix)
        bbox = tuple(map(int, bbox))
        return bbox
    else:
        return None


def get_circle(center_tuple, radius):
    return Point(center_tuple).buffer(radius)


def get_polygon(list_of_xy):
    return Polygon(list_of_xy)


def get_polygon_centroid(list_or_polygons):
    if isinstance(list_or_polygons, list):
        return np.array(unary_union(list_or_polygons).centroid.xy).ravel().T
    else:
        return np.array(list_or_polygons.centroid.xy).ravel()


def check_trace_in_area(trace_df, area_polygon_list):
    return trace_df.apply(lambda x: any([poly.contains(Point(x)) for poly in area_polygon_list]), axis=1)


def get_sphere_coordinates(center_point, radius, resolution=20):
    """
    Returns the coordinates of a sphere with center_point and radius.
    Taken from https://stackoverflow.com/a/71053527

    :param center_point: tuple of floats (x, y, z)
    :type center_point: tuple of float
    :param radius: radius of the sphere
    :type radius: float
    :param resolution: resolution of the sphere
    :type resolution: int
    :return: tuple of arrays (X, Y, Z)
    :rtype: tuple of np.ndarray
    """

    x, y, z = center_point
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

