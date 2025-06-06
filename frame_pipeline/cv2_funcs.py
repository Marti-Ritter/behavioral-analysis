"""
A collection of functions that use OpenCV to work with images. Many functions in this module are adapted from the
corresponding functions in the pil_funcs module. The cv2 functions are still slower, and not as well implemented.
Try to avoid these.
"""
import cv2
import numpy as np

from .np_funcs import get_padded_roi_from_frame
from ..utility.builtin_classes.iterables import split_chunks, ensure_list
from ..utility.general import standardize_padding


def ensure_cv2_frame(frame_or_path):
    """
    Ensures that the specified object is a numpy.ndarray object. If the object is a path to an image file, the image
    will be loaded from the file. If the object is already a numpy.ndarray object, it will be returned as is. If the
    object is a PIL.Image.Image object, it will be converted to a numpy.ndarray object.
    This function is useful as an adapter for functions that accept either a numpy.ndarray object or a path to an
    image file.

    :param frame_or_path: A path to an image file or a numpy.ndarray object
    :type frame_or_path: str or numpy.ndarray
    :return: A numpy.ndarray object
    :rtype: numpy.ndarray
    """

    if isinstance(frame_or_path, str):
        return cv2.imread(frame_or_path, cv2.IMREAD_UNCHANGED)
    elif isinstance(frame_or_path, np.ndarray):
        frame_or_path = np.atleast_3d(frame_or_path)
        assert frame_or_path.shape[2] in [1, 3, 4], "image_or_path must have 1, 3, or 4 channels!"
        return frame_or_path
    else:
        raise TypeError("image_or_path must be a path to an image file or a numpy.ndarray object!")


def ensure_frame_is_rgb(frame):
    """
    Ensures that the given image is in RGB format. If the image is in grayscale format, it is converted to RGB format.

    :param frame: The image to ensure is in RGB format
    :type frame: numpy.ndarray
    :return: The image in RGB format
    :rtype: numpy.ndarray
    """
    frame = np.atleast_3d(frame)
    if frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return frame


def ensure_frame_has_alpha_channel(frame):
    """
    Ensures that the given image has an alpha channel. If the image does not have an alpha channel, an alpha channel is
    added to the image with a value of 255 (fully opaque).

    :param frame: The image to ensure has an alpha channel
    :type frame: numpy.ndarray
    :return: The image with an alpha channel
    :rtype: numpy.ndarray
    """

    if frame.shape[2] == 3:
        return np.dstack((frame, np.full(frame.shape[:2], 255, dtype=np.uint8)))
    return frame


def blend_bgr_and_bgra_image(bgr_background, bgra_foreground, x_offset=None, y_offset=None):
    """
    Blends an BGRA image with an BGR image. The alpha channel of the BGRA image is used to blend the two images
    together. The foreground (BGRA) image is placed on top of the background (BGR) image at the specified offsets.
    Taken from https://stackoverflow.com/a/71701023.

    :param bgr_background: The background image
    :type bgr_background: numpy.ndarray
    :param bgra_foreground: The foreground image
    :type bgra_foreground: numpy.ndarray
    :param x_offset: The x-offset of the foreground image relative to the background image. If None, the foreground
    image is centered on the background image.
    :type x_offset: int or None
    :param y_offset: The y-offset of the foreground image relative to the background image. If None, the foreground
    image is centered on the background image.
    :type y_offset: int or None
    :return: The blended image
    :rtype: numpy.ndarray
    """

    bgr_background = bgr_background.copy()
    bgra_foreground = bgra_foreground.copy()

    bg_h, bg_w, bg_channels = bgr_background.shape
    fg_h, fg_w, fg_channels = bgra_foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    bgra_foreground = bgra_foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = bgr_background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = bgra_foreground[:, :, :3]
    alpha_channel = bgra_foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = alpha_channel[:, :, np.newaxis]

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    bgr_background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

    return bgr_background


def create_blank_frame_like(rgb_frame, color=(255, 255, 255)):
    """
    Creates a blank frame with the same dimensions as the given RGB frame.

    :param rgb_frame: The frame to create a blank frame like
    :type rgb_frame: numpy.ndarray
    :param color: The color of the blank frame
    :type color: tuple
    :return: The blank frame
    :rtype: numpy.ndarray
    """

    return np.full_like(rgb_frame, color, dtype=np.uint8)


def pad_image(rgb_frame, padding, pad_color=(255, 255, 255)):
    """
    Pads an image with a specified color.

    :param rgb_frame: The image to pad
    :type rgb_frame: numpy.ndarray
    :param padding: The amount of padding to add to the image. Can have multiple formats, see standardize_padding() for
    details.
    :type padding: tuple of (int or float)
    :param pad_color: The color to pad the frame with
    :type pad_color: tuple
    :return: The padded image
    :rtype: numpy.ndarray
    """

    padding = standardize_padding(padding)
    return cv2.copyMakeBorder(rgb_frame, padding[0], padding[1], padding[2], padding[3], cv2.BORDER_CONSTANT,
                              value=pad_color)


def merge_images(*frames, positions=None, pad_color=(255, 255, 255)):
    """
    Merges multiple RGBA frames into a single frame. The frames are placed at the specified positions in the resulting
    frame. If there is empty space due to the size of the frames, then the empty space is filled with the specified
    pad_color.

    :param frames: The frames to merge
    :type frames: numpy.ndarray
    :param positions: The positions of the frames in the resulting frame
    :type positions: list of tuple of (int, int)
    :param pad_color: The color to fill empty space with
    :type pad_color: tuple
    :return: The merged image
    :rtype: numpy.ndarray
    """

    if len(frames) == 0:
        raise ValueError("At least one image must be specified!")
    if positions is not None and len(frames) != len(positions):
        raise ValueError("The number of images and the number of positions must be the same!")
    if positions is None:
        _image_widths = [frame.shape[1] for frame in frames]
        positions = [(sum(_image_widths[:i]), 0) for i in range(len(_image_widths))]

    max_x = max([pos[0] + frame.shape[1] for frame, pos in zip(frames, positions)])
    max_y = max([pos[1] + frame.shape[0] for frame, pos in zip(frames, positions)])

    merged_frame = np.full((max_y, max_x, 3), pad_color, dtype=np.uint8)

    for frame, pos in zip(frames, positions):
        x, y = pos
        if frame.shape[2] == 4:
            # If the frame has an alpha channel, blend it with the merged frame
            merged_frame = blend_bgr_and_bgra_image(merged_frame, frame, x, y)
        else:
            # If the frame does not have an alpha channel, simply place it on the merged frame
            merged_frame[y:y + frame.shape[0], x:x + frame.shape[1]] = frame

    return merged_frame


def concat_images(*frames, append_bottom=False, pad_color=(255, 255, 255)):
    """
    Concatenates multiple RGB frames into a single frame. The frames are placed one after the other in the resulting
    frame. If the frames have different sizes, the empty space is filled with the specified pad_color.

    :param frames: The frames to concatenate
    :type frames: numpy.ndarray
    :param append_bottom: If True, the frames are appended to the bottom of the resulting frame. If False, the frames
    are appended to the right of the resulting frame.
    :type append_bottom: bool
    :param pad_color: The color to fill empty space with
    :type pad_color: tuple
    :return: The concatenated image
    :rtype: numpy.ndarray
    """

    if len(frames) == 0:
        raise ValueError("At least one image must be specified!")
    else:
        if append_bottom:
            positions = [(0, sum([frame.shape[0] for frame in frames[:i]])) for i in range(len(frames))]
        else:
            positions = None
        return merge_images(*frames, positions=positions, pad_color=pad_color)


def stitch_image_list(*frames, rows=None, columns=None, pad_color=(255, 255, 255)):
    """
    Stitches multiple RGB frames into a single frame. The frames are placed in a grid in the resulting frame. The number
    of rows and columns in the grid can be specified. If the number of rows or columns is None, the number of rows or
    columns is determined automatically based on the number of frames.

    :param frames: The frames to stitch
    :type frames: numpy.ndarray
    :param rows: The number of rows in the grid
    :type rows: int or None
    :param columns: The number of columns in the grid
    :type columns: int or None
    :param pad_color: The color to fill empty space with
    :type pad_color: tuple
    :return: The stitched image
    :rtype: numpy.ndarray
    """

    if rows is None and columns is None:
        raise ValueError("At least one of rows and columns must be specified!")
    if rows is None:
        rows = len(frames) // columns + (columns % len(frames) > 0)
    if columns is None:
        columns = len(frames) // rows + (rows % len(frames) > 0)

    _row_images = []
    for _, image_row in zip(range(rows), split_chunks(frames, columns)):
        _row_images.append(concat_images(*image_row, append_bottom=False, pad_color=pad_color))

    merged_image = concat_images(*_row_images, append_bottom=True, pad_color=pad_color)
    return merged_image


def add_text_to_frame(frame, text,
                      relative_org=(0.05, 0.9), font=cv2.FONT_HERSHEY_COMPLEX,
                      font_scale=1.1, color=(255, 255, 255), thickness=2,
                      line_type=cv2.LINE_AA, bottom_left_origin=False, line_spacing=1.5,
                      relative_x=None, relative_y=None):
    frame_copy = frame.copy()
    height, width, depth = frame.shape
    if relative_x is not None and relative_y is not None:
        relative_org = (relative_x, relative_y)
    absolute_org = (int(width * relative_org[0]), int(height * relative_org[1]))

    # https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    for line in text.splitlines():
        (text_width, text_height), _ = cv2.getTextSize(text=line,
                                                       fontFace=font,
                                                       fontScale=font_scale,
                                                       thickness=thickness
                                                       )
        line_org = absolute_org[0], absolute_org[1] + text_height
        cv2.putText(frame_copy, line, line_org,
                    font, font_scale, color, thickness, line_type, bottom_left_origin)
        absolute_org = absolute_org[0], absolute_org[1] + int(text_height * line_spacing)
    return frame_copy


def add_rgba_overlay_to_frame(frame, rgba_overlay, use_alpha_composite=True):
    """
    Adds an rgba overlay to a frame.

    :param frame: The frame to add the overlay to.
    :type frame: np.ndarray
    :param rgba_overlay: The overlay to add to the frame. The overlay must be in rgba format.
    :type rgba_overlay: np.ndarray
    :param use_alpha_composite: If True, the alpha channel of the overlay is used to blend the overlay with the frame.
        If False, the alpha channel is ignored and the overlay is simply added to the frame, while the frame is masked
        with the inverse of the alpha channel. This masking means that the alpha channel is either 0 or 255. This
        results in a faster execution, as the combination is done with cv2.bitwise_and and cv2.bitwise_or instead of
        blend_bgr_and_bgra_image.
    :type use_alpha_composite: bool
    :return: The frame with the overlay added.
    :rtype: np.ndarray
    """
    frame = frame.copy()
    b, g, r, a = cv2.split(cv2.cvtColor(rgba_overlay, cv2.COLOR_RGBA2BGRA))

    if not use_alpha_composite:
        bgr_overlay = cv2.merge((b, g, r))
        bgr_overlay = cv2.bitwise_and(bgr_overlay, bgr_overlay, mask=(a != 0).astype("uint8"))
        frame = cv2.bitwise_and(frame, frame, mask=(a == 0).astype("uint8"))
        return cv2.addWeighted(frame, 1, bgr_overlay, 1, 0)
    else:
        bgra_overlay = cv2.merge((b, g, r, a))
        return blend_bgr_and_bgra_image(frame, bgra_overlay)


def add_annotations_to_frame(frame, *function_args, text_overlays=None, text_overlay_functions=None, rgba_overlays=None,
                             rgba_overlay_functions=None, use_alpha_composite=False, **function_kwargs):
    """
    Adds text and rgba overlays to a frame. The overlays can be either static or dynamic. If they are dynamic, they
    are expected to be functions that take the frame as input and return the overlay. Additionally, the functions can
    take *args and **kwargs as input.

    :param frame: The frame to add the overlays to.
    :type frame: np.ndarray
    :param function_args: Additional arguments for the functions.
    :type function_args: Any
    :param text_overlays: A list of kwarg-dicts or a single dict for add_text_to_frame.
    :type text_overlays: list[dict[str, Any]] or dict[str, Any]
    :param text_overlay_functions: A list of functions or a single function that return kwarg-dicts for
        add_text_to_frame.
    :type text_overlay_functions: list[function] or function
    :param rgba_overlays: A list of kwarg-dicts or a single dict for add_rgba_overlay_to_frame.
    :type rgba_overlays: list[dict[str, Any]] or dict[str, Any]
    :param rgba_overlay_functions: A list of functions or a single function that return kwarg-dicts for
        add_rgba_overlay_to_frame.
    :type rgba_overlay_functions: list[function] or function
    :param function_kwargs: Additional keyword arguments for the functions.
    :type function_kwargs: Any
    :param use_alpha_composite: Whether to use alpha compositing for the rgba overlays.
    :type use_alpha_composite: bool
    :return: The frame with the overlays added.
    :rtype: np.ndarray
    """
    static_texts = ensure_list(text_overlays) if text_overlays is not None else []
    function_texts = ensure_list(text_overlay_functions) if text_overlay_functions is not None else []
    static_overlays = ensure_list(rgba_overlays) if rgba_overlays is not None else []
    function_overlays = ensure_list(rgba_overlay_functions) if rgba_overlay_functions is not None else []

    for static_text in static_texts:
        frame = add_text_to_frame(frame, **static_text)
    for function_text in function_texts:
        frame = add_text_to_frame(frame, **function_text(*function_args, **function_kwargs))
    for static_overlay in static_overlays:
        frame = add_rgba_overlay_to_frame(frame, static_overlay, use_alpha_composite=use_alpha_composite)
    for function_overlay in function_overlays:
        frame = add_rgba_overlay_to_frame(frame, function_overlay(*function_args, **function_kwargs),
                                          use_alpha_composite=use_alpha_composite)

    return frame


def apply_functions_to_frame(frame, frame_functions=None, *args, **kwargs):
    """
    Applies a list of functions to a frame.

    :param frame: The frame to modify.
    :type frame: np.ndarray
    :param frame_functions: A list of functions or a single function that modify the frame.
    :type frame_functions: list[function] or function
    :param args: Additional arguments for the functions.
    :type args: Any
    :param kwargs: Additional keyword arguments for the functions.
    :type kwargs: Any
    :return: The modified frame.
    :rtype: np.ndarray
    """
    functions = ensure_list(frame_functions) if frame_functions is not None else []
    for function in functions:
        frame = function(frame, *args, **kwargs)
    return frame


def rotate_frame(frame, angle, rotation_center=None):
    """
    Rotates a frame by a specified angle. Adapted from https://stackoverflow.com/a/9042907.
    :param frame: The frame to rotate.
    :type frame: np.ndarray
    :param angle: The angle to rotate the frame by in degrees.
    :type angle: float
    :param rotation_center: The center of rotation. If None, the center of the frame is used.
    :type rotation_center: tuple[int]
    :return: The rotated frame.
    :rtype: np.ndarray
    """
    if rotation_center is None:
        rotation_center = tuple(np.array(frame.shape[1::-1]) / 2)  # center of image
    rotation_center = (float(rotation_center[0]), float(rotation_center[1]))
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
    return cv2.warpAffine(frame, rotation_matrix, frame.shape[1::-1], flags=cv2.INTER_LINEAR)


def correct_frame_orientation(frame, orientation, target_orientation=0, rotation_center=None):
    """
    Corrects the orientation of a frame. The frame is rotated by the difference between the orientation and the target
    orientation.

    :param frame: The frame to correct the orientation of.
    :type frame: np.ndarray
    :param orientation: The orientation of the frame.
    :type orientation: int
    :param target_orientation: The target orientation of the frame.
    :type target_orientation: int
    :return: The frame with the corrected orientation.
    :rtype: np.ndarray
    """
    return rotate_frame(frame=frame, angle=orientation+target_orientation, rotation_center=rotation_center)


def extract_reoriented_roi_around_point(frame, center_point, roi_shape, roi_orientation, target_orientation=0,
                                        **pad_kwargs):
    """
    Extracts a reoriented region of interest (ROI) from a frame, centered around a specified point. The ROI is extracted
    as a rectangle with the specified shape. The orientation of the ROI is corrected to the target orientation.

    :param frame: The frame to extract the ROI from.
    :type frame: np.ndarray
    :param center_point: The center point of the ROI.
    :type center_point: tuple[int]
    :param roi_shape: The shape of the ROI.
    :type roi_shape: tuple[int]
    :param roi_orientation: The current orientation of the ROI.
    :type roi_orientation: int
    :param target_orientation: The target orientation of the ROI.
    :type target_orientation: int
    :param pad_kwargs: Keyword arguments for the np.pad function.
    :type pad_kwargs: dict[str, Any]
    :return: The reoriented ROI extracted from the frame.
    :rtype: np.ndarray
    """
    reoriented_frame = correct_frame_orientation(frame, roi_orientation, target_orientation,
                                                 rotation_center=center_point)
    roi_bbox = (center_point[0] - roi_shape[0] // 2, center_point[1] - roi_shape[1] // 2, roi_shape[0], roi_shape[1])
    return get_padded_roi_from_frame(reoriented_frame, roi_bbox, **pad_kwargs)


def fit_arena_to_bbox(arena_polygon_dict, bbox_to_fit):
    from shapely.ops import unary_union
    from shapely.affinity import scale, translate
    from shapely.geometry import Point

    bbox_array = np.array(bbox_to_fit)
    bbox_xy, bbox_wh = bbox_array[:2], bbox_array[2:]

    arena_union = unary_union(list(arena_polygon_dict.values()))
    arena_bounds = arena_union.bounds
    arena_xy, arena_wh = np.array(arena_bounds[:2]), np.array(arena_bounds[2:])

    translate_xy = bbox_xy - arena_xy
    scale_wh = bbox_wh / arena_wh

    translated_arena_dict = {k: translate(v, *translate_xy) for k, v in arena_polygon_dict.items()}
    scaled_arena_dict = {k: scale(v, *scale_wh, origin=Point(bbox_xy)) for k, v in translated_arena_dict.items()}
    return scaled_arena_dict


def perspective_transform_polygon(input_polygon, perspective_matrix):
    from shapely.geometry import Polygon
    coordinates = input_polygon.exterior.coords.xy
    coordinate_array = np.stack(coordinates).T[np.newaxis]

    transformed_coordinate_array = cv2.perspectiveTransform(coordinate_array, perspective_matrix)[0]
    return Polygon(transformed_coordinate_array)


def perspective_transform_arena(arena_polygon_dict, perspective_matrix):
    return {k: perspective_transform_polygon(v, perspective_matrix) for k,v in arena_polygon_dict.items()}


def get_cv2_video_properties(video_path, *cv2_cap_props):
    """
    Gets the properties of a video file using cv2.VideoCapture.

    :param video_path: The path to the video file.
    :type video_path: str
    :param cv2_cap_props: The properties to get. Can be any of the cv2.CAP_PROP_* constants or their integer values.
    :type cv2_cap_props: int
    :return: The properties of the video file.
    :rtype: list[Any]
    """

    cap = cv2.VideoCapture(video_path)
    properties = [cap.get(prop) for prop in cv2_cap_props]
    cap.release()

    return properties


def rescale_frame(frame, scale_factor=1.0):
    """
    Rescales a frame to the specified scale factor.

    :param frame: A frame
    :type frame: np.ndarray
    :param scale_factor: The scale factor to use.
    :type scale_factor: float
    :return: The rescaled frame.
    :rtype: np.ndarray
    """
    width = int(frame.shape[1] * scale_factor)
    height = int(frame.shape[0] * scale_factor)
    resized_frame = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
    return resized_frame
