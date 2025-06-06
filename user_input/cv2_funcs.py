import cv2
import pandas as pd
from ..utility.builtin_classes.funcs import apply_filtered_parameters
from ..utility.builtin_classes.iterables import ensure_list

from ..utility.general import standardize_padding
from ..frame_pipeline.matplotlib_funcs import fig2cv2
from ..frame_pipeline.cv2_funcs import add_text_to_frame


def select_markers_gui(frame, marker_n_or_names=1, frame_padding=(0, 0, 0, 0), auto_close=True, **marker_kwargs):
    """
    Opens a window displaying a frame and prompts the us  er to select a specified number of points on the frame. The
    points are selected by left-clicking on the frame. Right-clicking on the frame removes the nearest point. Once all
    points have been selected, no further points can be selected and the window can be closed by pressing the escape or
    enter key. Any clicked points are marked by a circle on the frame. Circle_kwargs are passed to cv2.circle.

    :param frame: The frame to display.
    :type frame: np.ndarray
    :param marker_n_or_names: Either an integer specifying how many markers to select or a sequence of strings
    :type marker_n_or_names: int or list or tuple
    :param frame_padding: The amount of padding to add to the frame before displaying it. Can have multiple formats, see
    standardize_padding() for details.
    :type frame_padding: tuple of (int or float)
    :param auto_close: Whether to automatically close the window after all markers have been selected.
    :type auto_close: bool
    :param circle_kwargs: Keyword arguments for cv2.circle.
    :type circle_kwargs: dict[str, Any]
    :return: A list of the selected points.
    :rtype: list[tuple[int, int]]
    """

    input_is_int = isinstance(marker_n_or_names, int)
    input_is_basic_sequence = isinstance(marker_n_or_names, (tuple, list))

    assert input_is_int or input_is_basic_sequence, "marker_n_or_names must be an int or iterable"

    marker_names = marker_n_or_names if input_is_basic_sequence else [str(i) for i in range(marker_n_or_names)]
    selected_marker_count = 0
    current_marker = marker_names[0]
    marker_dict = {}

    pad_top, pad_bottom, pad_left, pad_right = standardize_padding(frame_padding)
    padded_frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    default_marker_kwargs = dict(color=(0, 255, 0))
    marker_kwargs = {**default_marker_kwargs, **marker_kwargs}

    window_name = "selection_window"

    def _mouse_click_event(event, x, y, *_):
        nonlocal current_marker, selected_marker_count

        left_clicked, right_clicked = event == cv2.EVENT_LBUTTONDOWN, event == cv2.EVENT_RBUTTONDOWN
        padded_x, padded_y = (x - pad_left, y - pad_top)
        selected_names, selected_positions = list(marker_dict.keys()), marker_dict.values()

        if left_clicked:
            if len(selected_names) < len(marker_names):
                marker_dict[current_marker] = (padded_x, padded_y)
        elif right_clicked:
            if len(marker_dict) > 0:
                distances = [((p[0] - padded_x) ** 2 + (p[1] - padded_y) ** 2) for p in selected_positions]
                closest_marker_index = distances.index(min(distances))
                marker_dict.pop(selected_names[closest_marker_index])

        if left_clicked or right_clicked:
            selected_names, selected_positions = list(marker_dict.keys()), marker_dict.values()

            current_frame = padded_frame.copy()
            for name, position in zip(selected_names, selected_positions):
                padded_position = position[0] + pad_left, position[1] + pad_top
                cv2.circle(current_frame, padded_position, radius=5, thickness=-1, **marker_kwargs)
                cv2.putText(current_frame, name, padded_position, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.0,
                            **marker_kwargs)
            cv2.imshow(window_name, current_frame)

            selected_marker_count = len(marker_dict)
            if selected_marker_count == len(marker_names):
                cv2.setWindowTitle(window_name, "All markers have been selected. Press escape to exit.")
            else:
                current_marker = [m for m in marker_names if not m in marker_dict.keys()][0]
                cv2.setWindowTitle(window_name, "Select marker {}.".format(current_marker))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(window_name, padded_frame)
    cv2.setMouseCallback(window_name, _mouse_click_event)
    cv2.setWindowTitle(window_name, "Select marker {}.".format(current_marker))

    while True:
        wait_result = cv2.waitKey(10)
        if selected_marker_count == len(marker_names) and auto_close:
            break  # All markers have been selected

        if wait_result == 27 or wait_result == 13:  # escape or enter key pressed
            break

    cv2.destroyWindow(window_name)
    return marker_dict


def get_arena_perspective_transform(input_frame, arena_polygon_dict, reference_polygon=None, **selection_kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    from shapely.ops import unary_union
    from .matplotlib_funcs import create_canvas, multiply_ax_limits
    from .plots import plot_arena

    n_ref_points = 4

    if reference_polygon is None:
        arena_union = unary_union(list(arena_polygon_dict.values()))
        x, y, w, h = arena_union.bounds
    else:
        x, y, w, h = arena_polygon_dict[reference_polygon].bounds
    xx, yy = [x, x, w, w], [y, h, h, y]
    coordinate_array = np.stack([xx, yy])

    arena_canvas = create_canvas(*np.max(coordinate_array, axis=1), dpi=100, alpha=0)
    plot_arena(arena_polygon_dict, ax=arena_canvas.gca())
    multiply_ax_limits(arena_canvas.gca(), 1.5)
    for i, (x, y) in enumerate(coordinate_array.T[:n_ref_points]):
        arena_canvas.gca().text(x, y, str(i), color="red")

    src_window_name = "polygon_window"
    cv2.imshow(src_window_name, fig2cv2(arena_canvas))
    cv2.setWindowTitle(src_window_name, "Source Shape and Points")
    dst_points_dict = select_markers_gui(input_frame, marker_n_or_names=n_ref_points, **selection_kwargs)
    cv2.destroyWindow(src_window_name)
    plt.close(arena_canvas)

    src_points = coordinate_array.T[:n_ref_points].astype("float32")
    sorted_dst_points_dict = {k: dst_points_dict[k] for k in sorted(dst_points_dict.keys())}
    dst_points = np.stack(list(sorted_dst_points_dict.values())).astype("float32")

    return cv2.getPerspectiveTransform(src_points, dst_points)


def get_polygon_perspective_transform(input_frame, input_polygon):
    return get_arena_perspective_transform(input_frame, {"Input Polygon": input_polygon},
                                           reference_polygon="Input Polygon")


def image_tagger(image_paths, tag_dict, tag_df=None, annotation_df=None,
                 display_width=720,
                 multi_choice=True,
                 title="Tagging images", text_overlay_dict=None):
    if tag_df is None:
        output_df = ~pd.DataFrame(index=image_paths, columns=tag_dict.keys(), dtype=bool)
    else:
        assert (tag_df.index.to_list() == image_paths
                ) and (tag_df.columns.to_list() == list(tag_dict.keys())
                       ), "Input df has to match input paths and tags in index and columns."
        output_df = tag_df.copy()

    text_overlay_dict = dict() if text_overlay_dict is None else text_overlay_dict
    image_stack = [cv2.imread(image_path) for image_path in image_paths]
    for i, image in enumerate(image_stack):
        width, height, depth = image.shape
        scaling_factor = display_width / width
        image_stack[i] = cv2.resize(image, dsize=None, fx=scaling_factor, fy=scaling_factor)
    image_index = 0

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(title, "Tagging {}.".format(image_paths[image_index]))

    expand_gui = True
    while True:
        frame = image_stack[image_index].copy()

        if image_index in text_overlay_dict:
            static_texts = ensure_list(text_overlay_dict[image_index])
            for static_text in static_texts:
                frame = add_text_to_frame(frame, **static_text)

        image_tags = output_df.loc[image_paths[image_index]]
        if expand_gui:
            tag_string = "\'<\' / \'>\' previous/next, \'#\' to hide info\n"
            frame_string = "Frame index: " + str(image_index)
        else:
            tag_string = "\'#\' to show info\n"
            frame_string = str(image_index)
        for key, value in image_tags.to_dict().items():
            tag_string += f"{key}[\'{tag_dict[key]}\']: {value}\n" \
                if expand_gui else f"[\'{tag_dict[key]}\']: {int(value)}\n"
        frame = add_text_to_frame(frame, tag_string, relative_org=(0.05, 0.05), font_scale=0.5, color=(255, 255, 255))
        frame = add_text_to_frame(frame, frame_string, relative_org=(0.8 if expand_gui else 0.9, 0.05), font_scale=0.5,
                                  color=(255, 255, 255))

        if annotation_df is not None:
            try:
                annotation = annotation_df.loc[[image_paths[image_index]]]
                for _, text_series in annotation.iterrows():
                    text_dict = text_series.dropna().to_dict()
                    if "text" in text_dict.keys():
                        frame = apply_filtered_parameters(add_text_to_frame, frame=frame, **text_dict)
            except KeyError:
                pass

        cv2.imshow(title, frame)

        wait_result = cv2.waitKey(1)
        for tag, key in tag_dict.items():
            if wait_result == ord(key):
                if not multi_choice:
                    output_df.loc[image_paths[image_index]] = False
                output_df.loc[image_paths[image_index], tag] = ~output_df.loc[image_paths[image_index], tag]

        previous_frame, next_frame = wait_result == ord("<"), wait_result == ord(">")
        gui_expansion = wait_result == ord("#")
        if previous_frame or next_frame:
            if previous_frame:
                image_index = min(image_index + 1, len(image_stack) - 1)
            elif next_frame:
                image_index = max(image_index - 1, 0)
            cv2.setWindowTitle(title, "Tagging {}.".format(image_paths[image_index]))
        elif gui_expansion:
            expand_gui = not expand_gui
        elif wait_result == 27:  # Esc key to stop
            break

    cv2.destroyAllWindows()
    return output_df


def video_tagger(video_paths, tag_dict, tag_df=None, annotation_df=None,
                 display_width=720,
                 multi_choice=True,
                 title="Tagging videos", text_overlay_dict=None,
                 tb_update_freq=30):
    if tag_df is None:
        output_df = ~pd.DataFrame(index=video_paths, columns=tag_dict.keys(), dtype=bool)
    else:
        assert (tag_df.index.to_list() == video_paths
                ) and (tag_df.columns.to_list() == list(tag_dict.keys())
                       ), "Input df has to match input paths and tags in index and columns."
        output_df = tag_df.copy()

    text_overlay_dict = dict() if text_overlay_dict is None else text_overlay_dict
    video_index = 0

    def set_frame(frame_index):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setWindowTitle(title, "Tagging {}.".format(video_paths[video_index]))
    video = cv2.VideoCapture(video_paths[video_index])
    nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if nr_of_frames == 0:
        raise ValueError("Video {} contains zero frames!".format(video_paths[video_index]))
    scaling_factor = display_width / video.get(cv2.CAP_PROP_FRAME_WIDTH)
    cv2.createTrackbar("Frame", title, 0, nr_of_frames, set_frame)

    tb_update_count = 0
    previous_frame = [[0]]

    expand_gui = True
    while True:
        ret, frame = video.read()
        if not ret:
            frame = previous_frame.copy()
        else:
            previous_frame = frame.copy()

        frame = cv2.resize(frame, dsize=None, fx=scaling_factor, fy=scaling_factor)

        if video_index in text_overlay_dict:
            static_texts = ensure_list(text_overlay_dict[video_index])
            for static_text in static_texts:
                frame = add_text_to_frame(frame, **static_text)

        video_tags = output_df.loc[video_paths[video_index]]

        if expand_gui:
            tag_string = "\'<\' / \'>\' previous/next, \'#\' to hide info, use \'ESC\' to close!\n"
            video_string = "Video index: " + str(video_index)
        else:
            tag_string = "\'#\' to show info\n"
            video_string = str(video_index)
        for key, value in video_tags.to_dict().items():
            tag_string += f"{key}[\'{tag_dict[key]}\']: {value}\n" \
                if expand_gui else f"[\'{tag_dict[key]}\']: {int(value)}\n"
        frame = add_text_to_frame(frame, tag_string, relative_org=(0.05, 0.05), font_scale=0.5, color=(255, 255, 255))
        frame = add_text_to_frame(frame, video_string, relative_org=(0.8 if expand_gui else 0.9, 0.05), font_scale=0.5,
                                  color=(255, 255, 255))

        if annotation_df is not None:
            try:
                annotation = annotation_df.loc[[video_paths[video_index]]]
                for _, text_series in annotation.iterrows():
                    text_dict = text_series.dropna().to_dict()
                    frame_start = text_dict.pop("frame_start") if "frame_start" in text_dict else 0
                    frame_end = text_dict.pop("frame_end") if "frame_end" in text_dict else (nr_of_frames + 1)
                    if frame_start <= int(
                            video.get(cv2.CAP_PROP_POS_FRAMES)) < frame_end and "text" in text_dict.keys():
                        frame = apply_filtered_parameters(add_text_to_frame, frame=frame, **text_dict)
            except KeyError:
                pass

        cv2.imshow(title, frame)

        wait_result = cv2.waitKey(1)
        for tag, key in tag_dict.items():
            if wait_result == ord(key):
                if not multi_choice:
                    output_df.loc[video_paths[video_index]] = False
                output_df.loc[video_paths[video_index], tag] = ~output_df.loc[video_paths[video_index], tag]

        previous_video, next_video = wait_result == ord("<"), wait_result == ord(">")
        gui_expansion = wait_result == ord("#")
        if previous_video or next_video:
            if previous_video:
                video_index = min(video_index + 1, len(video_paths) - 1)
            elif next_video:
                video_index = max(video_index - 1, 0)
            video.release()
            video = cv2.VideoCapture(video_paths[video_index])
            nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if nr_of_frames == 0:
                raise ValueError("Video {} contains zero frames!".format(video_paths[video_index]))
            scaling_factor = display_width / video.get(cv2.CAP_PROP_FRAME_WIDTH)

            cv2.setWindowTitle(title, "Tagging {}.".format(video_paths[video_index]))
            cv2.setTrackbarMax("Frame", title, nr_of_frames)
            cv2.setTrackbarPos("Frame", title, 0)

            tb_update_count = -1

        elif gui_expansion:
            expand_gui = not expand_gui
        elif wait_result == 27:  # Esc key to stop
            break

        tb_update_count += 1
        if tb_update_freq is not None and tb_update_count >= tb_update_freq:
            tb_update_count = 0
            cv2.setTrackbarPos("Frame", title, int(video.get(cv2.CAP_PROP_POS_FRAMES)))

    cv2.destroyAllWindows()
    return output_df
