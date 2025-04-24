import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ..utility.builtin_classes.iterables import ensure_list, multi_zip_unequal
from ..utility.cli_tools.powershell import run_command_in_powershell
from ..utility.files.file_tools import ensure_directory

from .cv2_funcs import add_text_to_frame, add_rgba_overlay_to_frame, add_annotations_to_frame, \
    apply_functions_to_frame, get_padded_roi_from_frame, extract_reoriented_roi_around_point
from .frame_funcs import get_roi_from_frame
from ..visualization.pil_funcs import stitch_image_list, ensure_pil_image, merge_images

default_fourcc = "H264"  # default TopScan encoding


def get_fourcc(cap_or_video_path):
    """
    A function that returns the fourcc (four character code) of a video as a string.
    The byteorder is checked on the system to ensure that the correct fourcc is returned.
    For more information see https://stackoverflow.com/a/76008953.

    :param cap_or_video_path: A cv2.VideoCapture object or a path to a video file.
    :type cap_or_video_path: cv2.VideoCapture or str
    :return: The fourcc of the video as a string.
    :rtype: str
    """
    if isinstance(cap_or_video_path, str):
        cap = cv2.VideoCapture(cap_or_video_path)
    else:
        cap = cap_or_video_path
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, byteorder=sys.byteorder).decode()
    if isinstance(cap_or_video_path, str):
        cap.release()
    return fourcc


def skip_frames(cv2_video_capture, frames_to_skip):
    """

    :param cv2_video_capture:
    :type cv2_video_capture:
    :param frames_to_skip:
    :type frames_to_skip:
    :return:
    :rtype:
    """
    for i in range(frames_to_skip):
        ret = cv2_video_capture.grab()
        if not ret:
            break


def count_frames_iteratively(video_path):
    """
    Counts the number of frames in a video file by iterating over the frames.

    :param video_path: The path to the video file.
    :type video_path: str
    :return: The number of frames in the video file.
    :rtype: int
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        frame_count += 1
    cap.release()
    return frame_count


def cv2_frame_reader(video_path, start_frame=0, end_frame=None, start_end_speed_sequence=None,
                     target_fps=None, read_speed=1, yield_index=True, jump_to_start=False, frame_idx_transformer=None):
    """
    A generator that yields frames from a video file. Adapted from https://stackoverflow.com/a/69312152

    :param video_path: A path to a video file.
    :type video_path: str
    :param start_frame: The index of the first frame to read. Default is 0.
    :type start_frame: int
    :param end_frame: The index of the last frame to read. If None, the last frame of the video is read.
    :type end_frame: int
    :param start_end_speed_sequence: A list of tuples that specify the start and end frame of a frame sequence and the
        read_speed of the sequence. This is particularly useful for videos that contain a slow motion sequence. The
        contained tuples are passed to cv2_sequence_reader.
    :type start_end_speed_sequence: tuple of (int, int, float)
    :param target_fps: The target fps of the video. If None, the original fps of the video is used.
    :type target_fps: float
    :param read_speed: The speed at which the frames are read. If 1, the frames are read at the target_fps. If 2, the
        frames are read at twice the target_fps. If 0.5, reads the frames at half the target_fps.
    :type read_speed: float
    :param yield_index: If True, yields the index of the frame together with the frame.
    :type yield_index: bool
    :param jump_to_start: If True, jumps the video to the start_frame. If False, the video is read from the
        beginning and the first frames are skipped until the start_frame. If frames are missing, this will
        not work as expected.
    :type jump_to_start: bool
    :param frame_idx_transformer: A function that transforms the frame index. This is useful if the frame index is
        different from the frame number. For example, if the frame index is the timestamp of the frame, or if the frame
        index is shifted due to missing frames. This has no effect if yield_index is False.
    :type frame_idx_transformer: function or callable
    :return: A generator that yields frames from a video file.
    :rtype: (int, np.ndarray) or np.ndarray
    """
    if start_end_speed_sequence is not None:
        yield from cv2_sequence_reader(video_path, start_end_speed_sequence, yield_index=yield_index)
        return

    cap = cv2.VideoCapture(video_path)

    try:
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        fps_out = read_speed * (target_fps if target_fps is not None else fps_in)

        if frame_idx_transformer is not None:
            start_frame = frame_idx_transformer(start_frame, invert=True)

        if not jump_to_start:
            skip_frames(cap, start_frame * int((target_fps / fps_in) if target_fps is not None else 1))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame * int((target_fps / fps_in) if target_fps is not None else 1))
        end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if end_frame is None else end_frame

        if frame_idx_transformer is not None:
            end = frame_idx_transformer(end, invert=True)

        index_out = index_in = start_frame

        while index_in < end:
            out_due = int((index_in - start_frame) / fps_out * fps_in) + start_frame
            if out_due > index_out:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                index_out += 1
                yield (frame_idx_transformer(index_in) if frame_idx_transformer is not None else index_in,
                       frame) if yield_index else frame
                continue

            ret = cap.grab()
            if not ret:
                break
            index_in += 1
    finally:
        cap.release()


def frame_roi_extractor(cv2_frame_generator, roi_series):
    for frame_index, frame in cv2_frame_generator:
        x, y, w, h = roi_series.loc[frame_index]
        yield frame_index, frame[y:y + h, x:x + w]


def cv2_sequence_reader(video_path, start_end_speed_sequence, yield_index=True):
    """
    A generator that yields frames from a video file. The frames are read from a sequence of frames with different
    read_speeds.

    :param video_path:
    :type video_path:
    :param start_end_speed_sequence:
    :type start_end_speed_sequence:
    :param yield_index:
    :type yield_index:
    :return:
    :rtype:
    """
    start_end_speed_sequence = ensure_list(start_end_speed_sequence)
    for start_frame, end_frame, read_speed in start_end_speed_sequence:
        yield from cv2_frame_reader(video_path, start_frame=start_frame, end_frame=end_frame, read_speed=read_speed,
                                    yield_index=yield_index)


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


def annotated_video_frames_generator(video_path, initial_modification_functions=None, reader_kwargs=None,
                                     show_progress=True, annotator_video_shift=0, roi_series=None,
                                     extract_roi_first=False, heading_series=None, final_modification_functions=None,
                                     **annotator_kwargs):
    """
    Generates annotated and modified video frames from a video file. Modifications are given through
    initial_modification_functions, final_modification_functions, and annotator_kwargs. The
    annotator_kwargs are passed to the add_annotations_to_frame function. See the documentation for
    add_annotations_to_frame for more information on the annotator_kwargs.

    :param video_path: The path to the video file to generate the frames from.
    :type video_path: str
    :param initial_modification_functions: A list of functions or a single function that modify the frame before any
        annotations are added.
    :type initial_modification_functions: list[function] or function
    :param reader_kwargs: Keyword arguments for the frame reader.
    :type reader_kwargs: dict[str, Any]
    :param show_progress: Whether to show a progress bar.
    :type show_progress: bool
    :param annotator_video_shift: An integer indicating the relative shift between the annotator funcs and the input
    video. This is helpful if one has a large dataset spanning multiple videos (and thus video frame 0 being not the
    same as data frame 0) or a long video where the dataset only describes a small segment. E.g. if the video starts
    after the dataset, this must be negative, or, if the dataset starts in the middle of the video, positive.
    :type annotator_video_shift: int
    :param roi_series: A single roi_series to cut the frame before applying the annotations.
    :type roi_series: pd.Series
    :param final_modification_functions: A list of functions or a single function that modify the frame after all
    :type final_modification_functions: list[function] or function
    :param annotator_kwargs: Keyword arguments for the add_annotations_to_frame function. See the documentation for
        add_annotations_to_frame for more information.
    :type annotator_kwargs: dict[str, Any]
    :yield: The annotated and modified frame.
    """

    def _apply_roi_series_and_heading(corrected_frame_index, frame, roi_series=None, heading_series=None):
        if roi_series is not None and not roi_series.loc[corrected_frame_index].isna().any():
            x, y, w, h = roi_series.loc[corrected_frame_index].astype(int)
            if heading_series is not None and not pd.isna(heading_series.loc[corrected_frame_index]):
                roi_center, roi_shape = (int(x + w / 2), int(y + h / 2)), (w, h)
                roi_heading = heading_series.loc[corrected_frame_index]
                return extract_reoriented_roi_around_point(frame, roi_center, roi_shape, roi_heading)
            else:
                return get_padded_roi_from_frame(frame, (x, y, w, h))
        return frame

    reader_kwargs = dict() if reader_kwargs is None else reader_kwargs

    total_frames = int(get_cv2_video_properties(video_path, cv2.CAP_PROP_FRAME_COUNT)[0])

    frame_reader = cv2_frame_reader(video_path, **reader_kwargs)

    start = reader_kwargs["start_frame"] if "start_frame" in reader_kwargs else 0
    end = reader_kwargs["end_frame"] if "end_frame" in reader_kwargs else total_frames
    frames_to_read = end - start - 1
    frame_iterator = tqdm(frame_reader, total=frames_to_read if "read_speed" not in reader_kwargs else int(
        frames_to_read / reader_kwargs["read_speed"])) if show_progress else frame_reader

    for frame_index, frame in frame_iterator:
        corrected_frame_index = frame_index + annotator_video_shift
        frame = apply_functions_to_frame(frame, frame_functions=initial_modification_functions)

        if extract_roi_first:
            frame = _apply_roi_series_and_heading(corrected_frame_index, frame, roi_series=roi_series,
                                                  heading_series=heading_series)
            frame = add_annotations_to_frame(frame, corrected_frame_index, **annotator_kwargs)
        else:
            frame = add_annotations_to_frame(frame, corrected_frame_index, **annotator_kwargs)
            frame = _apply_roi_series_and_heading(corrected_frame_index, frame, roi_series=roi_series,
                                                  heading_series=heading_series)

        frame = apply_functions_to_frame(frame, frame_functions=final_modification_functions)

        yield frame


def write_video_to_file(output_avi_path, video_path, target_fps=None, fourcc=None, roi_series=None, **generator_kwargs):
    """
    Writes a video to a file. The video can be modified by applying modifications to the frames. The modifications are
    applied by the annotated_video_frames_generator function. See the annotated_video_frames_generator function for
    more information.

    :param output_avi_path: The path to the output video file.
    :type output_avi_path: str
    :param video_path: The path to the input video file.
    :type video_path: str
    :param target_fps: The target fps of the output video file. If None, the fps of the input video file is used.
    :type target_fps: int
    :param fourcc: The fourcc code of the output video file. If None, the default fourcc code is used.
    :type fourcc: str
    :param roi_series: A single roi_series to cut the frame before applying the annotations.
    :type roi_series: pd.Series
    :param generator_kwargs: Additional keyword arguments for the annotated_video_frames_generator function. See the
        annotated_video_frames_generator function for more information.
    :type generator_kwargs: Any
    :rtype: None
    """
    width, height, fps = get_cv2_video_properties(video_path, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                                                  cv2.CAP_PROP_FPS)
    if roi_series is not None:
        generator_kwargs["roi_series"] = roi_series
        assert len(roi_series["w"].unique()) == 1 and len(roi_series["h"].unique()) == 1, \
            "All roi_series must have the same width and height."
        width, height = roi_series["w"].iloc[0], roi_series["h"].iloc[0]
    width, height, fps = int(width), int(height), int(fps) if target_fps is None else target_fps
    fourcc = fourcc if fourcc is not None else default_fourcc
    video_writer = cv2.VideoWriter(output_avi_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))

    frame_iterator = annotated_video_frames_generator(video_path, **generator_kwargs)

    try:
        for annotated_frame in frame_iterator:
            video_writer.write(annotated_frame)
    except KeyboardInterrupt:
        pass
    finally:
        video_writer.release()


def write_stitched_video_to_file(output_avi_path, video_path_list, rows=None, columns=None, pad_color=(255, 255, 255),
                                 target_fps=None, fourcc=None, generator_kwargs_dict=None, **generator_kwargs):
    """
    Uses the stitch_image_list function to stitch the frames of multiple videos together and writes the stitched video
    to a file. The videos can be modified by applying modifications to the frames. The modifications are applied by the
    annotated_video_frames_generator function. See the annotated_video_frames_generator function for more information.
    The generator_kwargs_dict is a dictionary that maps video paths to keyword arguments for the
    annotated_video_frames_generator function and overrides the generator_kwargs argument for that
    specific video.

    :param output_avi_path: The path to the output video file.
    :type output_avi_path: str
    :param video_path_list: A list of paths to the input video files.
    :type video_path_list: list[str]
    :param rows: The number of rows of the stitched video.
    :type rows: int
    :param columns: The number of columns of the stitched video.
    :type columns: int
    :param pad_color: The color of the padding between the videos.
    :type pad_color: tuple[int]
    :param target_fps: The target fps of the output video file. If None, the fps of the input video files is used.
    :type target_fps: int
    :param fourcc: The fourcc code of the output video file. If None, the default fourcc code is used.
    :type fourcc: str
    :param generator_kwargs_dict: A dictionary that maps video paths to keyword arguments for the
        annotated_video_frames_generator function. The keyword arguments override the generator_kwargs argument for that
        specific video.
    :type generator_kwargs_dict: dict[str, Any]
    :param generator_kwargs: Additional keyword arguments for the annotated_video_frames_generator function. See the
        annotated_video_frames_generator function for more information.
    :type generator_kwargs: Any
    :return: None
    """

    generator_kwargs_dict = {f: dict() for f in
                             video_path_list} if generator_kwargs_dict is None else generator_kwargs_dict

    # grab first frame of each video and calculate the size of the stitched frame
    first_frames = [ensure_pil_image(get_first_frame(video_path)) for video_path in video_path_list]
    _fill_frames = [ensure_pil_image(np.zeros_like(np.array(frame))) for frame in first_frames]
    for i, frame in enumerate(_fill_frames):
        frame.paste(pad_color, (0, 0, frame.width, frame.height))

    stitched_frames = stitch_image_list(*first_frames, rows=rows, columns=columns, pad_color=pad_color)
    width, height = stitched_frames.width, stitched_frames.height

    # check same fps for all videos
    fps_list = [int(get_cv2_video_properties(video_path, cv2.CAP_PROP_FPS)[0]) for video_path in video_path_list]
    if len(set(fps_list)) != 1 and target_fps is None:
        raise ValueError("All videos must have the same fps if target_fps is not specified.")
    fps = fps_list[0] if target_fps is None else target_fps

    fourcc = fourcc if fourcc is not None else default_fourcc
    video_writer = cv2.VideoWriter(output_avi_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))

    frame_iterator_list = []
    for video_path in video_path_list:
        video_generator_kwargs = {**generator_kwargs, **generator_kwargs_dict[video_path]}
        frame_iterator_list.append(annotated_video_frames_generator(video_path, **video_generator_kwargs))

    try:
        frame_list_list = [list(frame_iterator) for frame_iterator in frame_iterator_list]
        zipped_frame_iterator_list = multi_zip_unequal(*frame_list_list, length_limit_index=None,
                                                       fill_value=_fill_frames)
        for annotated_frames in zipped_frame_iterator_list:
            annotated_frames = [ensure_pil_image(frame) for frame in annotated_frames]
            stitched_frames = stitch_image_list(*annotated_frames, rows=rows, columns=columns,
                                                pad_color=pad_color)
            video_writer.write(np.array(stitched_frames))
    except KeyboardInterrupt:
        pass
    finally:
        video_writer.release()


def write_merged_video_to_file(output_avi_path, video_path_list, video_bbox_list, pad_color=(255, 255, 255),
                               video_alpha_list=None, target_fps=None, fourcc=None, generator_kwargs_dict=None,
                               length_limit_video_index=None, **generator_kwargs):
    """
    Uses the stitch_image_list function to stitch the frames of multiple videos together and writes the stitched video
    to a file. The videos can be modified by applying modifications to the frames. The modifications are applied by the
    annotated_video_frames_generator function. See the annotated_video_frames_generator function for more information.
    The generator_kwargs_dict is a dictionary that maps video paths to keyword arguments for the
    annotated_video_frames_generator function and overrides the generator_kwargs argument for that
    specific video.

    :param output_avi_path: The path to the output video file.
    :type output_avi_path: str
    :param video_path_list: A list of paths to the input video files.
    :type video_path_list: list[str]
    :param video_bbox_list: A list of bounding boxes for each video. The bounding boxes are given as a list of tuples
        (x, y, w, h) where x and y are the top left corner of the bounding box and w and h are the width and height of
        the bounding box.
    :type video_bbox_list: list[tuple[int]]
    :param pad_color: The color of the padding between the videos.
    :type pad_color: tuple[int]
    :param target_fps: The target fps of the output video file. If None, the fps of the input video files is used.
    :type target_fps: int
    :param fourcc: The fourcc code of the output video file. If None, the default fourcc code is used.
    :type fourcc: str
    :param generator_kwargs_dict: A dictionary that maps video paths to keyword arguments for the
        annotated_video_frames_generator function. The keyword arguments override the generator_kwargs argument for that
        specific video.
    :type generator_kwargs_dict: dict[str, Any]
    :param generator_kwargs: Additional keyword arguments for the annotated_video_frames_generator function. See the
        annotated_video_frames_generator function for more information.
    :type generator_kwargs: Any
    :return: None
    """

    generator_kwargs_dict = {f: dict() for f in
                             video_path_list} if generator_kwargs_dict is None else generator_kwargs_dict

    # grab first frame of each video and calculate the size of the stitched frame
    first_frames = [ensure_pil_image(get_first_frame(video_path)) for video_path in video_path_list]
    _fill_frames = [ensure_pil_image(np.zeros_like(np.array(frame))) for frame in first_frames]

    resized_frames = [frame.resize((w, h)) for frame, (x, y, w, h) in zip(_fill_frames, video_bbox_list)]
    top_left_coordinates = [(x, y) for x, y, _, _ in video_bbox_list]

    merged_frame = merge_images(*resized_frames, positions=top_left_coordinates, pad_color=pad_color)
    width, height = merged_frame.width, merged_frame.height

    # check same fps for all videos
    fps_list = [int(get_cv2_video_properties(video_path, cv2.CAP_PROP_FPS)[0]) for video_path in video_path_list]

    if len(set(fps_list)) != 1 and target_fps is None:
        raise ValueError("All videos must have the same fps if target_fps is not specified.")
    fps = fps_list[0] if target_fps is None else target_fps

    fourcc = fourcc if fourcc is not None else default_fourcc
    video_writer = cv2.VideoWriter(output_avi_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))

    frame_iterator_list = []
    for video_path in video_path_list:
        video_generator_kwargs = {**generator_kwargs, **generator_kwargs_dict[video_path]}
        frame_iterator_list.append(annotated_video_frames_generator(video_path, **video_generator_kwargs))

    try:
        frame_list_list = [list(frame_iterator) for frame_iterator in frame_iterator_list]
        zipped_frame_iterator_list = multi_zip_unequal(*frame_list_list,
                                                       length_limit_index=length_limit_video_index,
                                                       fill_value=_fill_frames)

        for annotated_frames in tqdm(zipped_frame_iterator_list):
            annotated_frames = [ensure_pil_image(frame) for frame in annotated_frames]

            if video_alpha_list is not None:
                for frame, alpha in zip(annotated_frames, video_alpha_list):
                    frame.putalpha(alpha)

            resized_frames = [frame.resize((w, h)) for frame, (x, y, w, h) in zip(annotated_frames, video_bbox_list)]
            top_left_coordinates = [(x, y) for x, y, _, _ in video_bbox_list]
            merged_frame = merge_images(*resized_frames, positions=top_left_coordinates, pad_color=pad_color)
            video_writer.write(np.array(merged_frame.convert("RGB")))
    except KeyboardInterrupt:
        pass
    finally:
        video_writer.release()


def get_frame_msec_series(video_path):
    """
    Extracts the frame positions and timestamps from a video and returns them as a pandas series.
    
    :param video_path: A path to a video
    :type video_path: str
    :return: A pandas series with the frame positions as index and the timestamps in milliseconds as values
    :rtype: pd.Series
    """
    cap = cv2.VideoCapture(video_path)

    frame_pos_list = []
    timestamp_list = []

    while True:
        ret = cap.grab()
        if not ret:
            break
        frame_pos_list.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        timestamp_list.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    return pd.Series(data=timestamp_list,
                     index=pd.Index(frame_pos_list, name="frame_index"),
                     name="frame_msec")


def get_frame_diff_series(video_path):
    cap = cv2.VideoCapture(video_path)

    try:
        frame_pos_list = []
        frame_diff_list = []
        previous_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_pos_list.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

            if previous_frame is not None:
                frame_diff_list.append((frame - previous_frame).sum(axis=None))
            else:
                frame_diff_list.append(np.nan)
            previous_frame = frame

        return pd.Series(data=frame_diff_list,
                         index=pd.Index(frame_pos_list, name="frame_index"),
                         name="frame_diff")

    finally:
        cap.release()


def extract_frame_msec_series_to_file(video_path_or_list, show_progress=True):
    """
    Extracts the frame positions and timestamps from a video or multiple videos and saves them to a csv file (each).

    :param video_path_or_list: Either a single video path or a list of video paths
    :type video_path_or_list: str or list[str]
    :param show_progress: Whether to show a progress bar
    :type show_progress: bool
    :rtype: None
    """
    video_path_list = ensure_list(video_path_or_list)
    video_path_iterator = tqdm(video_path_list) if show_progress else video_path_list
    for video_path in video_path_iterator:
        output_path = video_path + ".frame_msec_series.csv"
        if os.path.exists(output_path):
            continue

        frame_msec_series = get_frame_msec_series(video_path)
        frame_msec_series.to_csv(output_path)


def fix_frame_msec_series(frame_msec_series, remove_values=(0,)):
    """

    :param frame_msec_series:
    :type frame_msec_series:
    :param remove_values:
    :type remove_values:
    :return:
    :rtype:
    """
    return frame_msec_series.replace(remove_values, np.nan).interpolate(
        method="slinear", fill_value="extrapolate", limit_direction="both")


def count_frame_skips(msec_position_series, intended_frame_rate):
    """

    :param msec_position_series:
    :type msec_position_series:
    :param intended_frame_rate:
    :type intended_frame_rate:
    :return:
    :rtype:
    """
    return get_actual_frame_indices(msec_position_series, intended_frame_rate).index.to_series().diff().fillna(
        1).astype(int) - 1


def count_frame_skips_old(msec_position_series, intended_frame_rate, round_to_msec_decimal=1):
    return ((msec_position_series.diff().round(round_to_msec_decimal)
             ) // round(1 / intended_frame_rate * 1000, round_to_msec_decimal)).fillna(1, limit=1).astype(int) - 1


def get_video_shape(input_video_path):
    """

    :param input_video_path:
    :type input_video_path:
    :return:
    :rtype:
    """
    cap = cv2.VideoCapture(input_video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Unreadable video-file!")
    cap.release()
    return frame.shape


def get_actual_frame_indices(frame_msec_series, intended_frame_rate):
    fixed_frame_idx_array = (
            fix_frame_msec_series(frame_msec_series).values / (1 / intended_frame_rate * 1000)).round().astype(int)
    fixed_frame_idx = pd.Index(fixed_frame_idx_array, name="frame_index")
    return frame_msec_series.set_axis(fixed_frame_idx)


def get_frame_exists_series(frame_pos, intended_frame_rate):
    actual_frame_indices = get_actual_frame_indices(frame_pos, intended_frame_rate=intended_frame_rate)
    existing_frames_boolean = pd.Series(data=True, index=actual_frame_indices.index)
    all_frames_boolean = existing_frames_boolean.reindex(pd.RangeIndex(1, actual_frame_indices.index.max() + 1)).fillna(
        False)
    return all_frames_boolean


def get_frame_skipped_series_old(input_video_path, intended_frame_rate):
    frame_skips = count_frame_skips(fix_frame_msec_series(get_frame_msec_series(input_video_path)),
                                    intended_frame_rate)
    frame_skipped_list = []
    for skipped_count in frame_skips:
        for _ in range(skipped_count):
            frame_skipped_list.append(True)
        frame_skipped_list.append(False)
    return pd.Series(frame_skipped_list, index=range(1, len(frame_skipped_list) + 1))


def get_frame_skipped_series(msec_position_series, intended_frame_rate):
    """
    :param msec_position_series:
    :type msec_position_series:
    :param intended_frame_rate:
    :type intended_frame_rate:
    :return:
    :rtype:
    """
    real_frame_pos = get_actual_frame_indices(msec_position_series, intended_frame_rate)
    return real_frame_pos.reindex(range(real_frame_pos.index.min(), real_frame_pos.index.max())).isna()


def fill_frame_skips_in_video(input_video_path, target_frame_rate, output_path=None, video_codec=None, codec_args=None,
                              overwrite=False):
    """
    Fills the frame-skips in a video by duplicating the previous frame, using ffmpeg. Please make sure that ffmpeg is
    installed and added to the PATH variable. See https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/.

    :param input_video_path: Path to the video to be processed.
    :type input_video_path: str
    :param target_frame_rate: The intended frame rate of the video.
    :type target_frame_rate: float
    :param output_path: Path to the output video. If None, the input video path will be extended with "_full".
    :type output_path: str
    :param video_codec: The video codec to be used. If None, "h264" will be used.
    :type video_codec: str
    :param codec_args: Additional arguments for the video codec. If None, "-crf 18" will be used.
    :type codec_args: str
    :param overwrite: If True, the output video will be overwritten if it already exists.
    :type overwrite: bool
    :rtype: None
    """
    remux_video(input_video_path, output_path=output_path, video_codec=video_codec, codec_args=codec_args,
                overwrite=overwrite, fps=target_frame_rate)


def remux_video(input_video_path, output_path=None, video_codec=None, codec_args=None, overwrite=False, **filter_kwargs):
    """
    Remuxes a video file to the specified codec and filters.

    :param input_video_path: The path to the input video file.
    :type input_video_path: str
    :param output_path: The path where the remuxed video will be saved. If not provided, the original filename with
    "_remuxed." prepended will be used as the output path.
    :type output_path: str or None
    :param video_codec: The codec to use for encoding the video. Default is "h264".
    :type video_codec: str or None
    :param codec_args: Additional arguments to pass to ffmpeg when encoding the video. Default is "-crf 18".
    :type codec_args: str or None
    :param overwrite: Whether to overwrite an existing file with the same name as the input file. Default is False.
    :type overwrite: bool
    :param filter_kwargs: Keyword arguments to pass to ffmpeg when encoding the video. These should be in the format
    "filter_name=value".
    :type filter_kwargs: dict
    :return: The path to the remuxed video file.
    :rtype: str
    """

    target_path = output_path if output_path is not None else "_remuxed.".join(input_video_path.rsplit(".", 1))
    if os.path.exists(target_path) and not overwrite:
        print("Video already exists!")
        return

    video_codec = video_codec if video_codec is not None else "h264"
    codec_args = codec_args if codec_args is not None else "-crf 18"
    filter_string = (
            "-vf \"" + ";".join([f"{key}={value}" for key, value in filter_kwargs.items()]) + "\" "
    ) if filter_kwargs else ""

    powershell_command = f"ffmpeg -i '{input_video_path}' {filter_string}-c:v {video_codec} {codec_args} '{output_path}'"
    run_command_in_powershell(powershell_command, show=True, keep_open=False, blocking=True, return_result=False)


def reset_timestamps_in_h264_video(input_video_path, target_frame_rate=None, output_path=None, overwrite=False,
                                   show_progress=True):
    """
    Copies a H264 video such that its timestamps are reset to a fixed spacing, according to the target frame rate. This
    is sometimes necessary when the video has a variable frame rate and the timestamps are not evenly spaced. In this
    case the access of specific frames is not always possible. The copying is done using ffmpeg and will create a
    temporary .h264 file in the same directory as the output video. The temporary file will be deleted after
    reassembling the video is done. See https://superuser.com/a/1358974 (source of this solution) for more information.
    Please make sure that ffmpeg is installed and added to the PATH variable. See
    https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/.

    :param input_video_path: Path to the video to be processed.
    :type input_video_path: str
    :param target_frame_rate: The intended frame rate of the video. If None, the original frame rate will be used.
    :type target_frame_rate: float or None
    :param output_path: Path to the output video. If None, the input video path will be extended with "_remuxed".
    :type output_path: str or None
    :param overwrite: If True, the output video will be overwritten if it already exists.
    :type overwrite: bool
    :param show_progress: Whether to show the command line output.
    :type show_progress: bool
    :return: None
    """
    target_path = output_path if output_path is not None else "_remuxed.".join(input_video_path.rsplit(".", 1))
    if os.path.exists(target_path) and not overwrite:
        print("Video already exists!")
        return

    if target_frame_rate is None:
        target_frame_rate = get_cv2_video_properties(input_video_path, cv2.CAP_PROP_FPS)[0]

    temp_out_path = target_path.replace(".mp4", "_temp.h264")
    powershell_command = f"ffmpeg -y -i '{input_video_path}' -map 0:v -c copy -bsf:v h264_mp4toannexb '{temp_out_path}'"

    # invoke process
    run_command_in_powershell(powershell_command, show=show_progress, keep_open=False, blocking=True,
                              return_result=False)

    powershell_command = f"ffmpeg -fflags +genpts -r {target_frame_rate} -y -i '{temp_out_path}' " \
                         f"-c copy '{target_path}'"

    # invoke process
    run_command_in_powershell(powershell_command, show=show_progress, keep_open=False, blocking=True,
                              return_result=False)
    os.remove(temp_out_path)  # remove temporary file


def fill_frame_skips_in_video_old(input_video_path, intended_frame_rate, output_path=None, show_progress=True,
                                  fourcc=None):
    """
    Detects skipped frames in a video and fills them with black frames.

    :param input_video_path: Path to the video to be processed.
    :type input_video_path: str
    :param intended_frame_rate: The intended frame rate of the video.
    :type intended_frame_rate: float
    :param output_path: Path to the output video. If None, the input video path will be extended with "_full".
    :type output_path: str
    :param show_progress: Whether to show a progress bar.
    :type show_progress: bool
    :param fourcc: FourCC code of the output video. If None, the default FourCC code will be used.
    :type fourcc: str
    :rtype: None
    """
    target_path = output_path if output_path is not None else "_full.".join(input_video_path.rsplit(".", 1))

    print("Calculating skipped frames...")
    frame_skips = count_frame_skips(fix_frame_msec_series(get_frame_msec_series(input_video_path)),
                                    intended_frame_rate)

    height, width, depth = get_video_shape(input_video_path)
    fourcc = fourcc if fourcc is not None else default_fourcc
    output_video = cv2.VideoWriter(target_path, cv2.VideoWriter_fourcc(*fourcc), intended_frame_rate, (width, height))

    cap = cv2.VideoCapture(input_video_path)

    skip_iterator = tqdm(frame_skips.iteritems(), total=len(frame_skips)) if show_progress else frame_skips.iteritems()
    print("Creating fixed video file...")
    try:
        for frame_index, skipped_count in skip_iterator:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Video ends before frame skips")

            for _ in range(skipped_count):
                output_video.write(np.zeros_like(frame))
            output_video.write(frame)
    finally:
        output_video.release()
        cap.release()


def extract_video_from_roi(output_mp4_path, video_path, roi_bbox, target_fps=None, reader_kwargs=None,
                           text_overlays=None, text_overlay_functions=None, rgba_overlays=None,
                           rgba_overlay_functions=None, show_progress=True, fourcc=None):
    reader_kwargs = dict() if reader_kwargs is None else reader_kwargs
    x, y, w, h = roi_bbox

    cap = cv2.VideoCapture(video_path)
    width, height = int(w), int(h)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if target_fps is None else target_fps
    fourcc = fourcc if fourcc is not None else default_fourcc
    video_writer = cv2.VideoWriter(output_mp4_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    cap.release()

    frame_reader = cv2_frame_reader(video_path, **reader_kwargs)

    start = reader_kwargs["start_frame"] if "start_frame" in reader_kwargs else 0
    end = reader_kwargs["end_frame"] if "end_frame" in reader_kwargs else total_frames
    frames_to_read = end - start - 1
    frame_iterator = tqdm(frame_reader, total=frames_to_read if "read_speed" not in reader_kwargs else int(
        frames_to_read / reader_kwargs["read_speed"])) if show_progress else frame_reader

    static_texts = ensure_list(text_overlays) if text_overlays is not None else []
    function_texts = ensure_list(text_overlay_functions) if text_overlay_functions is not None else []
    static_overlays = ensure_list(rgba_overlays) if rgba_overlays is not None else []
    function_overlays = ensure_list(rgba_overlay_functions) if rgba_overlay_functions is not None else []

    try:
        for frame_index, frame in frame_iterator:
            for static_text in static_texts:
                frame = add_text_to_frame(frame, **static_text)
            for function_text in function_texts:
                frame = add_text_to_frame(frame, **function_text(frame_index))
            for static_overlay in static_overlays:
                frame = add_rgba_overlay_to_frame(frame, static_overlay)
            for function_overlay in function_overlays:
                frame = add_rgba_overlay_to_frame(frame, function_overlay(frame_index))
            video_writer.write(get_roi_from_frame(frame, roi_bbox))
    except KeyboardInterrupt:
        pass
    finally:
        video_writer.release()


def extract_videos_from_rois(output_mp4_path_roi_bbox_dict, video_path, target_fps=None, reader_kwargs=None,
                             frame_functions=None, text_overlays=None, text_overlay_functions=None, rgba_overlays=None,
                             rgba_overlay_functions=None, show_progress=True, fourcc=None):
    reader_kwargs = dict() if reader_kwargs is None else reader_kwargs

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if target_fps is None else target_fps

    fourcc = fourcc if fourcc is not None else default_fourcc
    video_writer_dict = {}
    for output_mp4_path, roi_bbox_parameters in output_mp4_path_roi_bbox_dict.items():
        width, height = int(roi_bbox_parameters["width"]), int(roi_bbox_parameters["height"])
        video_writer_dict[output_mp4_path] = cv2.VideoWriter(output_mp4_path, cv2.VideoWriter_fourcc(*fourcc), fps,
                                                             (width, height))
    cap.release()

    frame_reader = cv2_frame_reader(video_path, **reader_kwargs)

    start = reader_kwargs["start_frame"] if "start_frame" in reader_kwargs else 0
    end = reader_kwargs["end_frame"] if "end_frame" in reader_kwargs else total_frames
    frames_to_read = end - start - 1
    frame_iterator = tqdm(frame_reader, total=frames_to_read if "read_speed" not in reader_kwargs else int(
        frames_to_read / reader_kwargs["read_speed"])) if show_progress else frame_reader

    frame_functions = ensure_list(frame_functions) if frame_functions is not None else []
    static_texts = ensure_list(text_overlays) if text_overlays is not None else []
    function_texts = ensure_list(text_overlay_functions) if text_overlay_functions is not None else []
    static_overlays = ensure_list(rgba_overlays) if rgba_overlays is not None else []
    function_overlays = ensure_list(rgba_overlay_functions) if rgba_overlay_functions is not None else []

    try:
        for frame_index, frame in frame_iterator:
            for frame_function in frame_functions:
                frame = frame_function(frame)
            for static_text in static_texts:
                frame = add_text_to_frame(frame, **static_text)
            for function_text in function_texts:
                frame = add_text_to_frame(frame, **function_text(frame_index))
            for static_overlay in static_overlays:
                frame = add_rgba_overlay_to_frame(frame, static_overlay)
            for function_overlay in function_overlays:
                frame = add_rgba_overlay_to_frame(frame, function_overlay(frame_index))
            for output_mp4_path, roi_bbox_parameters in output_mp4_path_roi_bbox_dict.items():
                frame_root = roi_bbox_parameters["root"].loc[frame_index].astype(int)
                frame_bbox = (*frame_root, roi_bbox_parameters["width"], roi_bbox_parameters["height"])
                video_writer_dict[output_mp4_path].write(get_roi_from_frame(frame, frame_bbox))
    except KeyboardInterrupt:
        pass
    finally:
        for output_mp4_path, roi_bbox in output_mp4_path_roi_bbox_dict.items():
            video_writer_dict[output_mp4_path].release()


def extract_n_frames(video_path, output_directory, frame_n, output_format=".jpg"):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = pd.Series(range(frame_count)).sample(frame_n)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    for selected_frame in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)

        ret, frame = cap.read()
        if not ret:
            raise IndexError("Selected frame {} not available!".format(selected_frame))

        cv2.imwrite(
            os.path.join(output_directory, video_name + f"_frame{str(selected_frame).zfill(6)}" + output_format), frame)


def extract_frames(video_path, frame_indices, output_paths_or_pattern=None,
                   frame_index_padding=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_paths_or_pattern = output_paths_or_pattern if output_paths_or_pattern is not None \
        else video_name + "_frame{}.jpg"

    frame_index_padding = frame_index_padding if frame_index_padding is not None else len(str(max(frame_indices)))

    cap = cv2.VideoCapture(video_path)

    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret, frame = cap.read()
        if not ret:
            raise IndexError("Selected frame {} not available!".format(frame_index))

        if not isinstance(output_paths_or_pattern, list):
            output_path = output_paths_or_pattern.format(str(frame_index).zfill(frame_index_padding))
        else:
            output_path = output_paths_or_pattern[i]
        cv2.imwrite(output_path, frame)

    cap.release()


def extract_all_frames(video_path, output_directory, output_format=".png"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_directory, video_name, '/image_{:08d}{}'.format(i, output_format)), frame)
        i += 1

    cap.release()


def save_images_from_videos(video_files, output_directory, gap=1):
    output_path = os.path.join(output_directory, "image_info.csv")
    write_header = False if os.path.isfile(output_path) else True
    for video_file in tqdm(video_files):
        save_images_from_video(video_file, output_directory, gap).to_csv(output_path, mode='a', header=write_header)


def save_images_from_video(video_file, output_directory, gap=1):
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    ensure_directory(os.path.join(output_directory, video_name))

    output_frame_rows = []
    for i in range(0, frame_count, gap):
        image_save_name = video_name + '/image_{:08d}.png'.format(i)
        output_path = os.path.join(output_directory, image_save_name)

        if os.path.exists(output_path):
            new_row = {'image_name': image_save_name, 'video_name': video_name, 'frame_index': i, 'width': width,
                       'height': height}
            output_frame_rows.append(new_row)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imwrite(output_path, frame)

        new_row = {'image_name': image_save_name, 'video_name': video_name, 'frame_index': i, 'width': width,
                   'height': height}
        output_frame_rows.append(new_row)

    cap.release()

    return pd.DataFrame(output_frame_rows)


def get_frames(video_path, frame_indices, show_progress=False, cap_args=tuple(),
               jump_to_frames=True):
    """
    Get frames from a video using OpenCV

    :param video_path: Path to the video
    :type video_path: str
    :param frame_indices: List of frame indices to get
    :type frame_indices: list
    :param show_progress: Whether to show a progress bar
    :type show_progress: bool
    :param cap_args: Additional arguments for the OpenCV VideoCapture function
    :type cap_args: list or tuple
    :param jump_to_frames: Whether to jump to the desired frames or read iteratively until the desired frame index is
    reached. Jumping to the desired frames is faster but may not work with all video(s) or formats.
    :type jump_to_frames: bool
    :return: List of frames
    :rtype: list of np.ndarray
    """

    cap = cv2.VideoCapture(video_path, *cap_args)
    frames = []

    current_frame = -1
    if show_progress:
        progress = tqdm(total=len(frame_indices), leave=False, desc="Reading frames")

    while True:
        if jump_to_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[len(frames)])
            current_frame = frame_indices[len(frames)] - 1  # -1 to account for the increase after reading

        ret, frame = cap.read()
        current_frame += 1
        if not ret:
            break

        if current_frame in frame_indices:
            frames.append(frame)
            if show_progress:
                progress.update(1)
            if len(frames) == len(frame_indices):
                break

    if show_progress:
        progress.close()

    if len(frames) != len(frame_indices):
        # check for missing frames
        missing_indices = frame_indices[len(frames):]
        raise ValueError(f"Could not read frames {missing_indices} from video {video_path}")

    cap.release()
    return frames


def get_frames_old(video_path, frame_indices, show_progress=False, cap_args=tuple()):
    """
    Get frames from a video using OpenCV

    :param video_path: Path to the video
    :type video_path: str
    :param frame_indices: List of frame indices to get
    :type frame_indices: list
    :param show_progress: Whether to show a progress bar
    :type show_progress: bool
    :param cap_args: Additional arguments for the OpenCV VideoCapture function
    :type cap_args: list or tuple
    :return: List of frames
    :rtype: list of np.ndarray
    """
    cap = cv2.VideoCapture(video_path, *cap_args)
    frames = []

    index_iterator = tqdm(frame_indices) if show_progress else frame_indices
    for frame_index in index_iterator:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            raise IndexError("Selected frame {} not available!".format(frame_index))
        frames.append(frame)
    cap.release()
    return frames


def get_n_frames(video_path, n=1, offset=0, jump_to_frames=True):
    """
    Get n frames of a video using OpenCV starting from a given offset frame. This is a convenience function for
    get_frames(video_path, list(range(offset, offset + n))).

    :param video_path: Path to the video
    :type video_path: str
    :param n: Number of frames to get
    :type n: int
    :param offset: Offset frame
    :type offset: int
    :param jump_to_frames: Whether to jump to the desired frames or read iteratively until the desired frame index is
    reached. Jumping to the desired frames is faster but may not work with all video(s) or formats.
    :type jump_to_frames: bool
    :return: List of frames
    :rtype: list of np.ndarray
    """
    selected_frames = list(range(offset, offset + n))
    return get_frames(video_path, selected_frames, jump_to_frames=jump_to_frames)


def get_first_n_frames(video_path, n=1):
    """
    Get the first n frames of a video using OpenCV

    :param video_path: Path to the video
    :type video_path: str
    :param n: Number of frames to get
    :type n: int
    :return: List of frames
    :rtype: list of np.ndarray
    """
    return get_n_frames(video_path, n=n, offset=0)


def get_first_frame(video_path):
    """
    Get the first frame of a video using OpenCV. This is a convenience function that calls get_first_n_frames with n=1

    :param video_path: Path to the video
    :type video_path: str
    :return: First frame of the video
    :rtype: np.ndarray
    """
    return get_first_n_frames(video_path, 1)[0]


def create_video_with_fourcc(video_path, fourcc_string):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*fourcc_string)
    output_path = video_path.replace(".avi", f"{fourcc_string}.mp4")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame = get_first_frame(video_path)
    video_writer.write(frame)
    video_writer.release()

    return output_path


def get_initial_timestamp(video_path):
    """
    Get the initial timestamp of a video in milliseconds

    :param video_path: A path to a video file
    :type video_path: str
    :return: The initial timestamp of the video in milliseconds
    :rtype: float
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    initial_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    cap.release()
    if not ret:
        raise ValueError("Unreadable video!")
    return initial_timestamp


def timestamp_to_frame(timestamp, frame_rate):
    """
    Convert a timestamp in milliseconds to a frame index

    :param timestamp: A timestamp in milliseconds
    :type timestamp: float
    :param frame_rate: The frame rate of the video
    :type frame_rate: float or int
    :return: The frame index corresponding to the timestamp
    :rtype: int
    """
    return int(timestamp / 1000 * frame_rate)


def frame_to_timestamp(frame, frame_rate):
    """
    Convert a frame index to a timestamp in milliseconds

    :param frame: A frame index
    :type frame: int
    :param frame_rate: The frame rate of the video
    :type frame_rate: float or int
    :return: The timestamp corresponding to the frame index
    :rtype: float
    """
    return frame / frame_rate * 1000


def get_actual_start_frame_index(video_path, frame_rate=30):
    """
    Get the actual start frame index of a video, which is the frame index corresponding to the initial timestamp

    :param video_path: A path to a video file
    :type video_path: str
    :param frame_rate: The frame rate of the video
    :type frame_rate: float or int
    :return: The actual start frame index of the video
    :rtype: int
    """
    return timestamp_to_frame(get_initial_timestamp(video_path), frame_rate)


def extract_pattern_chunks_to_video(out_dir, extended_pattern_chunk_df, out_name_format=None,
                                    additional_generator_kwargs=None, **generator_kwargs):
    """
    Extracts the pattern chunks of a video to separate video files. The video files are named according to the
    out_name_format and saved in the out_dir directory. The additional_generator_kwargs is a dictionary that maps row
    indices to keyword arguments for the annotated_video_frames_generator function and overrides the generator_kwargs
    argument for that specific video. The extended_pattern_chunk_df must have the columns "video_path", "index_start",
    and "index_end".

    :param out_dir: A path to the output directory that will contain the extracted videos
    :type out_dir: str
    :param extended_pattern_chunk_df: A DataFrame with the columns "video_path", "index_start", and "index_end"
    :type extended_pattern_chunk_df: pd.DataFrame
    :param out_name_format: The format of the output video file names. The format can contain placeholders for the row
        index and other columns of the DataFrame. If None, the default format "vid_row_{row_idx}.mp4" is used.
    :type out_name_format: str or None
    :param additional_generator_kwargs: A dictionary that maps row indices to keyword arguments for the
        annotated_video_frames_generator function. The keyword arguments override the generator_kwargs argument for that
        specific video.
    :type additional_generator_kwargs: dict or None
    :param generator_kwargs: Additional keyword arguments for the annotated_video_frames_generator function. See the
        annotated_video_frames_generator function for more information. This argument is used for all videos if not
        overridden by additional_generator_kwargs.
    :type generator_kwargs: Any
    :return: None
    """
    if out_name_format is None:
        out_name_format = "vid_row_{row_idx}.mp4"

    if additional_generator_kwargs is None:
        additional_generator_kwargs = {}

    os.makedirs(out_dir, exist_ok=True)
    for row_idx, row in tqdm(extended_pattern_chunk_df.iterrows(), total=len(extended_pattern_chunk_df)):
        row_dict = row.to_dict()
        row_dict["row_idx"] = row_idx

        source_path = row["video_path"]

        out_name = out_name_format.format(**row_dict)
        out_path = os.path.join(out_dir, out_name)

        start_frame, end_frame = row["index_start"], row["index_end"]

        row_generator_kwargs = additional_generator_kwargs[row_idx] if row_idx in additional_generator_kwargs else {}
        generator_kwargs = {**generator_kwargs, **row_generator_kwargs, **dict(show_progress=False)}
        if not "reader_kwargs" in generator_kwargs.keys():
            generator_kwargs["reader_kwargs"] = dict(start_frame=start_frame, end_frame=end_frame)
        else:
            generator_kwargs["reader_kwargs"]["start_frame"] = start_frame
            generator_kwargs["reader_kwargs"]["end_frame"] = end_frame

        write_video_to_file(out_path, source_path, **generator_kwargs)


def extract_pattern_chunks_to_stitched_video(out_path, extended_pattern_chunk_df, rows=4, cols=None,
                                             additional_generator_kwargs=None, **stitching_kwargs):
    """
    Extracts the pattern chunks of a video to separate video files, stitches them together, and saves the stitched video
    to a file. The extended_pattern_chunk_df must have the columns "video_path", "index_start", and "index_end".
    Uses the extract_pattern_chunks_to_video and write_stitched_video_to_file functions.

    :param out_path: The path to the output video file
    :type out_path: str
    :param extended_pattern_chunk_df: A DataFrame with the columns "video_path", "index_start", and "index_end"
    :type extended_pattern_chunk_df: pd.DataFrame
    :param rows: The number of rows in the stitched video. Default is 4.
    :type rows: int
    :param cols: The number of columns in the stitched video. If None, the number of columns is calculated based on the
        number of rows and the number of videos. Default is None.
    :type cols: int or None
    :param additional_generator_kwargs: A dictionary that maps row indices to keyword arguments for the
        annotated_video_frames_generator function. The keyword arguments override the generator_kwargs argument for that
        specific video.
    :type additional_generator_kwargs: dict or None
    :param stitching_kwargs: Additional keyword arguments for the write_stitched_video_to_file function. See the
        write_stitched_video_to_file function for more information.
    :type stitching_kwargs: Any
    :return: None
    """
    out_dir = os.path.dirname(out_path)

    temp_name_format = "_temp_vid_row_{row_idx}.mp4"
    extract_pattern_chunks_to_video(out_dir, extended_pattern_chunk_df, out_name_format=temp_name_format,
                                    additional_generator_kwargs=additional_generator_kwargs)

    video_paths = extended_pattern_chunk_df.index.to_series().apply(
        lambda x: os.path.join(out_dir, temp_name_format.format(row_idx=x))).tolist()

    stitching_kwargs = {**stitching_kwargs, **dict(rows=rows, cols=cols, show_progress=False)}
    write_stitched_video_to_file(out_path, video_paths, **stitching_kwargs)

    for video_path in video_paths:
        os.remove(video_path)