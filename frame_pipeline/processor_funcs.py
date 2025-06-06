import itertools
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from .cv2_funcs import (apply_functions_to_frame, add_annotations_to_frame, correct_frame_orientation,
                        get_padded_roi_from_frame, add_text_to_frame, add_rgba_overlay_to_frame,
                        extract_reoriented_roi_around_point)
from .pil_funcs import (merge_images, stitch_image_list, ensure_pil_image)


def ensure_frame_index_func(input_object):
    if isinstance(input_object, Callable):
        return input_object
    elif isinstance(input_object, pd.DataFrame):
        def df_to_func(frame_index):
            return dict(input_object.loc[frame_index].dropna()) if frame_index in input_object.index else {}
        return df_to_func
    elif isinstance(input_object, dict):
        return input_object.get
    elif isinstance(input_object, (list, tuple)):
        array_dict = {i: l for i, l in enumerate(input_object)}
        return array_dict.get
    elif input_object is None:
        def empty_func(frame_index):
            return {}
        return empty_func
    else:
        def return_object_unchanged(frame_index):
            return input_object
        return return_object_unchanged


def annotated_frames_generator(frame_source, initial_modification_functions=None, annotator_video_shift=0,
                                     final_modification_functions=None, **annotator_kwargs):
    for frame_index, frame in frame_source:
        corrected_frame_index = frame_index + annotator_video_shift
        frame = apply_functions_to_frame(frame, frame_functions=initial_modification_functions)
        frame = add_annotations_to_frame(frame, corrected_frame_index, **annotator_kwargs)
        frame = apply_functions_to_frame(frame, frame_functions=final_modification_functions)

        yield (frame_index, frame)

def func_modified_frames_generator(frame_source, modification_func, modification_input=None,
                                   no_args_means_skip=False, *args, **kwargs):
    modification_input_func = ensure_frame_index_func(modification_input)

    original_args, original_kwargs = args, kwargs

    for frame_index, frame in frame_source:
        input_dict_or_object = modification_input_func(frame_index)

        if isinstance(input_dict_or_object, dict):
            kwargs = {**original_kwargs, **input_dict_or_object}
        else:
            args = [*original_args, input_dict_or_object]

        if no_args_means_skip and (len(args) == len(kwargs) == 0):
            yield frame_index, frame
        else:
            yield frame_index, modification_func(frame, *args, **kwargs)


def text_annotated_frames_generator(frame_source, text_func=add_text_to_frame, text_input=None, *args, **kwargs):
    yield from func_modified_frames_generator(frame_source, modification_func=text_func,
                                              modification_input=text_input, *args, **kwargs)


def rgba_annotated_frames_generator(frame_source, rgba_func=add_rgba_overlay_to_frame,
                                    rgba_overlay_input=None, *args, **kwargs):
    yield from func_modified_frames_generator(frame_source, modification_func=rgba_func,
                                              modification_input=rgba_overlay_input, *args, **kwargs)


def frame_roi_extractor(frame_source, roi_df):
    roi_df = roi_df[["x", "y", "w", "h"]].astype(int)
    for frame_index, frame in frame_source:
        yield frame_index, get_padded_roi_from_frame(frame, roi_bbox=roi_df.loc[frame_index])


def frame_heading_corrector(frame_source, orientation_series, target_orientation=0):
    for frame_index, frame in frame_source:
        current_orientation = orientation_series.loc[frame_index]
        yield frame_index, correct_frame_orientation(frame, orientation=current_orientation,
                                                     target_orientation=target_orientation)

def frame_reoriented_roi_extractor(frame_source, orientation_series, roi_df, target_orientation=0):
    roi_df = roi_df[["x", "y", "w", "h"]].astype(int)
    for frame_index, frame in frame_source:
        x, y, w, h = roi_df.loc[frame_index]
        center_point = (x+w//2, y+h//2)

        current_orientation = orientation_series.loc[frame_index]
        yield frame_index, extract_reoriented_roi_around_point(frame, center_point=center_point, roi_shape=(w, h),
                                                               roi_orientation=current_orientation,
                                                               target_orientation=target_orientation)


def merge_frame_sources(frame_source, frame_source_positions=None, rows=None, columns=None,
                        pad_color=(255, 255, 255), length_limit_source_index=None,
                        frame_index_source_index=None, frame_source_alpha_list=None):
    # grab first frame of each frame source to determine filler frames for uneven lengths
    frame_sources_list = [iter(f) for f in frame_source]
    first_outputs = [next(frame_source) for frame_source in frame_sources_list]
    first_frame_indices = [first_output[0] for first_output in first_outputs]
    first_frames = [ensure_pil_image(first_output[1]) for first_output in first_outputs]
    fill_frames = [ensure_pil_image(np.zeros_like(np.array(frame))) for frame in first_frames]

    if (rows is not None) or (columns is not None):
        merge_func = partial(stitch_image_list, rows=rows, columns=columns, pad_color=pad_color)
    else:
        merge_func = partial(merge_images, positions=frame_source_positions, pad_color=pad_color)

    merged_frame = merge_func(*first_frames)

    first_index = first_frame_indices[frame_index_source_index] if frame_index_source_index is not None else 0
    yield first_index, np.array(merged_frame)

    zipped_frame_iterator_list = itertools.zip_longest(*frame_sources_list)
    for frame_index, zipped_frame_source_output in enumerate(zipped_frame_iterator_list, start=0):
        if length_limit_source_index is not None and zipped_frame_source_output[length_limit_source_index] is None:
            break

        if frame_index_source_index is not None:
            frame_indices = [output[0] for output in zipped_frame_source_output]
            frame_index = frame_indices[frame_index_source_index]

        frames = [output[1] if output is not None else fill_frames[i] for i, output in
                  enumerate(zipped_frame_source_output)]
        frames = [ensure_pil_image(frame) for frame in frames]

        if frame_source_alpha_list is not None:
            for frame, alpha in zip(frames, frame_source_alpha_list):
                frame.putalpha(alpha)

        merged_frame = merge_func(*frames)

        yield frame_index, np.array(merged_frame)


