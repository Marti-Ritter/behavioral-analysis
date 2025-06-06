import pathlib
from tqdm.auto import tqdm

import cv2
from .pil_funcs import (ensure_pil_image, resize_with_preserved_aspect_ratio, pad_image,
                        stitch_image_list, add_text)

default_fourcc = "H264"


def write_frame_source_to_video(output_video_path, frame_source, fps_override=None, fourcc=None, grayscale=False,
                                show_progress=True):
    _first_index, first_frame = next(frame_source)
    if hasattr(frame_source, "height") and hasattr(frame_source, "width"):
        height, width = frame_source.height, frame_source.width
    else:
        height, width = first_frame.shape[:2]

    fourcc = fourcc if fourcc is not None else default_fourcc

    source_fps = frame_source.fps if hasattr(frame_source, "fps") else None
    fps = fps_override if fps_override is not None else source_fps
    fps = fps if fps is not None else 30

    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height),
                                   int(not grayscale))

    frame_source = tqdm(frame_source) if show_progress else frame_source
    try:
        video_writer.write(first_frame)
        for _frame_index, frame in frame_source:
            if grayscale and frame.ndim==3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            video_writer.write(frame)
    except KeyboardInterrupt:
        pass
    finally:
        video_writer.release()


def write_frame_source_to_gif(output_gif_path, frame_source, fps_override=None, loop=True, show_progress=False):
    source_fps = frame_source.fps if hasattr(frame_source, "fps") else None
    fps = fps_override if fps_override is not None else source_fps
    fps = fps if fps is not None else 30

    duration = int(1/fps*1000)

    frame_source = tqdm(frame_source, total=len(frame_source)) if show_progress else frame_source
    unindexed_images = [ensure_pil_image(frame, array_is_bgr=True) for (frame_index, frame) in frame_source]

    unindexed_images[0].save(output_gif_path, save_all=True, append_images=unindexed_images[1:],
                             duration=duration, loop=int(loop))


def write_frame_source_to_directory(output_dir_path, frame_source, output_extension=".png", index_reformatter_func=None,
                                    show_progress=False):
    output_dir_path = pathlib.Path(output_dir_path)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    indexed_images = [(frame_index, ensure_pil_image(frame, array_is_bgr=True)) for (frame_index, frame) in
                      frame_source]
    indexed_images = tqdm(indexed_images, total=len(indexed_images)) if show_progress else indexed_images

    for frame_index, image in indexed_images:
        frame_index = index_reformatter_func(frame_index) if index_reformatter_func is not None else frame_index
        image_output_path = output_dir_path / (str(frame_index) + output_extension)
        image.save(image_output_path)


def create_pipeline_overview(*frame_pipeline_columns, overview_width=1000, overview_height=400, frame_scale=0.9,
                             annotation_space=40, annotation_list=None):
    n_cols = len(frame_pipeline_columns)
    width_per_frame = overview_width // n_cols

    frame_pipeline_columns = [[frame_source_col] if not isinstance(frame_source_col, list) else frame_source_col for
                              frame_source_col in frame_pipeline_columns]

    col_frames = []
    for i, frame_source_col in enumerate(frame_pipeline_columns):
        height_per_frame = (overview_height - annotation_space) // len(frame_source_col)
        rescaled_width, rescaled_height = width_per_frame * frame_scale, height_per_frame * frame_scale

        col_example_frames = [ensure_pil_image(frame_source.example_frame, array_is_bgr=True) for frame_source in
                              frame_source_col]
        col_example_frames = [
            resize_with_preserved_aspect_ratio(ef, (int(rescaled_width), int(rescaled_height))) for ef in
            col_example_frames]

        def _pad_to_size(pil_image, target_size):
            current_size = pil_image.size
            width_difference, height_difference = target_size[0] - current_size[0], target_size[1] - current_size[1]
            w_pad = width_difference//2
            h_pad = height_difference//2
            return pad_image(pil_image, (w_pad, h_pad, w_pad, h_pad))

        col_example_frames = [_pad_to_size(ef, (int(width_per_frame), int(height_per_frame))) for ef in
                              col_example_frames]

        col_frame_image = stitch_image_list(*col_example_frames, rows=len(col_example_frames))
        col_frame_image = pad_image(col_frame_image, (0, annotation_space, 0, 0))
        col_annotation = str(i) if annotation_list is None else annotation_list[i]
        col_frame_image = add_text(col_frame_image, col_annotation, relative_xy=(0.5, 0), color=(0, 0, 0), anchor="ma")
        col_frames.append(col_frame_image)
    return stitch_image_list(*col_frames, columns=len(col_frames))
