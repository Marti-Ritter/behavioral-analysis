import numpy as np

from .cv2_funcs import add_rgba_overlay_to_frame
from .matplotlib_funcs import create_canvas, fig2cv2, get_function_added_artists
from .processor_funcs import ensure_frame_index_func, func_modified_frames_generator


def create_pyplot_canvas_for_frame(frame, dpi=100, alpha=0.):
    frame = np.atleast_3d(frame)
    height, width, n_channels = frame.shape
    return create_canvas(shape_x=width, shape_y=height, dpi=dpi, alpha=alpha, show_after_creation=False)


def pyplot_annotated_frames_generator(frame_source, pyplot_func, source_canvas, pyplot_input=None, *args, **kwargs):
    pyplot_input_func = ensure_frame_index_func(pyplot_input)

    def generate_pyplot_rgba_overlay(frame_index):
        local_args = args
        local_kwargs = kwargs

        input_dict_or_object = pyplot_input_func(frame_index)

        if isinstance(input_dict_or_object, dict):
            local_kwargs = {**local_kwargs, **input_dict_or_object}
        else:
            local_args = [*local_args, input_dict_or_object]

        # annotate source ax with pyplot_func
        new_artists = get_function_added_artists(pyplot_func, reference_figure=source_canvas, ax=source_canvas.gca(),
                                                 *local_args, **local_kwargs)

        # transfer result to np.ndarray
        pyplot_rgba_array = fig2cv2(source_canvas)

        # restore previous state
        for new_artist in new_artists:
            new_artist.remove()

        # return result
        return pyplot_rgba_array

    yield from func_modified_frames_generator(frame_source, modification_func=add_rgba_overlay_to_frame,
                                              modification_input=generate_pyplot_rgba_overlay,
                                              use_alpha_composite=True)
