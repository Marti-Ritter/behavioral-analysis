import matplotlib.pyplot as plt
import pandas as pd

from .base_classes import FrameProcessor
from .pyplot_funcs import create_pyplot_canvas_for_frame, pyplot_annotated_frames_generator
from ..tracking_tools.plots import plot_keypoint_instances, plot_keypoint_instance


class PyplotAnnotatedFramesGenerator(FrameProcessor):
    source_canvas_was_created = False
    def __init__(self, frame_source, pyplot_func, source_canvas=None, pyplot_input=None, *args, **kwargs):
        if source_canvas is None:
            example_index, example_frame = frame_source.example_index, frame_source.example_frame

            if example_frame is None:
                raise ValueError("No source_canvas given and frame_source is empty")

            self.source_canvas = create_pyplot_canvas_for_frame(example_frame)
            self.source_canvas_was_created = True
        else:
            self.source_canvas = source_canvas
            self.source_canvas_was_created = False

        super().__init__(frame_source, pyplot_annotated_frames_generator, pyplot_func=pyplot_func,
                         source_canvas=self.source_canvas, pyplot_input=pyplot_input, *args, **kwargs)

    def __del__(self):
        if self.source_canvas_was_created:
            plt.close(self.source_canvas)


class KeypointInstanceAnnotatedFramesGenerator(PyplotAnnotatedFramesGenerator):
    def __init__(self, frame_source, track_df, source_canvas=None, *args, **kwargs):
        if isinstance(track_df.index, pd.MultiIndex):
            plot_func = plot_keypoint_instances
            index_length = len(track_df.index.names)
            reformed_track_df = track_df.reorder_levels([index_length-1] + list(range(index_length-1)))
        else:
            plot_func = plot_keypoint_instance
            reformed_track_df = track_df

        reformed_keypoint_df = reformed_track_df.stack("keypoint_name", future_stack=True)

        def keypoint_access_func(frame_index):
            if frame_index in reformed_keypoint_df.index.get_level_values(level=0):
                return reformed_keypoint_df.loc[frame_index]
            return pd.DataFrame()

        super().__init__(frame_source, pyplot_func=plot_func, source_canvas=source_canvas,
                         pyplot_input=keypoint_access_func, *args, **kwargs)
