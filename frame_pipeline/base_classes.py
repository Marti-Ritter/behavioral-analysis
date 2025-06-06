import warnings
from functools import partial

import numpy as np


class FrameGenerator(object):
    """
    Defines a basic object that is able to extract frames from a source and returns them unchanged.
    It can read properties from the source, such as fps and length, or extract them itself through reading the first
    frame (for size and number of color channels).
    This is used as the base for any other classes in the toolbox.
    """
    fps = None
    length = None
    height, width = None, None
    n_channels = None

    example_index, example_frame = None, None

    def __init__(self, frame_source):
        """
        Create a new FrameGenerator object from a frame_source. This will just return whatever the source returns
        through iteration or next() calls.
        :param frame_source: Anything that returns a tuple of (frame_index, frame) when iterated over. Features get read
        when the frame_source is itself a FrameGenerator.
        :type frame_source: Any
        """
        self.copy_source_features(frame_source)

        try:
            self.example_index, self.example_frame = next(frame_source)
            self.height, self.width, self.n_channels = np.atleast_3d(self.example_frame).shape
        except StopIteration:
            warnings.warn("No frames available for processing")
        self.frame_source = self.get_frame_source()

    def copy_source_features(self, source=None):
        source = self.frame_source if source is None else source
        self.fps = source.fps if hasattr(source, "fps") else None
        self.length = len(source) if hasattr(source, "__len__") else None

    def get_frame_source(self):
        def _delayed_frame_source():
            yield self.example_index, self.example_frame
            yield from self.frame_source
        new_source = _delayed_frame_source()
        return new_source

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.frame_source)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if hasattr(self.frame_source, "__getitem__"):
            sliced_frame_source_output = self.frame_source[item]
            return sliced_frame_source_output
        else:
            raise NotImplementedError(f"Cannot index into frame source of type {type(self.frame_source)}")


class FrameProcessor(FrameGenerator):
    """
    Video stream features:
    fps, height, width, n_channels: as properties, height, width, n_channels can also be read from
    example_frame property
    length: accessible under len()
    frame_index: returned as a tuple with img

    Theoretically all of this should suffice to present the processed set of frames as a continuous video stream,
    for writing or visualization.
    """
    def __init__(self, frame_source, processor_func, *args, **kwargs):
        self.unmodified_source = frame_source
        new_source = frame_source.get_frame_source() if hasattr(frame_source, "get_frame_source") else frame_source
        self.partial_processor_func = partial(processor_func, frame_source=new_source, *args, **kwargs)
        super().__init__(frame_source=self.partial_processor_func())
        self.copy_source_features(self.unmodified_source)

    def get_frame_source(self, *args, **kwargs):
        frame_source = self.unmodified_source if not "frame_source" in kwargs else kwargs.pop("frame_source")
        new_source = frame_source.get_frame_source() if hasattr(frame_source, "get_frame_source") else frame_source
        return self.partial_processor_func(*args, frame_source=new_source, **kwargs)

    def __iter__(self):
        self.frame_source = self.get_frame_source()
        return self

    def __getitem__(self, item):
        # as we replace frame_source with the modified generator, we have to access the unmodified source here
        if hasattr(self.unmodified_source, "__getitem__"):
            sliced_frame_source_output = self.unmodified_source[item]
            if not isinstance(sliced_frame_source_output, list):
                sliced_frame_source_output = [sliced_frame_source_output]
            processed_output = [(frame_index, frame) for frame_index, frame in
                                self.get_frame_source(frame_source=sliced_frame_source_output)]

            if isinstance(item, int):
                return processed_output[0]
            return processed_output
        else:
            raise NotImplementedError(f"Cannot index into frame source of type {type(self.unmodified_source)}")


class FrameReader(FrameGenerator):
    def __init__(self, frame_source_func, *args, **kwargs):
        self.partial_frame_source = partial(frame_source_func, *args, **kwargs)
        super().__init__(frame_source=self.partial_frame_source())
        self.fps, self.length = None, None  # Has to implemented in child classes, along with slicing

    def get_frame_source(self, *args, **kwargs):
        return self.partial_frame_source(*args, **kwargs)

    def __iter__(self):
        self.frame_source = self.get_frame_source()
        return self
