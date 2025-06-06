import cv2

from .base_classes import FrameReader
from .cv2_funcs import get_cv2_video_properties
from .reader_funcs import (cv2_video_frame_reader, directory_frame_reader, get_indexed_frame_files_from_directory)
from ..utility.general import copy_signature_from


@copy_signature_from(cv2_video_frame_reader)
class Cv2VideoFrameReader(FrameReader):
    def __init__(self, video_path, *args, start_frame=0, end_frame=None, fps_override=None, read_speed=1, **kwargs):
        super().__init__(frame_source_func=cv2_video_frame_reader, *args, video_path=video_path,
                         start_frame=start_frame, end_frame=end_frame, fps_override=fps_override, read_speed=read_speed,
                         **kwargs)

        fps = get_cv2_video_properties(video_path, cv2.CAP_PROP_FPS)[0] if fps_override is None else fps_override
        self._total_frames = int(get_cv2_video_properties(video_path, cv2.CAP_PROP_FRAME_COUNT)[0])

        self.fps = fps
        end_frame = self._total_frames if end_frame is None else end_frame
        self.length = int((end_frame - start_frame + 1) / read_speed)

    def __getitem__(self, item):
        if isinstance(item, int):
            start_frame, end_frame = item, item
        elif isinstance(item, slice):
            start_frame = item.start or 0
            end_frame = item.stop or None
            step = item.step or 1

        if start_frame < 0:
            start_frame += self._total_frames
        if (end_frame is not None) and (end_frame < 0):
            end_frame += self._total_frames

        frame_source_output = [(frame_index, frame) for frame_index, frame in
                               self.get_frame_source(start_frame=start_frame, end_frame=end_frame)]

        if isinstance(item, int):
            return frame_source_output[0]
        return frame_source_output[::step]


class Cv2VideoSequenceReader(FrameReader):
    def __init__(self, video_path, *cv2_video_frame_reader_kwargs, **kwargs):
        def build_frame_source_from_kwargs_list(video_path, *cv2_video_frame_reader_kwargs):
            separate_frame_sources = [Cv2VideoFrameReader(video_path, **kwargs_dict) for kwargs_dict in
                                      cv2_video_frame_reader_kwargs]
            def concatenated_frame_source():
                for frame_source in separate_frame_sources:
                    yield from frame_source
            return concatenated_frame_source()

        super().__init__(frame_source_func=build_frame_source_from_kwargs_list, video_path=video_path,
                         *cv2_video_frame_reader_kwargs)

        fps_override = kwargs["fps_override"] if fps_override in kwargs.keys() else None
        self.fps = get_cv2_video_properties(video_path, cv2.CAP_PROP_FPS)[0] if fps_override is None else fps_override
        self.length = sum([len(frame_source) for frame_source in self.frame_source])

    def __getitem__(self, item):
        raise NotImplementedError("Slicing is not implemented for Cv2VideoSequenceReader")


class FrameStack(object):
    def __init__(self, indexed_frame_stack, start_frame=0, end_frame=None):
        indexed_frame_stack = list(indexed_frame_stack)
        if not isinstance(indexed_frame_stack[0][0], int):
            # ensure that the frame stack is indexed
            indexed_frame_stack = [(i, frame) for i, frame in enumerate(indexed_frame_stack)]

        self.start_frame = start_frame
        self.end_frame = (len(indexed_frame_stack) - 1) if end_frame is None else end_frame
        self.frame_stack = indexed_frame_stack
        self.current_index = start_frame
        self.length = self.end_frame - self.start_frame + 1

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index > self.end_frame:
            raise StopIteration("End of sequence reached")

        current_frame = self.frame_stack[self.current_index]
        self.current_index += 1
        return current_frame
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, item):
        return self.frame_stack[item]


class FrameStackReader(FrameReader):
    def __init__(self, indexed_frame_stack, fps=None, start_frame=0, end_frame=None):
        self._indexed_frame_stack = indexed_frame_stack

        super().__init__(frame_source_func=FrameStack, indexed_frame_stack=self._indexed_frame_stack,
                         start_frame=start_frame, end_frame=end_frame)
        self.fps = fps if fps is not None else 30
        self.length = len(self.frame_source)


@copy_signature_from(directory_frame_reader)
class DirectoryFrameReader(FrameReader):
    def __init__(self, directory_path, fps=None, *args, **kwargs):
        self.indexed_path_list = get_indexed_frame_files_from_directory(directory_path, *args, **kwargs)
        super().__init__(frame_source_func=directory_frame_reader, frame_directory=None,
                         _indexed_path_list_override=self.indexed_path_list)
        self.fps = fps if fps is not None else 30
        self.length = len(self.indexed_path_list)

    def __getitem__(self, item):
        partial_indexed_path_list = self.indexed_path_list[item]
        if isinstance(item, int):
            partial_indexed_path_list = [partial_indexed_path_list]
        frame_source_output = [(frame_index, frame) for frame_index, frame in
                               self.get_frame_source(_indexed_path_list_override=partial_indexed_path_list)]

        if isinstance(item, int):
            return frame_source_output[0]
        return frame_source_output
