from .base_classes import FrameProcessor, FrameGenerator
from .processor_funcs import (annotated_frames_generator, frame_roi_extractor,
                              frame_heading_corrector, frame_reoriented_roi_extractor,
                              func_modified_frames_generator,
                              text_annotated_frames_generator, rgba_annotated_frames_generator,
                              merge_frame_sources)
from .reader_classes import FrameStackReader
from ..utility.general import copy_signature_from

"""
This is a set of objects that allow for the sequential processing of frames.

The basic source objects (the Cv2FrameReader and FrameDirectoryReader allow for slicing. Slicing can be used to debug or
while working in a notebook. All objects also being generators helps when you want to write a large source (video or
frame directory) with modifications to a file or show it in a notebook without loading the entire source to memory.
"""


def frame_processor_factory(processor_func):
    @copy_signature_from(processor_func)
    class AppliedFrameProcessor(FrameProcessor):
        def __init__(self, frame_source, *args, **kwargs):
            super().__init__(frame_source, processor_func, *args, **kwargs)
    return AppliedFrameProcessor


FrameRoiExtractor = frame_processor_factory(frame_roi_extractor)
FrameHeadingCorrector = frame_processor_factory(frame_heading_corrector)
FrameReorientedRoiExtractor = frame_processor_factory(frame_reoriented_roi_extractor)

FuncModifiedFramesGenerator = frame_processor_factory(func_modified_frames_generator)
TextAnnotatedFramesGenerator = frame_processor_factory(text_annotated_frames_generator)
RgbaAnnotatedFramesGenerator = frame_processor_factory(rgba_annotated_frames_generator)
AnnotatedFramesGenerator = frame_processor_factory(annotated_frames_generator)


@copy_signature_from(merge_frame_sources)
class FrameSourceMerger(FrameProcessor):
    def __init__(self, frame_sources_list, fps_override=None, length_limit_source_index=None,
                 frame_index_source_index=None, *args, **kwargs):
        frame_sources_list = [
            frame_source if isinstance(frame_source, FrameGenerator) else FrameStackReader(frame_source) for
            frame_source in frame_sources_list]  # ensure that the source objects are compatible types

        super().__init__(frame_source=frame_sources_list, processor_func=merge_frame_sources,
                         length_limit_source_index=length_limit_source_index,
                         frame_index_source_index=frame_index_source_index, *args, **kwargs)

        fps_list = [frame_source.fps for frame_source in frame_sources_list if hasattr(frame_source, "fps")]
        fps = fps_override if fps_override is not None else fps_list[0]
        self.fps = fps if fps is not None else 30

        length_list = [len(frame_source) if hasattr(frame_source, "__len__") else 0 for frame_source in
                       frame_sources_list]
        self.length = length_list[length_limit_source_index] if length_limit_source_index is not None else max(
            length_list)

    def __iter__(self):
        # as we replace frame_source with the modified generator, we have to access the unmodified source here
        new_source_list = []
        for frame_source in self.unmodified_source:
            new_source_list.append(
                frame_source.get_frame_source() if hasattr(frame_source, "get_frame_source") else frame_source)
        self.frame_source = self.get_frame_source(frame_source=new_source_list)
        return self

    def __getitem__(self, item):
        sliced_frame_source_output = []

        for single_source in self.unmodified_source:
            if hasattr(single_source, "__getitem__"):
                single_sliced_frame_source_output = single_source[item]
                if not isinstance(single_sliced_frame_source_output, list):
                    single_sliced_frame_source_output = [single_sliced_frame_source_output]
                sliced_frame_source_output.append(FrameStackReader(single_sliced_frame_source_output))
            else:
                raise NotImplementedError(f"Cannot index into frame source of type {type(single_source)}")

        processed_output = [(frame_index, frame) for frame_index, frame in
                            self.partial_processor_func(frame_source=sliced_frame_source_output)]

        if isinstance(item, int):
            return processed_output[0]
        return processed_output
