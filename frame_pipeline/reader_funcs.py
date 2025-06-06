import pathlib
import re

import cv2


def cv2_video_frame_reader(video_path, start_frame=0, end_frame=None, fps_override=None,
                           read_speed=1, jump_to_start=False):
    """
    A generator that yields frames from a video file. Adapted from https://stackoverflow.com/a/69312152.
    Both start and end frame are inclusive, so reading from frame 10 to 199 will result in 190 frames read.

    :param video_path: A path to a video file.
    :type video_path: str
    :param start_frame: The index of the first frame to read. Default is 0.
    :type start_frame: int
    :param end_frame: The index of the last frame to read. If None, the last frame of the video is read.
    :type end_frame: int
    :param fps_override: Overrides the fps of the video. If None, the original fps of the video is used.
    :type fps_override: float
    :param read_speed: The speed at which the frames are read. If 1, the frames are read at the fps or fps_override. If 2, the
        frames are read at twice the fps or fps_override. If 0.5, reads the frames at half the fps orfps_override.
    :type read_speed: float
    :param jump_to_start: If True, jumps the video to the start_frame. If False, the video is read from the
        beginning and the first frames are skipped until the start_frame. If frames are missing, this will
        not work as expected.
    :type jump_to_start: bool
    :return: A generator that yields frames from a video file.
    :rtype: (int, np.ndarray) or np.ndarray
    """
    cap = cv2.VideoCapture(video_path)

    try:
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        fps_out = read_speed * (fps_override if fps_override is not None else fps_in)

        initial_frame_index = start_frame * int((fps_override / fps_in) if fps_override is not None else 1)
        set_cv2_video_capture_to_frame(cap, initial_frame_index, jump_to_start=jump_to_start)

        end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if end_frame is None else end_frame

        index_out = index_in = start_frame

        cap.grab()  # to ensure that cap is ready for retrieve at start frame 0
        while index_in <= end:
            out_due = int((index_in - start_frame) / fps_out * fps_in) + start_frame
            if out_due >= index_out:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                index_out += 1
                yield (index_in, frame)
                continue

            ret = cap.grab()
            if not ret:
                break
            index_in += 1
    finally:
        cap.release()


def get_indexed_frame_files_from_directory(frame_directory, frame_file_extension=".png", frame_index_pattern=None,
                                          start_frame=0, end_frame=None):
    frame_index_pattern = r"(\d+)" + frame_file_extension if frame_index_pattern is None else frame_index_pattern

    directory_path = pathlib.Path(frame_directory)
    found_files = list(directory_path.glob("*" + frame_file_extension))

    frame_index_list = [int(re.search(frame_index_pattern, str(frame_path)).groups()[0]) for frame_path in found_files]
    sorted_index_path_list = sorted(zip(frame_index_list, found_files))

    def filter_condition(frame_index):
        if frame_index < start_frame:
            return False
        if (end_frame is not None) and (frame_index > end_frame):
            return False
        return True

    filtered_index_path_list = [(frame_index, frame_path) for frame_index, frame_path in sorted_index_path_list if
                                filter_condition(frame_index)]
    return filtered_index_path_list


def directory_frame_reader(frame_directory, frame_file_extension=".png", sort_key=None,
                           start_frame=0, end_frame=None, _indexed_path_list_override=None):
    if _indexed_path_list_override is None:
        indexed_path_list = get_indexed_frame_files_from_directory(frame_directory,
                                                                   frame_file_extension=frame_file_extension,
                                                                   sort_key=sort_key, start_frame=start_frame,
                                                                   end_frame=end_frame)
    else:
        indexed_path_list = _indexed_path_list_override

    for frame_index, frame_path in indexed_path_list:
        yield frame_index, cv2.imread(frame_path)


def set_cv2_video_capture_to_frame(cv2_video_capture, target_frame, jump_to_start=False):
    current_frame = int(cv2_video_capture.get(cv2.CAP_PROP_POS_FRAMES))
    frames_to_skip = target_frame - current_frame
    if not jump_to_start and not (frames_to_skip < 0):
        for _ in range(frames_to_skip):
            ret = cv2_video_capture.grab()
            if not ret:
                break
    else:
        cv2_video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
