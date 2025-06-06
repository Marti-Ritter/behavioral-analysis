from time import perf_counter

import cv2
import pandas as pd
from ..utility.builtin_classes.funcs import apply_filtered_parameters
from ..utility.builtin_classes.iterables import ensure_list
from ..frame_pipeline.cv2_funcs import add_text_to_frame


class ImageTagger:
    """
    A class version of the image_tagger function. The class is initialized with the image paths, tag dictionary and
    an optional pre-existing tag dataframe and annotation dataframe. The class has a run method that runs the image
    tagger. The user is then able to tag the images or update the tag dataframe in a OpenCV window. The run method
    returns the updated tag dataframe after the user closes the window with the escape key.
    """

    def __init__(self, image_paths, tag_key_dict, tag_df=None, annotation_df=None, display_width=720, multi_choice=True,
                 title="Tagging images", text_overlay_dict=None):
        self._tag_key_dict = tag_key_dict
        self._annotation_df = annotation_df

        self._display_width = display_width
        self._multi_choice = multi_choice
        self._title = title

        self._output_df = pd.DataFrame(data=False, index=image_paths, columns=tag_key_dict.keys(), dtype=bool)
        if tag_df is not None:
            self.output_df = tag_df

        self._image_paths = image_paths
        self._image_stack = [cv2.imread(image_path) for image_path in self._image_paths]
        for i, image in enumerate(self._image_stack):
            width, height, depth = image.shape
            scaling_factor = display_width / width
            self._image_stack[i] = cv2.resize(image, dsize=None, fx=scaling_factor, fy=scaling_factor)

        self._text_overlay_dict = dict() if text_overlay_dict is None else text_overlay_dict
        self._image_index = 0

    @property
    def output_df(self):
        return self._output_df

    @output_df.setter
    def output_df(self, tag_df):
        assert tag_df.index.equals(self.output_df.index) and tag_df.columns.equals(self.output_df.columns), \
            "Input tag df has to match input paths and tags in index and columns."
        self._output_df = tag_df.copy()

    def run(self):
        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        cv2.setWindowTitle(self._title, "Tagging {}.".format(self._image_paths[self._image_index]))

        expand_gui = True
        while True:
            frame = self._image_stack[self._image_index].copy()
            frame_path = self._image_paths[self._image_index]

            if self._image_index in self._text_overlay_dict:
                static_texts = ensure_list(self._text_overlay_dict[self._image_index])
                for static_text in static_texts:
                    frame = add_text_to_frame(frame, **static_text)

            image_tags = self.output_df.loc[frame_path]
            if expand_gui:
                tag_string = "\'<\' / \'>\' previous/next, \'#\' to hide info\n"
                frame_string = "Frame index: " + str(self._image_index) + "/" + str(len(self._image_paths)-1)
            else:
                tag_string = "\'#\' to show info\n"
                frame_string = str(self._image_index) + "/" + str(len(self._image_paths)-1)
            for key, value in image_tags.to_dict().items():
                tag_string += f"{key}[\'{self._tag_key_dict[key]}\']: {value}\n" \
                    if expand_gui else f"[\'{self._tag_key_dict[key]}\']: {int(value)}\n"
            frame = add_text_to_frame(frame, tag_string, relative_org=(0.05, 0.05), font_scale=1,
                                      color=(255, 255, 255))
            frame = add_text_to_frame(frame, frame_string, relative_org=(0.8 if expand_gui else 0.85, 0.05),
                                      font_scale=1,
                                      color=(255, 255, 255))

            if self._annotation_df is not None:
                try:
                    annotation = self._annotation_df.loc[[frame_path]]
                    for _, text_series in annotation.iterrows():
                        text_dict = text_series.dropna().to_dict()
                        if "text" in text_dict.keys():
                            frame = apply_filtered_parameters(add_text_to_frame, frame=frame, **text_dict)
                except KeyError:
                    pass

            cv2.imshow(self._title, frame)

            wait_result = cv2.waitKey(1)
            for tag, key in self._tag_key_dict.items():
                if wait_result == ord(key):
                    if not self._multi_choice:
                        self.output_df.loc[frame_path] = False
                    self.output_df.loc[frame_path, tag] = ~self.output_df.loc[frame_path, tag]

            previous_frame, next_frame = wait_result == ord("<"), wait_result == ord(">")
            gui_expansion = wait_result == ord("#")
            if previous_frame or next_frame:
                if previous_frame:
                    self._image_index = min(self._image_index + 1, len(self._image_stack) - 1)
                elif next_frame:
                    self._image_index = max(self._image_index - 1, 0)
                cv2.setWindowTitle(self._title, "Tagging {}.".format(frame_path))
            elif gui_expansion:
                expand_gui = not expand_gui
            elif wait_result == 27:  # Esc key to stop
                break

        cv2.destroyAllWindows()
        return self.output_df


class VideoTagger:
    """
    A class version of the video_tagger function. The class is initialized with the video paths, tag dictionary and
    an optional pre-existing tag dataframe and annotation dataframe. The class has a run method that runs the video
    tagger. The user is then able to tag the videos or update the tag dataframe in a OpenCV window. The run method
    returns the updated tag dataframe after the user closes the window with the escape key.
    """

    def __init__(self, video_paths, tag_key_dict, tag_df=None, annotation_df=None, display_width=720, multi_choice=True,
                 title="Tagging videos", text_overlay_dict=None, tb_update_freq=30, video_speed_multiplicator=1.0):
        self._tag_key_dict = tag_key_dict
        self._annotation_df = annotation_df

        self._display_width = display_width
        self._multi_choice = multi_choice
        self._title = title

        self._output_df = ~pd.DataFrame(index=video_paths, columns=tag_key_dict.keys(), dtype=bool)
        if tag_df is not None:
            self.output_df = tag_df

        self._video_paths = video_paths
        self._video_index = 0

        self._text_overlay_dict = dict() if text_overlay_dict is None else text_overlay_dict
        self._tb_update_freq = tb_update_freq
        self._video_update_ms = 0
        self._video_speed_multiplicator = video_speed_multiplicator

    @property
    def output_df(self):
        return self._output_df

    @output_df.setter
    def output_df(self, tag_df):
        assert (tag_df.index == self.output_df.index) and (tag_df.columns == self.output_df.columns), \
            "Input tag df has to match input paths and tags in index and columns."
        self._output_df = tag_df.copy()

    def run(self):
        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        cv2.setWindowTitle(self._title, "Tagging {}.".format(self._video_paths[self._video_index]))
        video = cv2.VideoCapture(self._video_paths[self._video_index])
        self._video_update_ms = int(1 / video.get(cv2.CAP_PROP_FPS) * 1000 / self._video_speed_multiplicator)
        nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if nr_of_frames == 0:
            raise ValueError("Video {} contains zero frames!".format(self._video_paths[self._video_index]))
        scaling_factor = self._display_width / video.get(cv2.CAP_PROP_FRAME_WIDTH)
        cv2.createTrackbar("Frame", self._title, 0, nr_of_frames, lambda x: video.set(cv2.CAP_PROP_POS_FRAMES, x))

        tb_update_count = 0
        previous_frame = [[0]]
        last_frame_time = perf_counter()

        expand_gui = True
        while True:
            ret, frame = video.read()
            if not ret:
                frame = previous_frame.copy()
            else:
                previous_frame = frame.copy()

            frame = cv2.resize(frame, dsize=None, fx=scaling_factor, fy=scaling_factor)

            if self._video_index in self._text_overlay_dict:
                static_texts = ensure_list(self._text_overlay_dict[self._video_index])
                for static_text in static_texts:
                    frame = add_text_to_frame(frame, **static_text)

            video_tags = self.output_df.loc[self._video_paths[self._video_index]]

            if expand_gui:
                tag_string = "\'<\' / \'>\' previous/next, \'#\' to hide info, use \'ESC\' to close!\n"
                video_string = "Video index: " + str(self._video_index)
            else:
                tag_string = "\'#\' to show info\n"
                video_string = str(self._video_index)
            for key, value in video_tags.to_dict().items():
                tag_string += f"{key}[\'{self._tag_key_dict[key]}\']: {value}\n" \
                    if expand_gui else f"[\'{self._tag_key_dict[key]}\']: {int(value)}\n"
            frame = add_text_to_frame(frame, tag_string, relative_org=(0.05, 0.05), font_scale=0.5, color=(255, 255, 255))
            frame = add_text_to_frame(frame, video_string, relative_org=(0.8 if expand_gui
                                                                        else 0.9, 0.05), font_scale=0.5,
                                        color=(255, 255, 255))

            if self._annotation_df is not None:
                try:
                    annotation = self._annotation_df.loc[[self._video_paths[self._video_index]]]
                    for _, text_series in annotation.iterrows():
                        text_dict = text_series.dropna().to_dict()
                        frame_start = text_dict.pop("frame_start") if "frame_start" in text_dict else 0
                        frame_end = text_dict.pop("frame_end") if "frame_end" in text_dict else (nr_of_frames + 1)
                        if frame_start <= int(
                                video.get(cv2.CAP_PROP_POS_FRAMES)) < frame_end and "text" in text_dict.keys():
                            frame = apply_filtered_parameters(add_text_to_frame, frame=frame, **text_dict)
                except KeyError:
                    pass

            cv2.imshow(self._title, frame)

            current_frame_time = perf_counter()
            passed_ms = int((current_frame_time - last_frame_time) * 1000)
            remaining_wait_time = max(1, self._video_update_ms - passed_ms)
            wait_result = cv2.waitKey(remaining_wait_time)
            last_frame_time = current_frame_time

            for tag, key in self._tag_key_dict.items():
                if wait_result == ord(key):
                    if not self._multi_choice:
                        self.output_df.loc[self._video_paths[self._video_index]] = False
                    self.output_df.loc[self._video_paths[self._video_index], tag] = ~self.output_df.loc[
                        self._video_paths[self._video_index], tag]

            previous_video, next_video = wait_result == ord("<"), wait_result == ord(">")
            gui_expansion = wait_result == ord("#")
            if previous_video or next_video:
                if previous_video:
                    self._video_index = min(self._video_index + 1, len(self._video_paths) - 1)
                elif next_video:
                    self._video_index = max(self._video_index - 1, 0)
                video.release()
                video = cv2.VideoCapture(self._video_paths[self._video_index])
                self._video_update_ms = int(1 / video.get(cv2.CAP_PROP_FPS) * 1000 / self._video_speed_multiplicator)
                nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                if nr_of_frames == 0:
                    raise ValueError("Video {} contains zero frames!".format(self._video_paths[self._video_index]))
                scaling_factor = self._display_width / video.get(cv2.CAP_PROP_FRAME_WIDTH)

                cv2.setWindowTitle(self._title, "Tagging {}.".format(self._video_paths[self._video_index]))
                cv2.setTrackbarMax("Frame", self._title, nr_of_frames)
                cv2.setTrackbarPos("Frame", self._title, 0)

                tb_update_count = -1

            elif gui_expansion:
                expand_gui = not expand_gui
            elif wait_result == 27:
                break

            tb_update_count += 1
            if self._tb_update_freq is not None and tb_update_count >= self._tb_update_freq:
                tb_update_count = 0
                cv2.setTrackbarPos("Frame", self._title, int(video.get(cv2.CAP_PROP_POS_FRAMES)))

        cv2.destroyAllWindows()
        return self.output_df
