import os

import cv2
import pandas as pd
from tqdm.auto import tqdm
from ..visualization.video_funcs import get_frames


def dlc_el_pickle_to_trace_df_dict(el_pickle_path):
    """
    This function takes a path to a DLC _el.pickle file and returns a dictionary of dataframes, where each dataframe
    contains the x and y coordinates of a single tracklet. The keys of the dictionary are the tracklet names, and the
    index of each dataframe is the frame number.

    :param el_pickle_path: The path to the DLC _el.pickle file
    :type el_pickle_path: str
    :return: A dictionary of dataframes, where each dataframe contains the x and y coordinates of a single trace
    :rtype: dict of pandas.DataFrame
    """

    el_pkl = pd.read_pickle(el_pickle_path)
    columns = el_pkl.pop("header")

    trace_df_dict = {}
    for trace, trace_rows in tqdm(el_pkl.items()):
        trace_df_rows = {}
        for row, row_data in trace_rows.items():
            trace_df_rows[row] = row_data[:, :-1].reshape(1, -1).squeeze()
        trace_df_dict[trace] = pd.DataFrame.from_dict(trace_df_rows, orient="index", columns=columns)

    for trace_df in trace_df_dict.values():
        trace_df.index = trace_df.index.str.extract(r"(\d+)", expand=False).astype(int)

    return trace_df_dict


def extract_dlc_labels_to_files(dlc_format_labels, output_root, save_h5=True, save_csv=True, save_pkl=True):
    for idx, idx_frame in tqdm(dlc_format_labels.groupby(level=[0, 1])):
        scorer = idx_frame.columns.get_level_values("scorer").unique()[0]
        df_out_dir = os.path.join(output_root, *idx)
        df_out_path = os.path.join(df_out_dir, f"CollectedData_{scorer}")

        if save_h5:
            idx_frame.to_hdf(df_out_path + ".h5", key="keypoints", mode="w")
        if save_csv:
            idx_frame.to_csv(df_out_path + ".csv")
        if save_pkl:
            idx_frame.to_pickle(df_out_path + ".pkl")


def extract_dlc_frames_from_videos(dlc_format_labels, output_root, video_source_path_dict, frame_file_index_dict):
    for idx, idx_frame in tqdm(dlc_format_labels.groupby(level=[0, 1])):
        video_path = video_source_path_dict[idx[1]]
        df_out_dir = os.path.join(output_root, *idx)

        # check if all frames are already extracted
        for frame_file in idx_frame.index.get_level_values(2):
            frame_path = os.path.join(df_out_dir, frame_file)
            if not os.path.isfile(frame_path):
                frame_idx = frame_file_index_dict[frame_file]
                frame = get_frames(video_path, frame_idx)[0]
                cv2.imwrite(frame_path, frame)


def dlc_to_tracking_df(input_dlc_df):
    dlc_predictions_df = input_dlc_df.droplevel(level="scorer", axis=1)
    _index_df = dlc_predictions_df.index.to_series().apply(lambda x: os.path.basename(x)).str.split("_", expand=True)
    _index_df.columns = ["video_name", "frame_index"]
    _index_df["frame_index"] = _index_df["frame_index"].str.extract("(\d+)").astype(int)
    dlc_predictions_df.index = pd.MultiIndex.from_frame(_index_df)
    dlc_predictions_df = dlc_predictions_df.stack("individuals", future_stack=True).reorder_levels([0, 2, 1], axis=0)
    dlc_predictions_df = dlc_predictions_df.rename_axis(["keypoint_name", "keypoint_feature"], axis=1)
    dlc_predictions_df = dlc_predictions_df.rename({"likelihood": "score"}, axis=1, level="keypoint_feature")
    return dlc_predictions_df
