from math_tools.angle_funcs import rotate_2d_point_groups_np


import numpy as np
import pandas as pd


def normalize_trajectory_df(trajectory_df, norm_df, normalized_to_trajectory_starts=True):
    """

    :param trajectory_df:
    :param norm_df:
    :param normalized_to_trajectory_starts:
    :return:
    """
    normalized_trajectory_df = trajectory_df.stack("neighbourhood_index", future_stack=True)

    original_index = normalized_trajectory_df.index
    if not normalized_to_trajectory_starts:
        corrected_index = original_index.to_frame()
        corrected_index["frame_index"] += corrected_index["neighbourhood_index"]
        corrected_index = pd.MultiIndex.from_frame(corrected_index)
        normalized_trajectory_df.index = corrected_index

    normalized_trajectory_df = normalized_trajectory_df.stack("keypoint_name", future_stack=True)

    normalized_trajectory_df[["x", "y"]] -= norm_df[["x", "y"]]
    normalized_trajectory_df = normalized_trajectory_df.unstack("keypoint_name").reorder_levels(["keypoint_name", "keypoint_feature"],
                                                                                           axis=1)

    x_values = normalized_trajectory_df.loc[:, pd.IndexSlice[:, "x"]].values
    y_values = normalized_trajectory_df.loc[:, pd.IndexSlice[:, "y"]].values
    original_values = np.dstack([x_values, y_values])

    norm_headings = norm_df["heading"].reindex(index=normalized_trajectory_df.index)
    rotated_points = rotate_2d_point_groups_np(original_values, origins=np.array((0, 0)), angles=-norm_headings.values)
    rotated_x = rotated_points[:, :, 0]
    rotated_y = rotated_points[:, :, 1]

    normalized_trajectory_df.loc[:, pd.IndexSlice[:, "x"]] = rotated_x
    normalized_trajectory_df.loc[:, pd.IndexSlice[:, "y"]] = rotated_y

    if not normalized_to_trajectory_starts:
        normalized_trajectory_df.index = original_index

    return normalized_trajectory_df.unstack("neighbourhood_index")


def trajectory_df_to_keypoint_df(trajectory_df, summary_function="median"):
    """
    A function to translate a trajectory dataframe to a keypoint dataframe. If no summary function is provided, the
    median is used as default. If summary_function is set to None, the resulting trajectory dataframe will have the
    same leading indices as the levels of the input trajectory_df (i.e. the identifiers of that specific trajectory).

    :param trajectory_df: The trajectory dataframe to be translated.
    :type trajectory_df: pd.DataFrame
    :param summary_function: A function compatible with the pandas groupby.agg method to summarize the trajectory data, 
    defaults to "median"
    :type summary_function: str or None, optional
    :return: The keypoint dataframe
    :rtype: pd.DataFrame
    """
    trajectory_df = trajectory_df.stack("keypoint_name", future_stack=True)
    if "score" in trajectory_df.columns.get_level_values("keypoint_feature"):
        trajectory_df = trajectory_df.drop("score", level="keypoint_feature", axis=1)
    if "instance" in trajectory_df.index.get_level_values("keypoint_name"):
        trajectory_df = trajectory_df.drop("instance", level="keypoint_name", axis=1)
    if summary_function is not None:
        trajectory_df = trajectory_df.groupby("keypoint_name").agg(summary_function)
    else:
        trajectory_df = trajectory_df.stack("keypoint_name", future_stack=True)
    return trajectory_df