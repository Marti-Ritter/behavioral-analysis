import numpy as np
import pandas as pd

import warnings

from ..pandas_tools.funcs import extract_neighbourhood_df
from ..math_tools.angle_funcs import rotate_2d_point_groups_np, cart2pol
from ..math_tools.matrix_funcs import apply_transform_matrix


def extract_subtrack_df(input_track_df, subtrack_reference_indices, span_before=0, span_after=0,
                        subtrack_end_indices=None, span_index_level=-1):
    """
    Extract a subtrack from a track dataframe. Uses either the subtrack_reference_indices and the subtrack_end_indices to extract 
    the subtrack or the span_before and span_after the subtrack_reference_indices. If the subtrack_end_indices is not provided, the
    span_before and span_after will be used to extract the subtrack. The span_index_level is used to specify the level of the index
    to be used for the span_before and span_after. If the subtrack_end_indices is provided, the span_index_level is ignored.

    See extract_neighbourhood_df for more details.

    :param input_track_df: The input track dataframe. Can have a multiindex, but the subtrack_reference_indices and the subtrack_end_indices
    must match the shape of the multiindex.
    :type input_track_df: pandas.DataFrame
    :param subtrack_reference_indices: The reference indices to extract the subtrack from the input_track_df.
    :type subtrack_reference_indices: pd.Index or pd.MultiIndex
    :param span_before: The number of values to extract before the center indices, defaults to 0. Ignored if subtrack_end_indices is provided.
    :type span_before: int, optional
    :param span_after: The number of values to extract after the center indices, defaults to 0. Ignored if subtrack_end_indices is provided.
    :type span_after: int, optional
    :param subtrack_end_indices: The end indices of the subtracks, defaults to None. If None, the span_before and span_after
    parameters will be used to calculate the end indices. If not None, then each span will be calculated as the range
    from the start index to the end index
    :type subtrack_end_indices: pd.Index or pd.MultiIndex or None, optional
    :param span_index_level: The level of the MultiIndex to use for the span, defaults to -1. Do not change from the default -1 if the
    Series has only a simple Index. If the input_track_df follows the correct schema, then the last index should be the data index.
    :type span_index_level: int, optional
    :return: The subtrack dataframe extracted from the input_track_df.
    :rtype: pandas.DataFrame
    """

    subtrack_df = extract_neighbourhood_df(input_series_or_df=input_track_df, neighbourhood_reference_indices=subtrack_reference_indices,
                                           span_before=span_before, span_after=span_after, neighbourhood_end_indices=subtrack_end_indices,
                                           span_index_level=span_index_level)
    subtrack_df = subtrack_df.stack("neighbourhood_index", future_stack=True)
    
    index_names_list = list(subtrack_df.index.names)
    index_level_rename_dict = dict(frame_index="reference_frame_index", neighbourhood_index="frame_index")
    index_names_list = [index_level_rename_dict.get(name, name) for name in index_names_list]
    subtrack_df.index.names = index_names_list

    return subtrack_df


def normalize_track_df(input_track_df, norm_df):
    """
    Normalize the input_track_df using the norm_df. The norm_df should have the columns x, y, and heading, as well as the same index as the
    input_track_df. The input_track_df should have the expected schema of having the keypoint_feature as the first level of the columns. The 
    input_track_df will be normalized by subtracting the x and y values from the norm_df. The heading values will be used to rotate the x and 
    y values of the input_track_df. Any other values in the input_track_df will be left unchanged.

    :param input_track_df: The input track dataframe to be normalized.
    :type input_track_df: pandas.DataFrame
    :param norm_df: The dataframe to normalize the input_track_df. Should have the columns x, y, and heading and the same index as the input_track_df.
    :type norm_df: pandas.DataFrame
    :return: The normalized track dataframe.
    :rtype: pandas.DataFrame
    """

    norm_df = norm_df.rename_axis("keypoint_feature", axis=1)
    track_df = (input_track_df - norm_df[["x", "y"]]).fillna(input_track_df)

    x_values = track_df.xs("x", level="keypoint_feature", axis=1).values
    y_values = track_df.xs("y", level="keypoint_feature", axis=1).values
    original_values = np.dstack([x_values, y_values])

    norm_headings = norm_df["heading"].reindex(index=track_df.index)
    rotated_points = rotate_2d_point_groups_np(original_values, origins=np.array((0, 0)), angles=-norm_headings.values)
    rotated_x = rotated_points[:, :, 0]
    rotated_y = rotated_points[:, :, 1]

    track_df.loc[:, pd.IndexSlice[:, "x"]] = rotated_x
    track_df.loc[:, pd.IndexSlice[:, "y"]] = rotated_y

    return track_df


def apply_transform_matrix_to_track_df(input_track_df, transform_matrix):
    input_keypoint_df = input_track_df.stack("keypoint_name", future_stack=True)
    input_keypoint_df[["x", "y"]] = apply_transform_matrix(input_keypoint_df[["x", "y"]].values, transform_matrix)
    transformed_track_df = input_keypoint_df.unstack("keypoint_name").reorder_levels(["keypoint_name", "keypoint_feature"], axis=1)
    return transformed_track_df.reindex(columns=input_track_df.columns)


def aggregate_keypoint_df(input_keypoint_df, aggregation_function="median", levels_to_aggregate=None):
    if levels_to_aggregate is None:
        levels_to_aggregate = -1
    agg_keypoint_df = input_keypoint_df.groupby(level=levels_to_aggregate).agg(aggregation_function)
    return agg_keypoint_df


def track_to_centroid_df(track_df, aggregate_func="median"):
    track_x_df, track_y_df = (track_df.xs("x", level="keypoint_feature", axis=1),
                              track_df.xs("y", level="keypoint_feature", axis=1))
    agg_x, agg_y = track_x_df.agg(aggregate_func, axis=1), track_y_df.agg(aggregate_func, axis=1)
    return pd.DataFrame(dict(x=agg_x, y=agg_y))


def track_to_roi_df(track_df, size_override=None, aggregate_func="median", size_quantile=0.99, size_multiplier=1.05):
    track_x_df, track_y_df = (track_df.xs("x", level="keypoint_feature", axis=1),
                              track_df.xs("y", level="keypoint_feature", axis=1))
    centroid_df = track_to_centroid_df(track_df, aggregate_func=aggregate_func)
    centroid_x, centroid_y = centroid_df["x"], centroid_df["y"]

    if size_override is None:
        width = (track_x_df.T - centroid_x).T.abs().quantile(size_quantile).max()
        height = (track_y_df.T - centroid_y).T.abs().quantile(size_quantile).max()
        square_size = max(width, height) * size_multiplier
        width, height = square_size, square_size
    else:
        width, height = size_override

    return pd.DataFrame(dict(x=centroid_x - width//2, y=centroid_y - height//2, w=width, h=height))


def track_to_distance_orientation_df(track_df, orientation_reference_point, position_reference_point,
                                     orientation_in_degrees=True):
    sliced_track_df = track_df.loc[:, pd.IndexSlice[:, ["x", "y"]]]

    if isinstance(orientation_reference_point, pd.DataFrame):
        orientation_reference_df = orientation_reference_point
    else:
        orientation_reference_df = sliced_track_df.loc[:, orientation_reference_point]

    if isinstance(position_reference_point, pd.DataFrame):
        position_reference_df = position_reference_point
    else:
        position_reference_df = sliced_track_df.loc[:, position_reference_point]

    normalized_orientation_df = orientation_reference_df - position_reference_df
    length_orientation_df = normalized_orientation_df.apply(lambda r: cart2pol(r["x"], r["y"]), axis=1).apply(pd.Series)
    length_orientation_df.columns = ["rho", "phi"]

    if orientation_in_degrees:
        length_orientation_df["phi"] = length_orientation_df["phi"].apply(np.rad2deg)

    return length_orientation_df


def track_to_norm_df(track_df, orientation_reference_point, position_reference_point=None, aggregate_func="median"):
    if position_reference_point is None:
        position_reference_df = track_to_centroid_df(track_df, aggregate_func=aggregate_func)
    elif isinstance(position_reference_point, pd.DataFrame):
        position_reference_df = position_reference_point
    else:
        position_reference_df = track_df.loc[:, position_reference_point]

    length_orientation_df = track_to_distance_orientation_df(track_df,
                                                             orientation_reference_point=orientation_reference_point,
                                                             position_reference_point=position_reference_df)

    return position_reference_df.join(length_orientation_df["phi"].rename("heading"))


def kpms_style_example_trajectories(track_df, bout_df, minimum_bout_length=3, pre=5, post=15,
                                    pca_samples=50000, pca_dim=4, n_neighbors=50, num_samples=50,
                                    random_state=None):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    bout_df = bout_df[bout_df["bout_length"].ge(minimum_bout_length)]
    behavior_subtrack_df = pd.concat({behavior: extract_subtrack_df(
        track_df, behavior_bout_df.index, span_before=pre, span_after=post
    ) for behavior, behavior_bout_df in bout_df.groupby("behavior_name")}, names=["behavior_name"])

    rng = np.random.default_rng(random_state)

    X = behavior_subtrack_df.loc[:, pd.IndexSlice[:, ["x", "y"]]].unstack("frame_index")
    X = X.interpolate(method="linear", axis=0, limit_direction="both")

    if X.shape[0] > pca_samples:
        X = X.sample(n=pca_samples, random_state=rng)
    else:
        warnings.warn(f"Unable to get {pca_samples} samples from {X.shape[0]} bouts")

    pca = PCA(n_components=pca_dim).fit(X)
    X_pca = pca.transform(X)
    all_nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_pca)

    sampled_instances = {}

    for behavior_name, behavior_bouts in behavior_subtrack_df.groupby("behavior_name"):
        behavior_bouts = behavior_bouts.unstack("frame_index")

        if n_neighbors > len(behavior_bouts):
            continue

        X = behavior_bouts.copy()
        X = X.loc[:, pd.IndexSlice[:, ["x", "y"]]]
        X = X.interpolate(method="linear", axis=0, limit_direction="both")

        X_pca = pca.transform(X)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_pca)
        distances, indices = nbrs.kneighbors(X_pca)
        local_density = 1 / distances.mean(1)

        distances, _ = all_nbrs.kneighbors(X_pca)
        global_density = 1 / distances.mean(1)
        exemplar = np.argmax(local_density / global_density)
        samples = rng.choice(indices[exemplar], num_samples, replace=False)
        sampled_instances[behavior_name] = behavior_bouts.iloc[samples]
    return pd.concat(sampled_instances.values(), axis=0).stack("frame_index", future_stack=True)


def sampled_trajectories_to_bout_df(sampled_trajectory_df):
    bout_df_list = []
    for behavior_name, behavior_bouts in sampled_trajectory_df.groupby("behavior_name"):
        unique_indices = behavior_bouts.index.droplevel(["behavior_name", "frame_index"]).unique()
        unique_indices = unique_indices.rename("frame_index", level=-1)
        bout_df_list.append(pd.DataFrame(dict(end_frame=np.nan, bout_length=np.nan, behavior_name=behavior_name),
                                         index=unique_indices))
    return pd.concat(bout_df_list, axis=0)
