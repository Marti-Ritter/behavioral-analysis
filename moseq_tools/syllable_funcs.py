import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.spatial.distance import pdist, squareform
from tqdm.auto import tqdm
from ..tracking_tools.plots import normalize_keypoint_df_for_plotting

from ..utility.builtin_classes.iterables import filter_consecutive_duplicates
from ..utility.files.hdf5_tools import read_data_from_h5_file

from ..math_tools.angle_funcs import rotate_2d_point_groups_np
from ..math_tools.vector_funcs import calculate_velocity_2d, calculate_angular_speed
from ..pandas_tools.funcs import (pd_reindex_to_index_union, consecutive_boolean_counter, expand_index_from_limits)
from ..pandas_tools.old_code import flat_index_series_to_value_chunks, series_to_value_chunks, series_to_numeric_chunks


def get_syllable_transitions(syllable_series):
    shifted_series = syllable_series.shift(1)
    transition_series = syllable_series[syllable_series.ne(shifted_series)].copy()
    return transition_series


def get_syllable_sequence_series(syllable_series, sequence_length=1):
    if sequence_length > 1:
        syllable_series = pd.Series(
            [tuple(w) for w in syllable_series.rolling(window=sequence_length)][sequence_length - 1:])
    return syllable_series


def n_gram_syllable_transition_series(syllable_series, n=2, sequence_length=1, sequence_overlap=0, max_syllable_id=99,
                                      reindex_to_full=False):
    target_dtype = object if sequence_length > 1 else int
    syllable_series = get_syllable_sequence_series(syllable_series, sequence_length)
    shift_df = pd.DataFrame(
        {f"element{shift}": syllable_series.shift(-shift * (sequence_length + sequence_overlap)).iloc[:-n + 1] for shift
         in range(n)}, dtype=target_dtype)
    n_grams = shift_df.value_counts()
    max_syllable_filter = n_grams.index.to_frame().apply(lambda x: pd.Series(np.hstack(x)), axis=1).lt(
        max_syllable_id).all(axis=1)
    n_grams = n_grams[max_syllable_filter]
    if reindex_to_full:
        if sequence_length > 1:
            sequence_combinations_list = list(combinations(range(max_syllable_id), sequence_length))
        else:
            sequence_combinations_list = range(max_syllable_id)
        try:
            full_index = pd.MultiIndex.from_product([sequence_combinations_list] * n, names=shift_df.columns)
            n_grams = n_grams.reindex(full_index, fill_value=0)
        except MemoryError:
            warnings.warn("Memory error occurred, returning n_grams without reindexing! Please consider setting "
                          "reindex_to_full=False, or reducing sequence_length or max_syllable_id.")
    return n_grams


def reduce_transition_series_index(transition_series):
    def _reduction_func(x):
        if not hasattr(x, "__iter__"):
            return x
        elif len(set(x)) == 1:
            return x[0]
        else:
            return x

    transition_series = transition_series.copy()
    reduced_index = transition_series.index.to_frame().map(_reduction_func)
    transition_series.index = pd.MultiIndex.from_frame(reduced_index)
    if transition_series.index.nlevels == 1:
        transition_series.index = transition_series.index.levels[0]
    return transition_series


def syllable_transition_series_to_frame(transition_series, outgoing_levels=-1):
    return transition_series.unstack(level=outgoing_levels, fill_value=0)


def trim_syllable_transition_frame_to_monodirectional(transition_frame):
    """
    Trim a syllable transition frame to only contain the higher value of each pair of syllables. This is useful to check
    for monodirectional transitions in a bidirectional transition frame and simplify the frame for further analysis if
    only monodirectional transitions are of interest. Perfectly bidirectional transitions will be removed.
    :param transition_frame: A transition frame containing the transition counts/proportions between syllables.
    :type transition_frame: pd.DataFrame
    :return: A trimmed transition frame containing only the higher value of each pair of syllables.
    :rtype: pd.DataFrame
    """
    transition_frame = transition_frame.copy()
    transition_frame = transition_frame.where(transition_frame.gt(transition_frame.T), 0)
    return transition_frame


def normalize_syllable_transition_series(transition_series, normalize):
    """
    Keep in mind that the normalization of transition series containing sequences of length greater than 1 does not take
    into account the lost information due to the sequence length. To get the sequences a shift is applied and any rows
    containing shift-induced NaN values are removed. This means that the normalization is not perfect and may not
    accurately represent the true transition probabilities of the original data. The number of lost rows always relates
    to the sequence length by (sequence_length - 1) * n.

    :param transition_series:
    :type transition_series:
    :param normalize:
    :type normalize:
    :return:
    :rtype:
    """
    transition_series = transition_series.copy()
    if normalize is None:
        return transition_series
    elif normalize == "bigram":
        transition_series /= transition_series.sum()
    elif normalize == "rows":
        transition_series /= transition_series.groupby(level=0).sum()
    elif normalize == "columns":
        transition_series /= transition_series.groupby(level=-1).sum()
    elif isinstance(normalize, int):
        transition_series /= transition_series.groupby(level=normalize).sum()
    else:
        raise ValueError(f"Invalid normalization method specified: {normalize}")
    return transition_series


def get_syllable_transition_series(
        *input_series,
        normalize="bigram",
        smoothing=0.0,
        combine=False,
        **transition_kwargs
):
    all_series = []

    default_transition_kwargs = dict(max_syllable_id=100, n=2, sequence_length=1, sequence_overlap=0,
                                     reindex_to_full=False)
    transition_kwargs = {**default_transition_kwargs, **transition_kwargs}

    for syllable_series in input_series:
        # Get syllable transitions
        transitions = get_syllable_transitions(syllable_series)
        transition_series = n_gram_syllable_transition_series(transitions, **transition_kwargs) + (
            smoothing if not combine else 0)
        all_series.append(transition_series)

    if combine:
        # for combined transition series, sum all series and normalize
        all_series = pd_reindex_to_index_union(all_series, fill_value=0)
        all_series = sum(all_series)
        all_series = normalize_syllable_transition_series(all_series, normalize)
    else:
        # for individual transition matrices, normalize each matrix
        all_series = [normalize_syllable_transition_series(series, normalize) for series in all_series]

    return all_series


def get_syllable_frequencies(*input_series, masks=None, num_states=None, runlength=True):
    if masks is not None:
        input_series = [syllable_series[mask] for syllable_series, mask in zip(input_series, masks)]

    concatenated_series = pd.concat(input_series)

    if num_states is None:
        num_states = concatenated_series.max() + 1

    state_range = range(concatenated_series.min(), num_states)

    if runlength:
        chunks_list = [flat_index_series_to_value_chunks(syllable_series)["value"] for syllable_series in input_series]
        series_to_count = pd.concat(chunks_list)
    else:
        series_to_count = concatenated_series
    return series_to_count.value_counts(normalize=True, sort=False).reindex(state_range, fill_value=0)


def get_sequence_frequencies(*input_series, masks=None, runlength=True):
    if masks is not None:
        input_series = [sequence_series[mask] for sequence_series, mask in zip(input_series, masks)]

    concatenated_series = pd.concat(input_series)

    if runlength:
        chunks_list = [flat_index_series_to_value_chunks(sequence_series)["value"] for sequence_series in input_series]
        series_to_count = pd.concat(chunks_list)
    else:
        series_to_count = concatenated_series
    return series_to_count.value_counts(normalize=True, sort=False)


def syllable_series_to_sequence(syllable_series, keep_start_index=True):
    if keep_start_index:
        sequence, lengths = np.vstack(filter_consecutive_duplicates(syllable_series, compute_lengths=True))
        start_index = np.insert(np.cumsum(lengths), 0, 0)
        start_index = syllable_series.index[start_index[:-1]]
        return pd.Series(sequence, index=start_index)
    else:
        return pd.Series(filter_consecutive_duplicates(syllable_series.values, compute_lengths=False))


def get_syllable_durations(*input_series, masks=None):
    if masks is not None:
        input_series = [syllable_series[mask] for syllable_series, mask in zip(input_series, masks)]

    full_length_series = pd.Series(dtype=int)
    for syllable_series in input_series:
        chunks = series_to_numeric_chunks(syllable_series)
        chunks["length"] = chunks["index_end"] - chunks["index_start"] + 1
        full_length_series = full_length_series.append(chunks["length"])
    return full_length_series


def get_group_syllable_transition_series(*input_series, series_groups=None,
                                         relevant_groups=None, **transition_kwargs):
    all_transition_series = {}
    frequencies = {}

    # If no groups are given, assume all recordings are in the same group
    if series_groups is None:
        series_groups = [0] * len(input_series)

    if relevant_groups is None:
        relevant_groups = np.unique(series_groups)

    default_transition_kwargs = dict(normalize="bigram", n=2, sequence_length=1, sequence_overlap=0)
    transition_kwargs = {**default_transition_kwargs, **transition_kwargs, **{"combine": True}}

    # Computing transition matrices for each given group
    for group in relevant_groups:
        # list of syll labels in recordings in the group
        use_labels = [lbl for lbl, grp in zip(input_series, series_groups) if grp == group]

        # Get recordings to include in trans_mat
        # subset only syllable included
        transition_series = get_syllable_transition_series(*use_labels, **transition_kwargs)
        all_transition_series[group] = transition_series

        # Getting frequency information for node scaling
        if transition_kwargs["sequence_length"] > 1:
            use_labels = [get_syllable_transitions(lbl) for lbl in use_labels]
            use_labels = [get_syllable_sequence_series(lbl, transition_kwargs["sequence_length"]) for lbl in use_labels]
            group_frequencies = get_sequence_frequencies(*use_labels)
        else:
            group_frequencies = get_syllable_frequencies(*use_labels)
        frequencies[group] = group_frequencies

    return all_transition_series, frequencies


def h5_results_dict_to_df(h5_results_dict):
    """
    Convert a single results dictionary from a h5 file to a pandas dataframe.

    :param h5_results_dict: A dictionary obtained from reading a moseq results h5 file.
    :type h5_results_dict: dict
    :return: A pandas dataframe containing the results.
    :rtype: pd.DataFrame
    """
    df_dict = {}
    df_dict["centroid"] = pd.DataFrame(h5_results_dict["centroid"], columns=["centroid x", "centroid y"])
    df_dict["heading"] = pd.Series(h5_results_dict["heading"], name="heading")
    df_dict["latent_state"] = pd.DataFrame(h5_results_dict["latent_state"],
                                           columns=[f"latent_state {i}" for i in
                                                    range(h5_results_dict["latent_state"].shape[1])])
    df_dict["syllable"] = pd.Series(h5_results_dict["syllable"], name="syllable")
    return pd.concat(df_dict.values(), axis=1)


def get_results_dict(h5_path_or_dict_or_csv_dir):
    """
    Load a results dictionary from a moseq results h5 file, transform it from an already loaded h5 dictionary, or
    combine multiple csv files into a dictionary of dataframes.
    The returned dictionary of dataframes should have the general structure {recording_name: dataframe}.

    :param h5_path_or_dict_or_csv_dir: The path to the h5 file, a dictionary of h5 results, or a directory containing
    csv files. They all should be generated by the keypoint moseq pipeline.
    :type h5_path_or_dict_or_csv_dir: str or dict
    :return: A dictionary of dataframes containing the results.
    :rtype: dict[str, pd.DataFrame]
    """
    h5_path_check = h5_path_or_dict_or_csv_dir.endswith(".h5")
    dict_check = isinstance(h5_path_or_dict_or_csv_dir, dict)
    if h5_path_check:
        h5_dict = read_data_from_h5_file(h5_path_or_dict_or_csv_dir)
    if dict_check:
        h5_dict = h5_path_or_dict_or_csv_dir
    if h5_path_check or dict_check:
        return {k: h5_results_dict_to_df(v) for k, v in h5_dict.items()}
    else:
        csv_files = [f for f in os.listdir(h5_path_or_dict_or_csv_dir) if f.endswith(".csv")]
        return {f.removesuffix(".csv"): pd.read_csv(os.path.join(h5_path_or_dict_or_csv_dir, f)) for f in csv_files}


def filter_angle(angles, size=9, axis=0, method="median"):
    """
    Directly copied from keypoint_moseq.util.filter_angle.
    Perform median filtering on time-series of angles by transforming to a
    (cos,sin) representation, filtering in R^2, and then transforming back into
    angle space.

    Parameters
    -------
    angles: ndarray
        Array of angles (in radians)

    size: int, default=9
        Size of the filtering kernel

    axis: int, default=0
        Axis along which to filter

    method: str, default='median'
        Method for filtering. Options are 'median' and 'gaussian'

    Returns
    -------
    filtered_angles: ndarray
    """
    from scipy.ndimage import median_filter, gaussian_filter1d
    if method == "median":
        kernel = np.where(np.arange(len(angles.shape)) == axis, size, 1)
        filter = lambda x: median_filter(x, kernel)
    elif method == "gaussian":
        filter = lambda x: gaussian_filter1d(x, size, axis=axis)
    return np.arctan2(filter(np.sin(angles)), filter(np.cos(angles)))


def load_group_dict_from_csv(index_csv_path):
    """
    Load a group dictionary from a csv file. The csv file should have two columns, the first containing the recording
    names and the second containing the group names. The function returns a dictionary mapping recording names to group
    names.

    :param index_csv_path: The path to the csv file.
    :type index_csv_path: str
    :return: A dictionary mapping recording names to group names.
    :rtype: dict[str, str]
    """
    return pd.read_csv(index_csv_path, header=0, index_col=0).squeeze().to_dict()


def load_syllable_name_dict_from_csv(syll_info_csv_path):
    """
    Load a syllable name dictionary from a csv file. The csv file should have two columns, the first containing the
    syllable ids and the second containing the syllable names. The function returns a dictionary mapping syllable ids to
    syllable names.

    :param syll_info_csv_path: The path to the csv file.
    :type syll_info_csv_path: str
    :return: A dictionary mapping syllable ids to syllable names.
    :rtype: dict[int, str]
    """
    return pd.read_csv(syll_info_csv_path, index_col=0, header=0)["label"].to_dict()


def compute_syllable_moseq_df(results_path_or_dict, group_dict=None, fps=30, smooth_centroids=False,
                              smooth_heading=True, drop_latent_states=True):
    """
    Compute a dataframe containing the moseq results with additional columns for velocity, angular velocity, and
    syllable onsets. The dataframe is indexed by the recording name and frame index. The group_dict can be used to
    assign recordings to groups, which can be used for group-wise analysis. The fps parameter is used to calculate
    velocity and angular velocity. If smooth_heading is True, the heading is smoothed using a median filter.

    :param results_path_or_dict: Either a finished moseq results dict or a path. If a path is given, the function will
    load the results dict from the path using get_results_dict.
    :type results_path_or_dict: str or dict
    :param group_dict: A dictionary mapping recording names to group names. If None, all recordings are assigned to the
        "default" group. This can also be a dictionary mapping to iterables of group names, in which case the iterables
        are stored in the group column.
    :type group_dict: dict[str, str] or dict[str, list[str]]
    :param fps: The expected frames per second of the recordings.
    :type fps: int
    :param smooth_centroids: If True, the centroid x and y coordinates are smoothed using a median filter. During the
        regular moseq pipeline, this is not even an option, but during the calculation of the "typical" syllables in the
        modeling pipeline, this is always done, e.g. inside keypoint_moseq.util.get_instance_trajectories.
    :type smooth_centroids: bool
    :param smooth_heading: If True, the heading is smoothed using a median filter.
    :type smooth_heading: bool
    :param drop_latent_states: If True, the latent states are dropped from the dataframe.
    :type drop_latent_states: bool
    :return: A dataframe containing the moseq results with additional columns for velocity, angular velocity, and
        syllable onsets.
    :rtype: pd.DataFrame
    """
    if isinstance(results_path_or_dict, str):
        results_dict = get_results_dict(results_path_or_dict)

    if group_dict is None:
        group_dict = {k: "default" for k in results_dict.keys()}

    moseq_df = pd.concat(results_dict)
    moseq_df = moseq_df.rename({"centroid x": "centroid_x", "centroid y": "centroid_y"}, axis=1)
    if drop_latent_states:
        moseq_df.drop(columns=[col for col in moseq_df.columns if col.startswith("latent_state")], inplace=True)
    moseq_df.index.names = ["track", "frame_index"]

    moseq_df["group"] = moseq_df.index.get_level_values("track").map(group_dict.get)

    moseq_df_dict = {}
    for name, name_df in moseq_df.reset_index(level=["track"]).groupby("track"):
        name_df["velocity_px_s"] = calculate_velocity_2d(name_df["centroid_x"],
                                                         name_df["centroid_y"],
                                                         expected_fps=fps)[0]
        name_df.loc[0, "velocity_px_s"] = 0

        if smooth_centroids:
            name_df[["centroid_x", "centroid_y"]] = median_filter(name_df[["centroid_x", "centroid_y"]],
                                                                  size=(9, 1))

        if smooth_heading:
            name_df["heading"] = filter_angle(name_df["heading"])

        smoothed_heading = pd.Series(filter_angle(name_df["heading"], size=3, method="gaussian"),
                                     index=name_df.index)
        name_df["angular_velocity"] = calculate_angular_speed(smoothed_heading, expected_fps=fps)
        name_df.loc[0, "angular_velocity"] = 0

        name_df["onset"] = name_df["syllable"].diff().ne(0)

        moseq_df_dict[name] = name_df

    moseq_df = pd.concat(moseq_df_dict)
    moseq_df.index.names = ["track", "frame_index"]
    moseq_df.drop("track", axis=1, inplace=True)

    # small addition to original function, makes it easier to calculate syllable trajectories
    moseq_df["syllable_frame"] = consecutive_boolean_counter(~moseq_df["onset"])
    moseq_df["occurrence_id"] = moseq_df.groupby("syllable")["onset"].cumsum() - 1
    return moseq_df


def compute_syllable_stats_df(moseq_df, min_frequency=0.005, groupby=["group", "name"], stats_cols=None, fps=30):
    if stats_cols is None:
        # if no stats columns are specified, use all columns with numerical dtypes, except for syllable
        stats_cols = moseq_df.select_dtypes(include=[np.number]).columns.drop(["syllable"])

    # filter out syllables that are used less than threshold in all recordings
    frequencies = get_syllable_frequencies(moseq_df["syllable"])
    filtered_syllables = frequencies[frequencies.gt(min_frequency)].index
    filtered_df = moseq_df[moseq_df["syllable"].isin(filtered_syllables)].copy()
    print("shit")

    # construct frequency dataframe
    # syllable frequencies within one session add up to 1
    frequency_df_dict = {}
    for name, name_df in moseq_df.reset_index(level=["name"]).groupby("name"):
        syll_freq = get_syllable_frequencies(name_df["syllable"])
        freq_df = syll_freq.to_frame()
        freq_df["group"] = name_df["group"].values[0]
        frequency_df_dict[name] = freq_df
    frequency_df = pd.concat(frequency_df_dict)
    frequency_df.index.names = ["name", "syllable"]
    frequency_df = frequency_df.reset_index(level="name").rename({"proportion": "frequency"}, axis=1)
    if "name" not in groupby:
        frequency_df.drop(columns=["name"], inplace=True)

    features = filtered_df.groupby(groupby + ["syllable"])[stats_cols].describe()
    features.columns = ["_".join(col).strip() for col in features.columns]
    features.reset_index(inplace=True)

    # get durations
    print("shit")
    def _get_syllable_durations(grouped_df):
        syllable_chunks = series_to_numeric_chunks(grouped_df.reset_index()["syllable"])
        syllable_chunks["length"] = syllable_chunks["index_end"] - syllable_chunks["index_start"] + 1
        syllable_chunks.rename({"index_start": "frame_index", "value": "syllable"}, axis=1, inplace=True)
        return syllable_chunks[["frame_index", "syllable", "length"]]

    durations = filtered_df.groupby(groupby).apply(_get_syllable_durations)
    durations = durations.reset_index(level=-1, drop=True).set_index("frame_index", append=True)
    durations["duration"] = durations["length"] / fps
    durations = durations.reset_index().groupby(groupby + ["syllable"])["duration"].describe()
    durations.columns = ["duration_" + col for col in durations.columns]
    durations.rename({"duration_count": "syllable_runs"}, axis=1, inplace=True)

    stats_df = pd.merge(features, frequency_df, on=groupby + ["syllable"])
    stats_df = pd.merge(stats_df, durations, on=groupby + ["syllable"])
    return stats_df


def get_normalized_tracking_df(moseq_df, tracking_df, normalize_to_syllable_onset=True):
    """
    Normalize tracking data to the centroid and heading of the moseq data. The tracking data is first normalized to the
    centroid of the moseq data, then rotated by the heading of the moseq data. If normalize_to_syllable_onset is True,
    the tracking data is normalized to the centroid and heading of the moseq data at the onset of each syllable.
    If normalize_to_syllable_onset is False, the tracking data is normalized to the centroid and heading of the moseq
    data at each frame, leading to a continuous normalization showing only body movement relative to the centroid.

    :param moseq_df: A moseq dataframe containing centroid and heading information, created by
        compute_syllable_moseq_df. See compute_syllable_moseq_df for more information.
    :type moseq_df: pd.DataFrame
    :param tracking_df: A tracking dataframe containing x and y coordinates of keypoints, created by, e.g.,
        get_sleap_tracking_df. See get_sleap_tracking_df for more information.
    :type tracking_df: pd.DataFrame
    :param normalize_to_syllable_onset: If True, normalize the tracking data to the centroid and heading of the moseq
        data at the onset of each syllable. If False, normalize the tracking data to the centroid and heading of the
        moseq data at each frame.
    :type normalize_to_syllable_onset: bool
    :return: A normalized tracking dataframe.
    :rtype: pd.DataFrame
    """
    normalization_df = moseq_df[["centroid_x", "centroid_y", "heading"]].copy()
    if normalize_to_syllable_onset:
        normalization_df[~moseq_df["onset"]] = np.nan
        normalization_df = normalization_df.ffill()

    normalized_tracking_df = tracking_df.copy().drop("instance", level="keypoint", axis=1)
    for kp in tqdm(normalized_tracking_df.columns.get_level_values("keypoint").unique()):
        kp_xy_slice = normalized_tracking_df.loc[:, (kp, ["x", "y"])]
        kp_xy_slice -= normalization_df[["centroid_x", "centroid_y"]].reindex(index=kp_xy_slice.index).values
        normalized_tracking_df.loc[:, (kp, ["x", "y"])] = kp_xy_slice

    x_values = normalized_tracking_df.loc[:, pd.IndexSlice[:, "x"]].values
    y_values = normalized_tracking_df.loc[:, pd.IndexSlice[:, "y"]].values
    original_values = np.dstack([x_values, y_values])

    rotated_points = rotate_2d_point_groups_np(original_values, origins=np.array((0, 0)),
                                               angles=-normalization_df["heading"].values)
    rotated_x = rotated_points[:, :, 0]
    rotated_y = rotated_points[:, :, 1]

    normalized_tracking_df.loc[:, pd.IndexSlice[:, "x"]] = rotated_x
    normalized_tracking_df.loc[:, pd.IndexSlice[:, "y"]] = rotated_y

    return normalized_tracking_df


def get_sampled_trajectory_df(moseq_df, tracking_df, pre=None, post=None, filter_func=None,
                              sampling_value_error_to_warning=True, **sample_kwargs):
    """
    Sample tracking data based on the syllable onsets in the moseq data. The pre and post parameters can be used to
    specify how many frames before and after the onset of a syllable should be included in the sampled trajectory. The
    filter_func parameter can be used to filter the sampled syllables based on custom criteria. The sample_kwargs
    parameter can be used to specify the sampling method, e.g., the number of samples or the fraction of samples to
    include.

    If sampling_value_error_to_warning is True, the function will catch any ValueError that occurs during the sampling
    process and raise a warning instead of an error. This can be useful if is expected that some syllables will not be
    sampleable due to the sampling method and should not lead to a crash. In this case, the function will skip the
    corresponding syllable and continue with the next one.

    Important note: The tracking data must not be normalized to the moseq data before sampling, as this would lead to
    incorrect results due to the "across syllables" access with the pre and post parameters. The syllable chunks would
    be normalized to different centroids and headings, if normalized to the onset of each syllable.

    Important note #2: Both moseq_df and tracking_df must have the same index structure and contain the same datasets.
    I.e., the "frame_index" level must be a full range of frame indices, and the "grouped_index" level must contain the
    same values for the same frame indices. This is enforced by slicing by the intersection of the indices in the
    beginning of the function. Should the frame_index not be a full range, then the finding of syllable "chunks" will
    skip the missing segments of the index, leading to incorrect results.

    :param moseq_df: A moseq dataframe containing centroid and heading information, created by
        compute_syllable_moseq_df.
    :type moseq_df: pd.DataFrame
    :param tracking_df: A tracking dataframe containing x and y coordinates of keypoints, created by, e.g.,
        get_sleap_tracking_df.
    :type tracking_df: pd.DataFrame
    :param pre: A number of frames to include before the onset of a syllable. This overrides the normal start of a
        syllable, and will assign previous frames to the current syllable. Syllable occurrences that start earlier than
        pre frames after the beginning of a track will be excluded.
    :type pre: int or None
    :param post: A number of frames to include after the onset of a syllable. This overrides the normal end of a
        syllable, and will assign following frames to the current syllable. Syllable occurrences that end later than
        post frames before the end of a track will be excluded.
    :type post: int or None
    :param filter_func: A function to filter the sampled syllables. The function should take a dataframe containing the
        syllable chunks created by multi_index_series_to_value_chunks and return a filtered dataframe with the same
        structure.
    :type filter_func: function or None
    :param sampling_value_error_to_warning: If False, raise an Exception if an error occurs during the sampling process.
        If False, catch the Exception and skip the corresponding syllable.
    :type sampling_value_error_to_warning: bool
    :param sample_kwargs: Keyword arguments to pass to the sample method of the syllable chunks. If the "n" argument is
        present, the default "frac" argument will be ignored.
    :type sample_kwargs: Any
    :return: A dataframe containing the sampled trajectory data.
    :rtype: pd.DataFrame
    """
    index_intersection = moseq_df.index.intersection(tracking_df.index)
    moseq_df = moseq_df.loc[index_intersection, :]
    tracking_df = tracking_df.loc[index_intersection, :]

    syllable_chunks = series_to_value_chunks(moseq_df["syllable"], slice_index="frame_index")

    if syllable_chunks["grouped_index"].apply(len).eq(1).all():
        syllable_chunks["grouped_index"] = syllable_chunks["grouped_index"].apply(lambda x: x[0])

    syllable_chunks["occurrence_id"] = 0
    for syllable, single_syllable_chunks in syllable_chunks.groupby("value"):
        syllable_chunks.loc[single_syllable_chunks.index, "occurrence_id"] = range(len(single_syllable_chunks))

    if post is not None:
        grouped_index_names = list(moseq_df.index.names)
        grouped_index_names.remove("frame_index")
        maximum_frames = moseq_df.reset_index().groupby(grouped_index_names)["frame_index"].max()
        syllable_chunks["index_end"] = syllable_chunks["index_start"] + post
        syllable_chunks = syllable_chunks[
            syllable_chunks["index_end"].lt(maximum_frames[syllable_chunks["grouped_index"]].values)]

    if pre is not None:
        syllable_chunks["index_start"] -= pre
        syllable_chunks = syllable_chunks[syllable_chunks["index_start"] >= 0]

    default_sample_kwargs = dict(frac=1.0, random_state=42)
    if "n" in sample_kwargs:
        default_sample_kwargs.pop("frac")
    sample_kwargs = {**default_sample_kwargs, **sample_kwargs}

    syllable_trajectory_df_dict = {}
    for syllable, single_syllable_chunks in tqdm(syllable_chunks.groupby("value")):
        if filter_func is not None:
            single_syllable_chunks = filter_func(single_syllable_chunks)
        if single_syllable_chunks.empty:
            continue

        try:
            sampled_chunks = single_syllable_chunks.sample(**sample_kwargs)
        except ValueError as e:
            if sampling_value_error_to_warning:
                warnings.warn(f"ValueError during sampling leads to skipping of syllable {syllable}: {e}.")
                continue
            else:
                raise e

        if sampled_chunks.empty:
            continue

        sampled_chunks = sampled_chunks.sort_values(["grouped_index", "index_start"])

        sampled_indices = expand_index_from_limits(sampled_chunks.set_index("grouped_index"),
                                                   "index_start", "index_end", maintain_index=True)
        trajectory_df = tracking_df.loc[sampled_indices.index, :].copy()
        trajectory_df["occurrence_id"] = sampled_indices["occurrence_id"]

        if pre is not None or post is not None:
            new_syllable_frames_df = sampled_chunks.copy()
            new_syllable_frames_df["pre"] = -pre if pre is not None else 0
            new_syllable_frames_df["post"] = post if post is not None else new_syllable_frames_df["length"] - 1
            new_syllable_frames_df = expand_index_from_limits(new_syllable_frames_df, "pre", "post",
                                                              index_name="syllable_frame", end_inclusive=True)
            new_syllable_frames = new_syllable_frames_df.reset_index()["syllable_frame"]
            trajectory_df["syllable_frame"] = new_syllable_frames.values
        else:
            trajectory_df["syllable_frame"] = moseq_df.loc[sampled_indices.index, "syllable_frame"]

        syllable_trajectory_df_dict[syllable] = trajectory_df

    syllable_trajectory_df = pd.concat(syllable_trajectory_df_dict,
                                       names=["syllable", "name", "frame_index"]
                                       ).reset_index().drop("frame_index", axis=1, level=0)
    syllable_trajectory_df = syllable_trajectory_df.set_index(
        ["syllable", "name", "occurrence_id", "syllable_frame"]).sort_index()
    return syllable_trajectory_df


def get_normalized_trajectory_df(moseq_df, trajectory_df, normalize_to_syllable_onset=True):
    """
    Normalize tracking data to the centroid and heading of the moseq data. The tracking data is first normalized to the
    centroid of the moseq data, then rotated by the heading of the moseq data. If normalize_to_syllable_onset is True,
    the tracking data is normalized to the centroid and heading of the moseq data at the onset of each syllable.
    If normalize_to_syllable_onset is False, the tracking data is normalized to the centroid and heading of the moseq
    data at each frame, leading to a continuous normalization showing only body movement relative to the centroid.

    :param moseq_df: A moseq dataframe containing centroid and heading information, created by
        compute_syllable_moseq_df. See compute_syllable_moseq_df for more information.
    :type moseq_df: pd.DataFrame
    :param trajectory_df: A tracking dataframe containing x and y coordinates of keypoints, created by, e.g.,
        get_sleap_tracking_df. See get_sleap_tracking_df for more information.
    :type trajectory_df: pd.DataFrame
    :param normalize_to_syllable_onset: If True, normalize the tracking data to the centroid and heading of the moseq
        data at the onset of each syllable. If False, normalize the tracking data to the centroid and heading of the
        moseq data at each frame.
    :type normalize_to_syllable_onset: bool
    :return: A normalized tracking dataframe.
    :rtype: pd.DataFrame
    """
    trajectory_df = trajectory_df.copy()

    if normalize_to_syllable_onset:
        reindexed_moseq_df = moseq_df.reset_index().set_index(["syllable", "track", "occurrence_id", "syllable_frame"])
        norm_df = reindexed_moseq_df.loc[pd.IndexSlice[:, :, :, 0], ["centroid_x", "centroid_y", "heading"]]
        norm_df = norm_df.droplevel(axis=0, level=-1).sort_index()

    else:
        frame_lookup = moseq_df.reset_index().set_index(["syllable", "track", "occurrence_id", "syllable_frame"])
        frame_lookup = frame_lookup.loc[pd.IndexSlice[:, :, :, 0], "frame_index"].droplevel(level=-1, axis=0)
        recreated_frame_index = frame_lookup.reindex(index=trajectory_df.index)
        recreated_frame_index += recreated_frame_index.index.get_level_values("syllable_frame")

        trajectory_df["frame_index"] = recreated_frame_index
        trajectory_df = trajectory_df.reset_index().set_index(
            ["name", "frame_index", "syllable", "occurrence_id", "syllable_frame"])

        norm_df = moseq_df[["centroid_x", "centroid_y", "heading"]].copy()

    norm_centroids = norm_df[["centroid_x", "centroid_y"]].rename({"centroid_x": "x", "centroid_y": "y"}, axis=1)
    norm_headings = norm_df["heading"]

    stacked_df = trajectory_df.stack("keypoint_name", dropna=False)
    stacked_df[["x", "y"]] -= norm_centroids.reindex(index=stacked_df.index)
    normalized_trajectory_df = stacked_df.unstack("keypoint_name").reorder_levels(
        ["keypoint_name", "keypoint_feature"], axis=1)

    x_values = normalized_trajectory_df.loc[:, pd.IndexSlice[:, "x"]].values
    y_values = normalized_trajectory_df.loc[:, pd.IndexSlice[:, "y"]].values
    original_values = np.dstack([x_values, y_values])

    norm_headings = norm_headings.reindex(index=normalized_trajectory_df.index)
    rotated_points = rotate_2d_point_groups_np(original_values, origins=np.array((0, 0)), angles=-norm_headings.values)
    rotated_x = rotated_points[:, :, 0]
    rotated_y = rotated_points[:, :, 1]

    normalized_trajectory_df.loc[:, pd.IndexSlice[:, "x"]] = rotated_x
    normalized_trajectory_df.loc[:, pd.IndexSlice[:, "y"]] = rotated_y

    if not normalize_to_syllable_onset:
        normalized_trajectory_df = normalized_trajectory_df.droplevel(level="frame_index", axis=0)
        normalized_trajectory_df = normalized_trajectory_df.reorder_levels(
            ["syllable", "track", "occurrence_id", "syllable_frame"], axis=0)

    return normalized_trajectory_df


def trajectory_to_median_df(trajectory_df, return_count=False):
    if "score" in trajectory_df.columns.get_level_values("keypoint_feature"):
        trajectory_df = trajectory_df.drop("score", level="keypoint_feature", axis=1)
    median_trajectory_df = trajectory_df.groupby(["syllable", "syllable_frame"]).median().stack("keypoint_name")
    if return_count:
        median_trajectory_df["count"] = trajectory_df.groupby(["syllable", "syllable_frame"]).size()
    return median_trajectory_df


def trajectory_median_full_index_check(median_trajectory_df, keypoint_criterion="any", syllable_frame_criterion="all"):
    """
    Verify that syllables have a certain density in the given median trajectory dataframe. The keypoint_criterion
    specifies whether any or all keypoints should be present in a frame, and the syllable_frame_criterion specifies
    whether any or all frames should meet the keypoint criterion.
    This is useful to ensure that before plotting or distance calculation all syllables have a the same number of frames
    and keypoints.

    :param median_trajectory_df: A median trajectory dataframe containing syllable, syllable_frame, and keypoint
        indices. Can be obtained from trajectory_to_median_df.
    :type median_trajectory_df: pd.DataFrame
    :param keypoint_criterion: One of "any" or "all". Specifies whether any or all keypoints should be present in a
        frame.
    :type keypoint_criterion: str
    :param syllable_frame_criterion: One of "any" or "all". Specifies whether any or all frames should meet the keypoint
        criterion.
    :type syllable_frame_criterion: str
    :return: A boolean series indicating whether each syllable meets the criteria.
    :rtype: pd.Series
    """
    wide_median_trajectories = median_trajectory_df.unstack("keypoint_name")
    # Check if syllables have any/all keypoints present in a particular frame
    per_frame_keypoint_check = wide_median_trajectories.notna().agg(keypoint_criterion, axis=1)
    # Check if all frames are True (based on previous check) for each syllable
    across_frame_check = per_frame_keypoint_check.unstack("syllable_frame", fill_value=False).agg(
        syllable_frame_criterion, axis=1)
    return across_frame_check


def get_trajectory_median_distance_df(median_trajectory_df, **pdist_kwargs):
    full_index_check = len(
        median_trajectory_df.index.to_frame(index=False).value_counts(
            subset=["syllable", "keypoint_name"]).unique()) == 1
    if not full_index_check:
        warnings.warn("The median trajectory dataframe does not have a full index. This may lead to nan values in the "
                      "distance matrix. Consider filtering rare syllables and reducing the number of syllable frames.")

    flattened_median_df = median_trajectory_df.unstack(["syllable_frame", "keypoint_name"])
    flattened_median_df = flattened_median_df.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)

    default_pdist_kwargs = dict(metric="cosine")
    pdist_kwargs = {**default_pdist_kwargs, **pdist_kwargs}
    distance_df = pd.DataFrame(squareform(pdist(flattened_median_df, **pdist_kwargs)),
                               index=flattened_median_df.index, columns=flattened_median_df.index)
    return distance_df


def normalize_median_trajectories_for_plotting(median_trajectory_df, target_size=256):
    normalized_median_trajectory_df_dict = {}
    for syllable, syllable_median_trajectory_df in median_trajectory_df.groupby("syllable"):
        syllable_median_trajectory_df = syllable_median_trajectory_df.droplevel("syllable")
        normalized_median_trajectory_df_dict[syllable] = normalize_keypoint_df_for_plotting(
            syllable_median_trajectory_df, target_size=target_size)
    normalized_median_trajectory_df = pd.concat(normalized_median_trajectory_df_dict,
                                                names=["syllable", "syllable_frame", "keypoint"])
    return normalized_median_trajectory_df
