import numpy as np
import pandas as pd
import warnings
from ..pandas_tools.rle import rle_series, expand_rle_df_to_series
from tqdm.auto import tqdm
from joblib import Parallel, delayed


def behavior_series_to_df(input_behavior_series):
    behavior_df = pd.DataFrame(
        {behavior_name: input_behavior_series.eq(behavior_name) for behavior_name in input_behavior_series.unique()})
    behavior_df.columns.name = "behavior_name"
    return behavior_df


def behavior_df_to_series(input_behavior_df, multi_behavior_handling="raise"):
    row_sums = input_behavior_df.astype(int).sum(axis=1)

    if multi_behavior_handling == "raise" and row_sums.gt(1).any():
        raise ValueError("Input behavior dataframe has multiple behaviors per row")

    if multi_behavior_handling == "drop" and row_sums.gt(1).any():
        input_behavior_df = input_behavior_df[row_sums.le(1)]

    behavior_series = input_behavior_df.idxmax(axis=1).rename("behavior_name")

    if multi_behavior_handling == "include":
        multi_rows = input_behavior_df[row_sums.gt(1)]
        combined_behavior_series = multi_rows.apply(lambda x: x.index[x].tolist(), axis=1)
        behavior_series.loc[combined_behavior_series.index] = combined_behavior_series

    behavior_series[row_sums.eq(0)] = pd.NA

    return behavior_series


def behavior_to_bout_df(input_behavior_df, drop_nan_run_values=True):
    try:
        behavior_series = behavior_df_to_series(input_behavior_df, multi_behavior_handling="raise")
        bout_df = rle_series(behavior_series)
        if drop_nan_run_values:
            bout_df = bout_df.dropna(subset="run_value")
        bout_df = bout_df.astype(input_behavior_df.columns.dtype)
    except ValueError:
        warnings.warn("Input behavior dataframe has multiple behaviors per frame. Running parallel extraction of bouts")

        def rle_behavior_series(input_behavior_series):
            rle_df = rle_series(input_behavior_series, drop_run_values=(False,))
            rle_df["run_value"] = input_behavior_series.name
            return rle_df

        bout_df_list = Parallel(n_jobs=-1)(
            delayed(rle_behavior_series)(input_behavior_df[behavior_name]) for behavior_name in
            tqdm(input_behavior_df.columns))
        bout_df = pd.concat(bout_df_list, axis=0)

    bout_df = bout_df.rename(dict(end_index="end_frame", run_length="bout_length", run_value="behavior_name"), axis=1)
    bout_df.index.names = list(bout_df.index.names[:-1]) + ["frame_index"]
    return bout_df.sort_index().rename_axis("bout_feature", axis=1)


def get_full_frame_index(input_index):
    if isinstance(input_index, pd.MultiIndex):
        input_index_df = input_index.to_frame()
        group_levels = input_index_df.columns[:-1]
        group_levels = group_levels[0] if len(group_levels) == 1 else group_levels

        full_index_df_list = []
        for group, grouped_index_df in input_index_df.groupby(level=group_levels):
            min_frame, max_frame = grouped_index_df["frame_index"].min(), grouped_index_df["frame_index"].max()
            if not isinstance(group, tuple):
                group = (group,)
            full_index_df_list.append(
                pd.MultiIndex.from_product(iterables=[group, list(range(min_frame, max_frame + 1))],
                                           names=input_index_df.columns).to_frame())
        return pd.MultiIndex.from_frame(pd.concat(full_index_df_list))
    else:
        return pd.Index(range(input_index.min(), input_index.max() + 1), name="frame_index")


def bout_to_behavior_df(input_bout_df, reindex_to_full_index=True):
    input_bout_df = input_bout_df.rename(
        dict(end_frame="end_index", bout_length="run_length", behavior_name="run_value"),
        axis=1)
    input_bout_df.index.names = list(input_bout_df.index.names[:-1]) + ["start_index"]

    behavior_series = expand_rle_df_to_series(input_bout_df, index_name="frame_index", series_name="behavior_name")
    behavior_series = behavior_series.sort_index()
    unique_index = behavior_series.index.drop_duplicates()
    unique_behavior_names = pd.Index(behavior_series.unique(), name="behavior_name")

    if reindex_to_full_index:
        unique_index = get_full_frame_index(unique_index)

    behavior_df = pd.DataFrame(False, index=unique_index, columns=unique_behavior_names)

    for behavior_name in tqdm(unique_behavior_names):
        behavior_name_series = behavior_series.eq(behavior_name)
        behavior_name_series = behavior_name_series[behavior_name_series]
        behavior_df[behavior_name] = behavior_name_series.reindex(unique_index, fill_value=False)

    return behavior_df


def bout_df_to_count_series(input_bout_df, combine_multi=False, ignore_lengths=True):
    def _sum_func(input_grouped_bout_df):
        if ignore_lengths:
            return len(input_grouped_bout_df)
        else:
            return input_grouped_bout_df["bout_length"].sum()

    if isinstance(input_bout_df.index, pd.MultiIndex) and not combine_multi:
        multi_levels = input_bout_df.index.names[:-1]
        group_levels = multi_levels[0] if len(multi_levels) == 1 else multi_levels

        proportion_series_dict = {}
        for group, grouped_bout_df in tqdm(input_bout_df.groupby(level=group_levels)):
            proportion_series = grouped_bout_df.groupby("behavior_name").apply(_sum_func, include_groups=False)
            proportion_series_dict[group] = proportion_series.sort_values(ascending=False)
        proportion_series = pd.concat(proportion_series_dict, names=list(multi_levels) + ["behavior_name"], axis=0)
        return proportion_series.rename("onset_proportion" if ignore_lengths else "frame_proportion")
    else:
        proportion_series = input_bout_df.groupby("behavior_name").apply(_sum_func, include_groups=False)
        return proportion_series.sort_values(ascending=False).rename(
            "onset_proportion" if ignore_lengths else "frame_proportion")


def count_to_proportion_series(count_series, combine_multi=False):
    if isinstance(count_series.index, pd.MultiIndex) and not combine_multi:
        multi_levels = count_series.index.names[:-1]
        group_levels = multi_levels[0] if len(multi_levels) == 1 else multi_levels
        group_sums = count_series.groupby(group_levels).transform("sum")
        return count_series / group_sums
    else:
        combined_count_series = count_series.groupby("behavior_name").sum()
        return combined_count_series / combined_count_series.sum()

def bout_df_to_proportion_series(input_bout_df, combine_multi=False, ignore_lengths=True):
    count_series = bout_df_to_count_series(input_bout_df, combine_multi=combine_multi, ignore_lengths=ignore_lengths)
    return count_to_proportion_series(count_series, combine_multi=combine_multi)


def single_bout_df_to_transition_series(input_bout_df, show_tqdm=True):
    unique_behaviors = input_bout_df["behavior_name"].unique()
    transition_series = pd.Series(0, index=pd.MultiIndex.from_product([unique_behaviors, unique_behaviors],
                                                                      names=["previous_behavior", "next_behavior"]))

    iterator = tqdm(input_bout_df.groupby("behavior_name")) if show_tqdm else input_bout_df.groupby("behavior_name")
    for behavior_name, behavior_bout_df in iterator:
        index_df = behavior_bout_df.index.to_frame()
        next_index = index_df.copy()
        next_index["frame_index"] = behavior_bout_df["end_frame"] + 1
        if len(next_index.columns) == 1:
            next_index = pd.Index(next_index["frame_index"], name="frame_index")
        else:
            next_index = pd.MultiIndex.from_frame(next_index)

        valid_indices = next_index.intersection(input_bout_df.index)
        next_bouts = input_bout_df.loc[valid_indices, :]
        next_bouts_count = next_bouts["behavior_name"].value_counts(normalize=False)

        transition_series.loc[behavior_name, next_bouts_count.index] = next_bouts_count.values

    return transition_series.rename("transition_count")


def bout_df_to_transition_series(input_bout_df, combine_multi=False):
    if isinstance(input_bout_df.index, pd.MultiIndex) and not combine_multi:
        multi_levels = input_bout_df.index.names[:-1]
        group_levels = multi_levels[0] if len(multi_levels) == 1 else multi_levels

        transition_series_dict = {}
        for group, grouped_bout_df in tqdm(input_bout_df.groupby(level=group_levels)):
            transition_series_dict[group] = single_bout_df_to_transition_series(grouped_bout_df, show_tqdm=False)
        return pd.concat(transition_series_dict, names=list(multi_levels) + ["previous_behavior", "next_behavior"])
    else:
        return single_bout_df_to_transition_series(input_bout_df)


def bout_to_vector_df(input_bout_df, ignore_lengths=True):
    """
    Converts a bout dataframe to a vector dataframe. Each row corresponds to a single sample in the bout_df and consists
    of the transition probabilities between behavior bouts and the proportions of bouts (either length-based or count-
    based, depending on ignore_lengths). The vector dataframe has the same index as the bout dataframe.

    :param input_bout_df: A bout DataFrame.
    :type input_bout_df: pd.DataFrame
    :param ignore_lengths: Whether to just count the number of bouts for each behavior per sample instead of adding up
    the frame lengths of all bouts for a single behavior.
    :type ignore_lengths: bool
    :return: A vector DataFrame with the same index as the input bout dataframe and columns consisting of transitions
    between pairs of behaviors and behavior counts.
    :rtype: pd.DataFrame
    """
    transition_series = bout_df_to_transition_series(input_bout_df)

    transition_vector_df = transition_series.unstack(["previous_behavior", "next_behavior"], fill_value=0)
    transition_vector_df.columns = transition_vector_df.columns.to_flat_index()

    counts_series = bout_df_to_count_series(input_bout_df, ignore_lengths=ignore_lengths)
    proportion_vector_df = counts_series.unstack("behavior_name", fill_value=0)

    return transition_vector_df.join(proportion_vector_df).sort_index()


def post_process_vector_df(vector_df, normalize_transitions=True):
    transition_cols = [col for col in vector_df.columns if isinstance(col, tuple)]
    occurrence_df = vector_df.drop(transition_cols, axis=1)

    expected_transitions = pd.MultiIndex.from_product([occurrence_df.columns, occurrence_df.columns]).to_flat_index()
    transition_df = vector_df[transition_cols]
    transition_df = transition_df.reindex(expected_transitions, axis=1, fill_value=0)

    if normalize_transitions:
        transition_df = transition_df.div(transition_df.sum(axis=1), axis=0)

    return pd.concat([transition_df, occurrence_df], axis=1)