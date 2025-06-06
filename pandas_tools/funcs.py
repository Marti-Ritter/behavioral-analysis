# built-ins
import os
from functools import reduce
from itertools import combinations
from random import sample, seed

# additional packages
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def pd_translate_by_dict(input_pd, translation_dict):
    """Translate the values in a pandas object (Series or DataFrame) according to the mapping in the translation_dict

    :param input_pd: Pandas object that will be translated
    :type input_pd: pd.Series or pd.DataFrame
    :param translation_dict: Dict containing a mapping of the values in the pandas object to target values
    :type translation_dict: dict
    :return: a Series containing the translated values
    :rtype: pd.Series
    """
    if isinstance(input_pd, pd.Series):
        return input_pd.apply(lambda x: translation_dict[x]
        if x in translation_dict
        else x)
    elif isinstance(input_pd, pd.DataFrame):
        return input_pd.apply(lambda x: translation_dict[tuple(x.values)]
        if tuple(x.values) in translation_dict
        else x.to_dict(), axis=1)
    else:
        raise ValueError("Unknown input of type {}!".format(type(input_pd)))


def pd_transitions_on_column(df, column):
    """Get the locations at which the numerical value in a column increases

    :param df: dataframe containing the column
    :type df: pandas.DataFrame
    :param column: column in the given dataframe
    :type column: str
    :return: Rows of the given dataframe at which the numerical value of the column increases
    :rtype: pandas.DataFrame
    """
    return df[(df[column].diff() > 0) | (df[column].diff() > 0).shift(-1).fillna(False)]


def pd_split_subsets(input_pd, subsets, random_state=0):
    """
    Given a pandas dataframe or series, split it into subsets subsets of equal size (or as close as possible).

    :param input_pd: The input pandas dataframe or series
    :type input_pd: pd.DataFrame or pd.Series
    :param subsets: The number of subsets to split the input into
    :type subsets: int
    :param random_state: The random state to use for the random sampling
    :type random_state: int
    :return: A list of the subsets
    :rtype: list of (pd.DataFrame or pd.Series)
    """
    remaining_pd = input_pd.copy()
    output_subsets = []
    for i in range(subsets):
        subset = remaining_pd.sample(frac=1 / (subsets - len(output_subsets)), random_state=random_state)
        output_subsets.append(subset)
        remaining_pd = remaining_pd.drop(subset.index)
    return output_subsets


def pd_split_share(input_pd, n_samples, share_fraction=0., share_count=2, random_state=0):
    """
    Let n=n_samples, p=share_fraction, k=share_count, s=input_length
    Total output:
    Unique fraction: 1-p
    Fraction with k copies: p

    Single output set:
    Length: s * (1-p + p*k) / n
    Fraction of data unique to this set: (1-p) / (1-p + p*k)
    Fraction of data shared with any other set: p*k / (1-p + p*k)

    Inter-set relationship:
    Fraction of data in one set shared with one other set: p*k / (1-p + p*k) * ((k-1) / (n-1))

    This function will split an input series or dataframe into n_samples subsets. In the total output data, there will
    be 1-share_fraction percent unique entries, and share_fraction percent of data multiplied share_count times and
    distributed evenly among the output subsets.

    The above calculations show what proportion of the data in each level will be unique, multiplied or shared.

    The point of this function is to have a clean and intuitive way to split data to give to n_samples
    experimenters for scoring and be able to cross-validate the output among the scorers by handing out
    share_fraction percent of the data share_count times (to multiple scorers) instead of a single time.

    :param input_pd:
    :type input_pd:
    :param n_samples:
    :type n_samples:
    :param share_fraction:
    :type share_fraction:
    :param share_count:
    :type share_count:
    :param random_state:
    :type random_state:
    :return:
    :rtype:
    """

    assert 0 <= share_fraction <= 1, "Fraction of shared elements must be between 0 and 1, 0 <= {} <= 1".format(
        share_fraction)
    assert 0 <= share_count <= n_samples, "Shared elements must be between 0 and n_samples, 0 <= {} <= {}".format(
        share_count, n_samples)

    shared = input_pd.sample(frac=share_fraction, random_state=random_state)
    unique = input_pd.drop(shared.index)

    output_samples = pd_split_subsets(unique, n_samples, random_state=random_state)

    comb_list = list(combinations(range(n_samples), r=share_count))
    shared_samples = pd_split_subsets(shared, len(comb_list), random_state=random_state)

    for comb, comb_share in zip(comb_list, shared_samples):
        for comb_element in comb:
            output_samples[comb_element] = pd.concat([output_samples[comb_element], comb_share])

    return output_samples


def pd_split_df(input_df, n_samples, group_cols=None, share_fraction=0., share_count=2, random_state=0):
    """
    Split a df with pd_split_share but keep the proportion of data in group_cols even for all created output sets.

    :param input_df: The input dataframe
    :type input_df: pd.DataFrame
    :param n_samples: The number of output samples to create
    :type n_samples: int
    :param group_cols: The columns to keep the proportion of data in even for all created output sets
    :type group_cols: list or tuple
    :param share_fraction: The fraction of data to share among the output sets
    :type share_fraction: float
    :param share_count: The number of times to share the data among the output sets
    :type share_count: int
    :param random_state: The random state to use for the random sampling
    :type random_state: int
    :return: A list of the output dataframes
    :rtype: list of pd.DataFrame
    """
    output_sets_grouped = []
    seed(random_state)

    if group_cols is None:
        output_sets_grouped = [pd_split_share(input_df, n_samples, share_fraction=share_fraction,
                                              share_count=share_count,
                                              random_state=random_state)]
    else:
        for group, group_df in input_df.groupby(group_cols):
            output_sets_grouped.append(
                sample(pd_split_share(group_df, n_samples, share_fraction=share_fraction, share_count=share_count,
                                      random_state=random_state), k=n_samples))
    return [pd.concat(output_grouped) for output_grouped in zip(*output_sets_grouped)]


def pd_verify_split(input_split):
    """
    An utility function to verify the quality of a split of a dataset created with pd_split_df.

    :param input_split: A list of pandas dataframes or series
    :type input_split: list of pd.DataFrame
    :return: None
    """
    print("Frequency of data in dataset:")
    lines = pd.concat(input_split).value_counts().value_counts(normalize=True).to_string()
    print("\n".join(["\t" + line for line in lines.splitlines()]))
    print()

    print("Intra-sample frequency")
    for i in range(len(input_split)):
        print("\tSample {}".format(i))
        lines = (pd.concat(input_split).loc[input_split[i].index].value_counts()).value_counts(
            normalize=True).to_string()
        print("\n".join(["\t\t" + line for line in lines.splitlines()]))

    print("Inter-sample overlap")
    for i in range(len(input_split)):
        for j in range(len(input_split)):
            if i >= j:
                continue
            print("\t{} shared with {}".format(i, j))
            lines = (pd.concat([input_split[i], input_split[j]]).loc[input_split[i].index].value_counts() > 1).astype(
                int).value_counts(normalize=True).to_string()
            print("\n".join(["\t\t" + line for line in lines.splitlines()]))
            print()
    print()


def pd_prepend_index_levels(input_pd, level_dict):
    """
    Prepend index levels to a pandas object. Useful for creating a multi-index dataframe from a single-index dataframe
    or multi-index series from a single-index series. The level_dict is a dictionary mapping the level names to the
    values to prepend.

    :param input_pd: The input dataframe or series
    :type input_pd: pd.DataFrame or pd.Series
    :param level_dict: A dictionary mapping the level names to the values to prepend
    :type level_dict: dict
    :return: The input dataframe with the prepended index levels
    :rtype: pd.DataFrame
    """
    return pd.concat({tuple(val for val in level_dict.values()): input_pd}, names=level_dict.keys())


def pd_prepend_column_levels_to_df(input_df, level_dict):
    """
    Prepend column levels to a dataframe. See pd_prepend_index_levels_to_df for more information. This does not work for
    series, as they do not have columns, just hashable names.
    """
    return pd_prepend_index_levels(input_df.T, level_dict).T


def pd_group_to_string_dict(input_df, format_string_pattern, group_list):
    return {values: format_string_pattern.format(*values) for values, _df in input_df.groupby(group_list)}


def pd_expand_dict_series(input_series):
    return pd.DataFrame(input_series.to_list())


def pd_remove_value_rows(input_df, column_value_dict):
    output_df = input_df.copy()
    for column, value in column_value_dict.items():
        output_df = output_df[output_df[column] != value]
    return output_df


def pd_missing_ints_in_list(input_list):
    input_series = pd.Series(input_list, index=input_list)
    is_int_missing = input_series.reindex(range(input_series.min(), input_series.max())).isna()
    return is_int_missing[is_int_missing].index.to_list()


def pd_count_duplications(input_df, cols_to_check):
    return input_df.groupby(list(cols_to_check), as_index=False).size().rename({"size": "duplications"}, axis=1)


def pd_share_elements(input_pd_list, share_fraction=None, avoid_inter_pair_overlap=True, random_state=0):
    if share_fraction is None:
        share_fraction = 1 / (len(input_pd_list) - 1)
    if avoid_inter_pair_overlap:
        assert (share_fraction * (len(input_pd_list) - 1)) <= 1, "Cannot share more than a 100% of individual datasets!"
    remaining_pd_list = [input_pd.copy() for input_pd in input_pd_list]
    shared_elements = {}
    for i in range(len(remaining_pd_list)):
        for j in range(len(remaining_pd_list)):
            if i == j:
                continue
            shared_n = min(round(len(input_pd_list[i]) * share_fraction), len(remaining_pd_list[i]))
            shared = remaining_pd_list[i].sample(n=shared_n, random_state=random_state)
            shared_elements[(i, j)] = shared
            if avoid_inter_pair_overlap:
                remaining_pd_list[i].drop(shared.index, inplace=True)
    output_pd_list = [remaining_pd.copy() for remaining_pd in remaining_pd_list]
    for i in range(len(remaining_pd_list)):
        for j in range(len(remaining_pd_list)):
            if i == j:
                continue
            output_pd_list[i] = pd.concat(
                [output_pd_list[i], shared_elements[(i, j)], shared_elements[(j, i)]]).drop_duplicates()
    return output_pd_list


def pd_limited_interpolation_series(input_series, hard_limit=None, method="linear"):
    """

    :param input_series:
    :type input_series: pd.Series
    :param hard_limit:
    :type hard_limit:
    :param method:
    :type method: str or Literal
    :return:
    :rtype: pd.Series
    """
    mask = input_series.copy()
    grp = (mask.notnull() != mask.shift().notnull()).cumsum()
    mask = (grp.groupby(grp).transform('size') <= hard_limit if hard_limit is not None else len(
        input_series)) | input_series.notnull()
    output_series = input_series.interpolate(method=method, limit=hard_limit, limit_area="inside")
    output_series[~mask] = np.nan
    return output_series


# https://stackoverflow.com/a/30538371
def pd_limited_interpolation_df(input_df, hard_limit=None, method="linear"):
    return input_df.apply(lambda x: pd_limited_interpolation_series(x, hard_limit=hard_limit, method=method), axis=0)


def split_pd_with_boolean_series(input_pd, boolean_series, return_index=False, bool_indicates_start=True):
    """

    :param input_pd:
    :type input_pd:
    :param boolean_series:
    :type boolean_series:
    :param return_index:
    :type return_index:
    :param bool_indicates_start:
    :type bool_indicates_start:
    :return:
    :rtype: list of ((pd.Series, pd.DataFrame or pd.Series) or pd.DataFrame)
    """
    split_groups = boolean_series.cumsum().set_axis(input_pd.index)
    if not bool_indicates_start:
        split_groups = split_groups.shift(1, fill_value=split_groups.iloc[0] - boolean_series.iloc[0])
    return [(split_index, sub_pd) if return_index else sub_pd for split_index, sub_pd in input_pd.groupby(split_groups)]


def pd_series_stat_in_limits(input_series, limit_list, func=pd.Series.sum):
    return [func(input_series[start:stop]) for start, stop in limit_list]


def pd_categorical_histogram(input_pd, density=False, cumulative=False,
                             sort_alphabetically=False, width=1, **kwargs):
    histogram_data = input_pd.value_counts(normalize=density)
    histogram_data = histogram_data.cumsum() if cumulative else histogram_data
    histogram_data = histogram_data.sort_index() if sort_alphabetically else histogram_data
    return histogram_data.plot(kind='bar', width=width, **kwargs)


def pd_filter_df(input_df, filter_dict):
    filter_list = []
    for filter_key, filter_states in filter_dict.items():
        if isinstance(filter_states, list) or isinstance(filter_states, tuple):
            filter_list.append(input_df[filter_key].isin([str(filter_state) for filter_state in filter_states]))
        else:
            filter_list.append(input_df[filter_key].eq(str(filter_states)))
    return input_df[np.bitwise_and.reduce(filter_list)]


def pd_multi_filter_df(input_df, filter_dict_list):
    output_df = pd.DataFrame()
    if not isinstance(filter_dict_list, list):
        filter_dict_list = [filter_dict_list]
    for filter_dict in filter_dict_list:
        output_df = pd.concat([output_df, pd_filter_df(input_df, filter_dict)])
    return output_df


def add_light_cycle_spans(ax, light_start_hour=6, dark_start_hour=18, light_kws=None, dark_kws=None, legend=True):
    light_kws = dict(dict(color="palegreen", alpha=0.5), **(light_kws if light_kws is not None else {}))
    dark_kws = dict(dict(color="cyan", alpha=0.5), **(dark_kws if dark_kws is not None else {}))

    date_limits = [pd.to_datetime(_, unit="D") for _ in ax.get_xlim()]
    date_span = pd.date_range(start=date_limits[0].normalize() - pd.to_timedelta(1, unit="days"),
                              end=date_limits[1].normalize() + pd.to_timedelta(1, unit="days"),
                              freq="D")
    light_starts = date_span + pd.to_timedelta(light_start_hour, unit="hours")
    dark_starts = date_span + pd.to_timedelta(dark_start_hour, unit="hours")

    for i, span in enumerate(zip(light_starts, dark_starts)):
        ax.axvspan(*span, label="_" * i + "light" if legend else None, **light_kws)
    for i, span in enumerate(zip(dark_starts, light_starts[1:])):
        ax.axvspan(*span, label="_" * i + "dark" if legend else None, **dark_kws)
    ax.set_xlim(*date_limits)


def pd_flatten_tuple_column_names(input_df):
    output_df = input_df.copy()
    output_df.columns = ["/".join(col) if isinstance(col, tuple) else col for col in output_df.columns]
    return output_df


def pd_sort_columns(input_df, type_order=(str, int, tuple)):
    sorted_columns = []
    for type_ in type_order:
        type_cols = [col for col in input_df.columns if isinstance(col, type_)]
        ordered_cols = sorted(type_cols)
        sorted_columns.extend(ordered_cols)
    sorted_columns.extend([col for col in input_df.columns if col not in sorted_columns])
    return input_df.reindex(sorted_columns, axis=1)


def pd_rolling_df_factory(input_df, apply_func):
    def pd_rolling_df_apply(rolling_series):
        return apply_func(input_df.loc[rolling_series.index])

    return pd_rolling_df_apply


def get_max_frequency(input_data, sampling_rate=30.00003):
    n = len(input_data)
    k = np.arange(n)

    frq = k / (n / sampling_rate)  # two sides frequency range
    frq = frq[:len(frq) // 2]  # one side frequency range, can only detect up to half sampling rate

    fft_frequencies = np.fft.fft(input_data) / n  # dft and normalization
    fft_frequencies = fft_frequencies[:n // 2]

    return frq[np.argmax(fft_frequencies)]


def pd_smooth_series(input_series, window_size=3, agg="mean", center=True, **window_kwargs):
    """
    uses pandas.Series.rolling under the hood. w
    :param input_series:
    :type input_series:
    :param window_size:
    :type window_size:
    :param agg:
    :type agg:
    :param center:
    :type center:
    :param window_kwargs: kwargs for pd.Series.rolling
    :return:
    :rtype:
    """
    return input_series.rolling(window_size, center=center, **window_kwargs).agg(agg)


def pd_smooth_df(input_df, param_dict=None, **params):
    output_df = input_df.copy()
    for col in input_df:
        smoothing_params = param_dict[col] if isinstance(param_dict, dict) and col in param_dict else params
        output_df[col] = pd_smooth_series(input_df[col], **smoothing_params)
    return output_df


def iter_series_to_df(pd_series, columns=None):
    """
    Converts a pandas series of iterables to a dataframe. Useful for converting a series of lists to a dataframe.

    :param pd_series: An input pd.Series of iterables
    :type pd_series: pd.Series
    :param columns: Optional column names for the output dataframe
    :type columns: list or tuple
    :return: A dataframe with the iterables expanded to columns
    :rtype: pd.DataFrame
    """
    df = pd_series.apply(pd.Series)
    if columns is not None:
        df.columns = columns
    return df


def series_to_dataframe(series, start, end, end_inclusive=True):
    """
    Converts a pandas series to a dataframe with a RangeIndex from start to end as the index and the series index as the
    columns and matching dtypes.

    :param series: A pandas series
    :type series: pd.Series
    :param start: An integer start index
    :type start: int
    :param end: An integer end index
    :type end: int
    :return: A dataframe with a RangeIndex from start to end as the index and the series index as the columns
    :rtype: pd.DataFrame
    """

    if end_inclusive:
        end += 1

    df = pd.DataFrame(series.values.reshape(1, -1).repeat(end - start, axis=0))
    df.index = pd.RangeIndex(start, end)
    df.columns = series.index

    return df


def expand_index_from_limits(input_df, start_col, end_col, index_name="index", end_inclusive=True,
                             maintain_index=False):
    """
    Expands a dataframe with start and end columns encoding an index into a dataframe with an additional index column
    based on start and end and all values in that span identical to the values in the original dataframe's row with the
    start and end values. This additional index can either be the new index of the dataframe or a new index level in a
    multi-index dataframe.

    :param input_df: A dataframe with start and end columns encoding an index
    :type input_df: pd.DataFrame
    :param start_col: A column in input_df with the start of the index
    :type start_col: str
    :param end_col: A column in input_df with the end of the index
    :type end_col: str
    :param index_name: The name of the index column in the output dataframe
    :type index_name: str
    :param end_inclusive: Whether the end of the index is inclusive
    :type end_inclusive: bool
    :param maintain_index: Whether to maintain the original index of the dataframe. If False, the original index is
        reset and replaced with the new index. If True, the new index is added as a new level to the original index.
    :return: A dataframe with a single index column based on start and end and all values in that span identical to the
    values in the original dataframe's row with the start and end values.
    :rtype: pd.DataFrame
    """
    expanded_df = input_df.reset_index(drop=True)
    expanded_df = expanded_df.loc[
        expanded_df.index.repeat(input_df[end_col] - input_df[start_col] + int(end_inclusive))]
    _input_df = input_df[[start_col, end_col]].astype(int)  # slice and cast to int for range
    expanded_df[index_name] = np.hstack(_input_df.apply(lambda x: range(x[start_col], x[end_col] + int(end_inclusive)),
                                                        axis=1))
    expanded_df[index_name] = expanded_df[index_name].astype(int)

    if maintain_index:
        original_index = input_df.index.to_frame().reset_index(drop=True)
        expanded_original_index = original_index.loc[
            original_index.index.repeat(input_df[end_col] - input_df[start_col] + int(end_inclusive))]
        expanded_df.index = pd.MultiIndex.from_frame(expanded_original_index)

    return expanded_df.set_index(index_name, append=maintain_index)


def expand_index_from_limits_old(input_df, start_col, end_col, index_name="index"):
    """
    Expands a dataframe with start and end columns encoding an index into a dataframe with a single index column based
    on start and end and all values in that span identical to the values in the original dataframe's row with the
    start and end values.

    :param input_df: A dataframe with start and end columns encoding an index
    :type input_df: pd.DataFrame
    :param start_col: A column in input_df with the start of the index
    :type start_col: str
    :param end_col: A column in input_df with the end of the index
    :type end_col: str
    :param index_name: The name of the index column in the output dataframe
    :type index_name: str
    :return: A dataframe with a single index column based on start and end and all values in that span identical to the
    values in the original dataframe's row with the start and end values.
    :rtype: pd.DataFrame
    """
    expanded_row_dfs = []
    for _, row in input_df.iterrows():
        expanded_row_dfs.append(series_to_dataframe(row, row[start_col], row[end_col]))
    return pd.concat(expanded_row_dfs, axis=0).astype(input_df.dtypes).rename_axis(index_name)


def slice_df_by_tuples_and_scalars(df, tuple_slice_indices=None, scalar_slice_indices=None, axis=0):
    """
    Slices a dataframe by a list of tuples and a list of scalars along the given axis. It is ensured that the tuples and
    scalars are in a list if the given objects are not lists themselves. This is useful for slicing a dataframe by
    tuples and scalars without encountering any issues in numpy due to the "ragged" shape of such a slice.

    This function is necessary because, as of 05.05.2023, slicing with a mixed list of tuples and scalars is causing a
    VisibleDeprecationWarning in numpy. This function is a workaround for that issue.

    The VisibleDeprecationWarning is shown as this:

    ``VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of
    lists-or-tuples-or-ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must
    specify 'dtype=object' when creating the ndarray.``

    ``result = np.asarray(values, dtype=dtype)``

    :param df: The dataframe to slice
    :type df: pd.DataFrame
    :param tuple_slice_indices: A list of tuples to slice by
    :type tuple_slice_indices: list or tuple
    :param scalar_slice_indices: A list of scalars to slice by
    :type scalar_slice_indices: list or Any
    :param axis: The axis to slice along, 0 for rows, 1 for columns
    :type axis: int
    :return: A dataframe sliced by the given tuples and scalars along the given axis
    :rtype: pd.DataFrame
    """

    tuple_slice_indices = tuple_slice_indices if tuple_slice_indices is not None else []
    scalar_slice_indices = scalar_slice_indices if scalar_slice_indices is not None else []

    tuple_list = [tuple_slice_indices] if not isinstance(tuple_slice_indices, list) else tuple_slice_indices
    scalar_list = [scalar_slice_indices] if not isinstance(scalar_slice_indices, list) else scalar_slice_indices

    if axis == 0:
        transposed_df = df.T
        tuple_slice = transposed_df[tuple_list]
        scalar_slice = transposed_df[scalar_list]
        return tuple_slice.join(scalar_slice).T

    elif axis == 1:
        tuple_slice = df[tuple_list]
        scalar_slice = df[scalar_list]
        return tuple_slice.join(scalar_slice)

    else:
        raise ValueError("axis must be 0 or 1")


def insert_level_into_index(input_pd, new_level_values,
                            new_level_index=0, new_level_name="new_level", inplace=False):
    """
    Inserts a new level into a pandas dataframe's MultiIndex at the given level with the given values and name.
    Taken from https://stackoverflow.com/a/56278736

    :param input_pd: A pandas dataframe or series with a MultiIndex on which to insert a new level
    :type input_pd: pd.DataFrame or pd.Series
    :param new_level_values: The values to insert into the new level
    :type new_level_values: list or tuple or pd.Series
    :param new_level_index: The index of the new level to insert
    :type new_level_index: int
    :param new_level_name: The name of the new level to insert
    :type new_level_name: str
    :param inplace: Whether to perform the operation in place
    :type inplace: bool
    :return: The input dataframe with the new level inserted into its MultiIndex if inplace is True, otherwise None
    :rtype: pd.DataFrame or pd.Series or None
    """

    if not inplace:
        input_pd = input_pd.copy()

    # Convert index to dataframe
    old_idx = input_pd.index.to_frame()

    # Insert new level at specified location
    old_idx.insert(new_level_index, new_level_name, new_level_values)

    # Convert back to MultiIndex
    input_pd.index = pd.MultiIndex.from_frame(old_idx)

    if not inplace:
        return input_pd


def pd_get_index_union(*pd_indices):
    """
    Gets the union of any passed pandas indices.
    :param pd_indices: Pandas indices
    :type pd_indices: pd.Index
    :return: The union of the given indices
    :rtype: pd.Index
    """

    all_multi_index = all([isinstance(pd_index, pd.MultiIndex) for pd_index in pd_indices])
    index_union_func = pd.MultiIndex.union if all_multi_index else pd.Index.union
    return reduce(index_union_func, pd_indices)


def pd_get_index_intersection(*pd_indices):
    """
    Gets the intersection of any pandas indices.

    :param pd_indices: Pandas indices
    :type pd_indices: pd.Index
    :return: The intersection of the given indices
    :rtype: pd.Index
    """

    all_multi_index = all([isinstance(pd_index, pd.MultiIndex) for pd_index in pd_indices])
    index_intersection_func = pd.MultiIndex.intersection if all_multi_index else pd.Index.intersection
    return reduce(index_intersection_func, pd_indices)


def pd_reindex_to_index_union(input_pd_list, **reindex_kwargs):
    """
    Re-indexes a list of pandas dataframes or series to the union of their indices and columns.

    :param input_pd_list: A list of pandas dataframes or series to reindex
    :type input_pd_list: list of (pd.DataFrame or pd.Series)
    :param reindex_kwargs: Keyword arguments to pass to the reindex function
    :type reindex_kwargs: Any
    :return: A list of reindexed dataframes
    :rtype: list of (pd.DataFrame or pd.Series)
    """
    shared_index = pd_get_index_union(*[input_pd.index for input_pd in input_pd_list])
    output_list = []

    if all([isinstance(input_pd, pd.DataFrame) for input_pd in input_pd_list]):
        shared_columns = pd_get_index_union(
            *[input_pd.columns for input_pd in input_pd_list if isinstance(input_pd, pd.DataFrame)])
        for input_pd in input_pd_list:
            output_list.append(input_pd.reindex(index=shared_index, columns=shared_columns, **reindex_kwargs))
    else:
        for input_pd in input_pd_list:
            output_list.append(input_pd.reindex(index=shared_index, **reindex_kwargs))

    return output_list


def pd_reindex_dict_values_to_index_union(pd_dict):
    """
    Re-indexes the values of a dictionary of pandas dataframes or series to the union of their indices and columns.
    Convenience func for pd_reindex_to_index_union.

    :param pd_dict: A dictionary of pandas dataframes or series to reindex
    :type pd_dict: dict of (pd.DataFrame or pd.Series)
    :return: A dictionary of reindexed dataframes
    :rtype: dict of (pd.DataFrame or pd.Series)
    """
    dict_keys = list(pd_dict.keys())
    dict_values = list(pd_dict.values())
    re_indexed_values = pd_reindex_to_index_union(dict_values)
    return dict(zip(dict_keys, re_indexed_values))


def pd_frames_to_timedelta(frames, fps=30):
    """
    Converts a number of frames to a pandas timedelta with the given fps.

    :param frames: The number of frames
    :type frames: int
    :param fps: The frames per second to use for the conversion
    :type fps: int
    :return: A pandas timedelta representing the given number of frames at the given fps in seconds
    :rtype: pd.Timedelta
    """
    return pd.to_timedelta(frames / fps, unit="s")


def get_column_comparison_matrix(input_df, comparison_func, calculate_diagonal=False):
    """
    Creates a matrix of the results of a comparison function applied to all combinations of columns in a dataframe.
    The resulting matrix is a dataframe with the columns and indices of the input dataframe as its columns and indices
    and the outputs of the comparison function applied to the columns as its values.

    :param input_df: The input dataframe
    :type input_df: pd.DataFrame
    :param comparison_func: The comparison function to apply to all combinations of columns
    :type comparison_func: function
    :param calculate_diagonal: Whether to calculate the diagonal of the matrix, i.e. the comparison of a column with
    itself
    :type calculate_diagonal: bool
    :return: A dataframe with the columns and indices of the input dataframe as its columns and indices and the values
    of the comparison function applied to the columns as its values
    :rtype: pd.DataFrame
    """
    comparison_matrix = pd.DataFrame(index=input_df.columns, columns=input_df.columns)
    for col1, col2 in combinations(input_df.columns, r=2):
        comparison_matrix.loc[col1, col2] = comparison_matrix.loc[col2, col1] = comparison_func(input_df[col1],
                                                                                                input_df[col2])
    if calculate_diagonal:
        for col in input_df.columns:
            comparison_matrix.loc[col, col] = comparison_func(input_df[col], input_df[col])
    return comparison_matrix


def replace_axis_values(input_df, replace_dict, axis=0):
    """
    Replace values in the index or columns of a DataFrame

    :param input_df:
    :type input_df:
    :param replace_dict:
    :type replace_dict:
    :param axis:
    :type axis:
    :return:
    :rtype:
    """
    out_df = input_df.copy()
    out_df = out_df.set_axis(out_df.axes[axis].to_series().replace(replace_dict), axis=axis)
    return out_df


def simplify_numeric_value_chunks(chunks_df, value_tolerance=0, aggregate_function=np.mean):
    """
    Simplifies a dataframe of value chunks by merging chunks with values within a given tolerance of each other and
    aggregating the values of the merged chunks with a given function. The following chunks are then evaluated against
    the merged chunk and merged if they are within the tolerance of the merged chunk.

    :param chunks_df: A dataframe with columns "index_start", "index_end", and "value" representing value chunks
    :type chunks_df: pd.DataFrame
    :param value_tolerance: The tolerance within which to merge chunks
    :type value_tolerance: float
    :param aggregate_function: The function to aggregate the values of merged chunks
    :type aggregate_function: function
    :return: A simplified dataframe of value chunks
    :rtype: pd.DataFrame
    """

    simplified_chunk_list = []
    current_chunk = None
    chunk_values = []
    for _, row in chunks_df.sort_values("index_start").iterrows():
        if current_chunk is None:
            current_chunk = row
            chunk_values = [row["value"]]
        elif abs(row["value"] - current_chunk["value"]) <= value_tolerance:
            current_chunk["index_end"] = row["index_end"]
            chunk_values.append(row["value"])
            current_chunk["value"] = aggregate_function(chunk_values)
        else:
            simplified_chunk_list.append(current_chunk)
            current_chunk = row
            chunk_values = [row["value"]]

    if current_chunk is not None:
        simplified_chunk_list.append(current_chunk)

    return pd.DataFrame(simplified_chunk_list)


def pd_make_df_square(input_df, **reindex_kwargs):
    """
    Makes a dataframe square by reindexing it to the union of its index and columns. This is useful for ensuring that a
    dataframe is square and can be used in matrix operations.

    :param input_df: The input dataframe
    :type input_df: pd.DataFrame
    :param reindex_kwargs: Keyword arguments to pass to the reindex method
    :type reindex_kwargs: Any
    :return: The input dataframe reindexed to the union of its index and columns
    :rtype: pd.DataFrame
    """
    combined_indices = pd_get_index_union(input_df.index, input_df.columns)
    return input_df.reindex(index=combined_indices, columns=combined_indices, **reindex_kwargs)


def pd_flatten_indices(input_pd):
    """
    Flattens all available axes on an arbitrary input pandas object (DataFrame or Series). This function uses the
    to_flat_index method of the index and columns of the input object to flatten them.

    :param input_pd: The input pandas object to flatten
    :type input_pd: pd.DataFrame or pd.Series
    :return: The input object with all axes flattened
    :rtype: pd.DataFrame or pd.Series
    """
    flattened_pd = input_pd.copy()
    flattened_pd.index = flattened_pd.index.to_flat_index()
    if isinstance(input_pd, pd.DataFrame):
        flattened_pd.columns = flattened_pd.columns.to_flat_index()
    return flattened_pd


def pd_filter_index(input_pd, index_filter, insert_missing=True, **reindex_kwargs):
    """
    Filters a pandas object by supplying an iterable of arbitrary values or boolean values to filter the index of
    the DataFrame or Series. Alternatively a callable can be supplied that takes the existing index as input and
    returns either an iterable of values or boolean values. Setting insert_missing to True and supplying non-boolean
    values will insert missing values into the index.

    :param input_pd: A pandas DataFrame or Series to filter
    :type input_pd: pd.DataFrame or pd.Series
    :param index_filter: An iterable of values or boolean values to filter the index of the input pandas object
    :type index_filter: Any
    :type index_filter: bool
    :param insert_missing: Whether to insert missing values into the index if the index_filter is not boolean
    :type insert_missing: bool
    :return: The input pandas object filtered by the index_filter
    :rtype: pd.DataFrame or pd.Series
    """

    if callable(index_filter):
        index_filter = index_filter(input_pd.index)

    # check whether all values are boolean
    if all(isinstance(val, bool) for val in index_filter):
        return input_pd[index_filter]

    index_filter = pd.Index(index_filter)
    if not insert_missing:
        index_filter = index_filter.intersection(input_pd.index)
    return input_pd.reindex(index=index_filter, **reindex_kwargs)


def consecutive_boolean_counter(boolean_series):
    """
    Adapted from https://stackoverflow.com/a/45965003.
    Counts the number of consecutive True values in a boolean series and resets the count when a False value is
    encountered. The function returns a series with the same index as the input series where each value represents the
    number of consecutive True values up to that point.
    :param boolean_series: A boolean series
    :type boolean_series: pd.Series
    :return: A series with the same index as the input series where each value represents the number of consecutive True
        values up to that point
    :rtype: pd.Series
    """
    return boolean_series.cumsum() - boolean_series.cumsum().where(~boolean_series).ffill().fillna(0).astype(int)


def combine_multi_index_levels(input_pd, levels_to_combine, combined_idx_name="combined_idx", combined_idx_position=-1):
    """
    Extracts a new index from an existing index by combining multiple levels into a single level. The levels to combine
    are specified by their names or positions in the index. The new index is created by combining the specified levels
    into a single level with the name combined_idx_name at the specified position in the index.

    :param input_pd: The input pandas object with a MultiIndex
    :type input_pd: pd.DataFrame or pd.Series
    :param levels_to_combine: The level labels to combine into a single level. Important: The levels to combine must be
    labels, as positions won't work with the groupby method.
    :type levels_to_combine: list of str
    :param combined_idx_name: The name of the new combined index level
    :type combined_idx_name: str
    :param combined_idx_position: The position to insert the new combined index level into the index
    :type combined_idx_position: int
    :return: A new index with the specified levels combined into a single level and the mapping of the original index
    values to the new combined index values
    :rtype: pd.MultiIndex, pd.Series
    """
    remaining_levels = input_pd.droplevel(levels_to_combine).index.names
    combined_idx_mapping = input_pd.groupby(levels_to_combine).ngroup().droplevel(remaining_levels).rename(
        combined_idx_name).drop_duplicates()
    new_index_frame = input_pd.index.to_frame().set_index(levels_to_combine)
    new_index_frame.insert(loc=range(len(new_index_frame.columns) + 1)[combined_idx_position],
                           column=combined_idx_name, value=combined_idx_mapping)
    new_index = pd.MultiIndex.from_frame(new_index_frame)
    return new_index, combined_idx_mapping


def extract_neighbourhood_df(input_series_or_df, neighbourhood_reference_indices, span_before=0, span_after=0,
                             neighbourhood_end_indices=None, span_index_level=-1):
    """
    Extracts a neighbourhood of values from the given indices from a series or dataframe with a MultiIndex. The
    neighbourhood is defined by the span_before and span_after parameters, which specify the number of values to extract
    before and after the given indices. The span_index_level parameter specifies the level of the MultiIndex to use for
    the span. The neighbourhood_reference_indices can be a list of tuples, an pd.Index-compatible iterable, or a MultiIndex
    or Index. It's dimensionality needs to match the dimensionality of the input_series_or_df's index. Alternatively, a
    neighbourhood_end_indices can be given, which will lead to neighbourhoods being extracted from between
    neighbourhood_reference_indices and neighbourhood_end_indices. The neighbourhood_end_indices must have the same
    dimensionality as the neighbourhood_reference_indices.

    Probably usually faster and more memory-efficient than extract_series_neighbourhood_df.

    :param input_series_or_df: The input series or dataframe
    :type input_series_or_df: pd.Series or pd.DataFrame
    :param neighbourhood_reference_indices: The indices around which to extract the neighbourhood. The dimensionality of
    this index must match the dimensionality of the input_series_or_df's index. If a MultiIndex is given, the level 
    names will be ignored (since the dimensionality must match anyway). 
    :type neighbourhood_reference_indices: pd.Index or pd.MultiIndex
    :param span_before: The number of values to extract before the center indices
    :type span_before: int
    :param span_after: The number of values to extract after the center indices
    :type span_after: int
    :param neighbourhood_end_indices: The end indices of the neighbourhoods. If None, the span_before and span_after
    parameters will be used to calculate the end indices. If not None, then each span will be calculated as the range
    from the start index to the end index.
    :type neighbourhood_end_indices: pd.Index or pd.MultiIndex or None
    :param span_index_level: The level of the MultiIndex to use for the span. Do not change from the default -1 if the
    Series has only a simple Index.
    :type span_index_level: int
    :return: A DataFrame with the extracted neighbourhoods around the given indices. Has a index of the given
    neighbourhood_reference_indices and columns of range(-span_before, span_after + 1) representing the values around the
    center indices in the index. The range is relative to the span_index_level level of neighbourhood_reference_indices.
    If the input_series_or_df is a DataFrame, the columns will be a MultiIndex with the original columns as the first
    levels and the range(-span_before, span_after + 1) as the final level.
    :rtype: pd.DataFrame
    """

    if not isinstance(neighbourhood_reference_indices, (pd.Index, pd.MultiIndex)):
        if isinstance(neighbourhood_reference_indices[0], tuple):
            neighbourhood_reference_indices = pd.MultiIndex.from_tuples(neighbourhood_reference_indices)
        else:
            neighbourhood_reference_indices = pd.Index(neighbourhood_reference_indices)

    if neighbourhood_end_indices is None:
        center_index_df = pd.DataFrame(index=neighbourhood_reference_indices,
                                       data=dict(span_before=span_before, span_after=span_after))
        center_index_df["span_start"] = center_index_df.index.get_level_values(span_index_level) - center_index_df[
            "span_before"]
        center_index_df["span_end"] = center_index_df.index.get_level_values(span_index_level) + center_index_df[
            "span_after"]
    else:
        start_indices = neighbourhood_reference_indices.get_level_values(span_index_level)
        end_indices = neighbourhood_end_indices.get_level_values(span_index_level)
        center_index_df = pd.DataFrame(index=neighbourhood_reference_indices,
                                       data=dict(span_start=start_indices, span_end=end_indices))

    slice_index = expand_index_from_limits(center_index_df, start_col="span_start",
                                           end_col="span_end", maintain_index=True,
                                           index_name="neighbourhood_index").index

    expanded_slice_index_df = slice_index.to_frame().droplevel(level="neighbourhood_index", axis=0)
    expanded_slice_index_df["neighbourhood_index"] -= expanded_slice_index_df.index.get_level_values(span_index_level)
    expanded_slice_index = pd.MultiIndex.from_frame(expanded_slice_index_df)

    if isinstance(span_index_level, str):
        normed_span_index_level = span_index_level
    else:
        normed_span_index_level = list(range(neighbourhood_reference_indices.nlevels))[span_index_level]
    slice_index = slice_index.droplevel(normed_span_index_level)

    slice_index_filter = [_idx in input_series_or_df.index for _idx in slice_index]
    output_df = input_series_or_df.loc[slice_index[slice_index_filter]]
    output_df.index = expanded_slice_index[slice_index_filter]
    output_df = output_df.unstack("neighbourhood_index").reindex(index=neighbourhood_reference_indices)
    output_df.index.names = input_series_or_df.index.names
    return output_df


def extract_series_neighbourhood_df(input_series, neighbourhood_center_indices,
                                    span_before=0, span_after=0, span_index_level=-1,
                                    ignore_missing_multiindex_centers=False, parallelize=True):
    """
    Extracts a neighbourhood of values around the given indices from a series with a MultiIndex. The neighbourhood is
    defined by the span_before and span_after parameters, which specify the number of values to extract before and after
    the given indices. The span_index_level parameter specifies the level of the MultiIndex to use for the span. If the
    ignore_missing_multiindex_centers parameter is set to True, missing MultiIndex centers will be ignored and an empty
    series will be returned for them.

    There is no detection of missing centers in Series with a simple Index. The resulting neighbourhood_df will contain
    NaN values for missing values, including completely missing centers in Series with an Index, or Series with a
    MultiIndex with ignored missing centers, or values in the range of the span that are missing.

    :param input_series: The input series with a MultiIndex
    :type input_series: pd.Series
    :param neighbourhood_center_indices: The indices around which to extract the neighbourhood
    :type neighbourhood_center_indices: list of tuple or list of int or pd.Index or pd.MultiIndex
    :param span_before: The number of values to extract before the center indices
    :type span_before: int
    :param span_after: The number of values to extract after the center indices
    :type span_after: int
    :param span_index_level: The level of the MultiIndex to use for the span. Do not change from the default -1 if the
    Series has only a simple Index.
    :type span_index_level: int
    :param ignore_missing_multiindex_centers: Whether to ignore missing MultiIndex centers and return an empty series in
    their place
    :type ignore_missing_multiindex_centers: bool
    :return: A DataFrame with the extracted neighbourhoods around the given indices. Has a index of the given
    neighbourhood_center_indices and columns of range(-span_before, span_after + 1) representing the values around the
    center indices in the index. The range is relative to the span_index_level level of neighbourhood_center_indices.
    """
    input_series = input_series.sort_index()
    input_is_df = isinstance(input_series, pd.DataFrame)

    center_is_multiindex = isinstance(neighbourhood_center_indices[0], tuple)

    def _extract_center_neighbourhood(center):
        if not center_is_multiindex:
            center = (center,)
        center_list = list(center)
        center_list[span_index_level] = slice(center_list[span_index_level] - span_before,
                                              center_list[span_index_level] + span_after, 1)

        try:
            if not input_is_df:
                neighbourhood_slice = input_series.loc[pd.IndexSlice[tuple(center_list)]]
            else:
                neighbourhood_slice = input_series.loc[pd.IndexSlice[tuple(center_list)], :]
            neighbourhood_slice.index = (neighbourhood_slice.index.get_level_values(span_index_level) - center[
                span_index_level]).rename("neighbourhood_index")
        except KeyError as e:
            if ignore_missing_multiindex_centers:
                if not input_is_df:
                    neighbourhood_slice = pd.Series()
                else:
                    neighbourhood_slice = pd.DataFrame(columns=input_series.columns)
            else:
                raise e

        neighbourhood_slice = neighbourhood_slice.reindex(range(-span_before, span_after + 1), fill_value=np.nan)
        if isinstance(neighbourhood_slice, pd.DataFrame):
            neighbourhood_slice = neighbourhood_slice.unstack()  # unstack to series
        neighbourhood_slice.name = center
        return neighbourhood_slice

    if parallelize:
        n_cpu = min(os.cpu_count(), len(neighbourhood_center_indices))
        batch_size = max(int(len(neighbourhood_center_indices) / n_cpu), 1)
        with Parallel(n_jobs=n_cpu, verbose=1, batch_size=batch_size) as parallel:
            neighbourhood_slice_list = parallel(delayed(_extract_center_neighbourhood)(center)
                                                for center in neighbourhood_center_indices)
    else:
        neighbourhood_slice_list = [_extract_center_neighbourhood(center) for center in neighbourhood_center_indices]

    neighbourhood_df = pd.concat(neighbourhood_slice_list, axis=1).T
    neighbourhood_df.index.names = input_series.index.names
    return neighbourhood_df


def get_value_sequence(input_series, values_before=0, values_after=0, include_nan=True):
    """
    A function that extracts a sequence of values around each value in a series. The sequence is defined by the number
    of values_before and values_after the center value. The function returns a series of tuples with the extracted
    sequences. The include_nan parameter specifies whether to include sequences with NaN values in them.
    The function attempts to preserve the original dtype of the input series if there are no NaN values in the output.

    An example of the output of this function is as follows:

    input_series = pd.Series([1, 2, 3, 4, 5], index=range(5), dtype=int)
    get_value_sequence(input_series, value_before=1, values_after=1, include_nan=False)

    Output:
    1    (1, 2, 3)
    2    (2, 3, 4)
    3    (3, 4, 5)
    dtype: int64

    :param input_series: The input series
    :type input_series: pd.Series
    :param values_before: The number of values to extract before the center value
    :type values_before: int
    :param values_after: The number of values to extract after the center value
    :type values_after: int
    :param include_nan: Whether to include sequences with NaN values in them
    :type include_nan: bool
    :return: A series of tuples with the extracted sequences
    :rtype: pd.Series
    """
    original_dtype = input_series.dtype
    shifted_series_list = [input_series.shift(-i) for i in range(-values_before, values_after + 1)]
    value_sequence_df = pd.concat(shifted_series_list, axis=1)
    if not include_nan:
        value_sequence_df = value_sequence_df.dropna(how="any")
    if not value_sequence_df.isna().any().any():
        value_sequence_df = value_sequence_df.astype(original_dtype)
    return value_sequence_df.apply(lambda x: tuple(x), axis=1)


def get_single_sequence_occurrence_series(input_series, target_sequence, values_before=None, values_after=None):
    if values_before is None:
        values_before = 0
    if values_after is None:
        values_after = len(target_sequence) - values_before - 1
    sequence_series = get_value_sequence(input_series, values_before=values_before, values_after=values_after, include_nan=False)
    return sequence_series.apply(lambda x: x == target_sequence)


def get_sequence_occurrence_series(input_series, *target_sequences, values_before=None, values_after=None):
    sequence_occurrence_list = []
    for target_sequence in target_sequences:
        sequence_occurrence_list.append(get_single_sequence_occurrence_series(input_series, target_sequence, values_before=values_before, values_after=values_after))
    sequence_occurence_df = pd.concat(sequence_occurrence_list, axis=1)
    return sequence_occurence_df.any(axis=1)


def get_single_pattern_occurrence_series(input_series, target_pattern):
    syllable_series = input_series
    syllable_chunks = series_to_value_chunks(syllable_series)

    values_before = 0
    values_after = len(target_pattern) - 1

    syllable_chunks["target_sequence_start"] = get_sequence_occurrence_series(
        syllable_chunks["value"], target_pattern, values_before=values_before, values_after=values_after)
    syllable_chunks["part_of_sequence"] = pd.DataFrame(
        get_value_sequence(syllable_chunks["target_sequence_start"], values_before=values_after,
                           values_after=values_before, include_nan=True).to_list()
    ).any(axis=1)
    sequence_value_chunks = syllable_chunks.loc[:, ["index_start", "index_end", "part_of_sequence"]].rename(
        {"part_of_sequence": "value"}, axis=1)
    return value_chunks_to_series(sequence_value_chunks, slice_index_name=input_series.index.name)


def get_pattern_occurrence_series(input_series, *target_patterns):
    pattern_occurrence_list = []
    for target_pattern in target_patterns:
        pattern_occurrence_list.append(get_single_pattern_occurrence_series(input_series, target_pattern))
    pattern_occurrence_df = pd.concat(pattern_occurrence_list, axis=1)
    return pattern_occurrence_df.any(axis=1)


def reindex_df_to_col_product(input_df, product_cols):
    product_values = [input_df[col].unique() for col in product_cols]
    reindexed_df = input_df.reset_index(drop=False).set_index(product_cols)
    return reindexed_df.reindex(index=pd.MultiIndex.from_product(product_values, names=product_cols))
