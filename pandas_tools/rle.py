import warnings

import pandas as pd
from .funcs import expand_index_from_limits


def rle_numeric_single_index_series(input_series, drop_run_values=tuple()):
    """
    Run-length encodes a single-index series of numeric values. The series index is assumed to be sorted.
    The drop_run_values parameter can be used to drop specific run values from the output.

    :param input_series: A single-index series of numeric values
    :type input_series: pd.Series
    :param drop_run_values: Values to drop from the output
    :type drop_run_values: tuple or list
    :return: A DataFrame with columns "end_index", "run_length", and "run_value" representing the run
        lengths of the input series, and a "start_index" index
    :rtype: pd.DataFrame
    """
    transitions = ~input_series.diff().fillna(0).eq(0)
    transitions.iloc[0] = True

    run_starts = input_series[transitions].index
    run_ends = input_series.index[transitions.shift(-1, fill_value=False)].append(pd.Index([input_series.index[-1]]))
    run_lengths = run_ends - run_starts + 1

    rle_df = pd.DataFrame(dict(
        end_index=run_ends,
        run_length=run_lengths,
        run_value=input_series[transitions].values
    ), index=run_starts.rename("start_index"))

    rle_df = rle_df[~rle_df["run_value"].isin(drop_run_values)]
    return rle_df


def rle_arbitrary_single_index_series(input_series, drop_run_values=tuple()):
    """
    Run-length encodes a single-index series of arbitrary values. The series index is assumed to be sorted.
    The drop_run_values parameter can be used to drop specific run values from the output.
    Factorizes and encodes the run values as integers before passing them to rle_numeric_single_index_series
    for run-length encoding.

    :param input_series: A single-index series of arbitrary values
    :type input_series: pd.Series
    :param drop_run_values: Values to drop from the output
    :type drop_run_values: tuple or list
    :return: A DataFrame with columns "end_index", "run_length", and "run_value" representing the run
        lengths of the input series, and a "start_index" index
    """
    codes, uniques = input_series.factorize(use_na_sentinel=False)
    uniques_dict = dict(enumerate(list(uniques)))
    inv_uniques_dict = {v:k for k,v in uniques_dict.items()}
    mapped_drop_values = [inv_uniques_dict.get(value) for value in drop_run_values if value in inv_uniques_dict]

    rle_df = rle_numeric_single_index_series(pd.Series(data=codes, index=input_series.index),
                                             drop_run_values=mapped_drop_values)
    rle_df["run_value"] = rle_df["run_value"].map(uniques_dict.get)
    return rle_df


def rle_multi_index_series(input_series, drop_run_values=tuple(), slice_index=-1):
    """
    Run-length encodes a multi-index series. The run-length is calculated for each group of the series based on all
    levels except for the slice_index level, which is used to calculate the run-lengths. The drop_run_values parameter
    can be used to drop specific run values from the output.

    Before passing the groups to rle_arbitrary_single_index_series, the index is sorted by all levels in order, with the
    slice_index level being sorted last.

    :param input_series: An arbitrary multi-index series
    :type input_series: pd.Series
    :param drop_run_values: Values to drop from the output
    :type drop_run_values: tuple or list
    :param slice_index: The index to slice the MultiIndex by. Can also be a string representing the name.
    :type slice_index: int or str
    :return: A DataFrame with columns "end_index", "run_length", and "run_value" representing the run lengths of the
        input series, and a multi-index containing the remaining index levels (excluding slice_index) followed by the
        "start_index" index.
    :rtype: pd.DataFrame
    """
    assert isinstance(input_series.index, pd.MultiIndex), "The input series must have a MultiIndex."

    if isinstance(slice_index, str):
        slice_index = input_series.index.names.index(slice_index)

    input_series = input_series.copy()

    index_level_list = list(range(input_series.index.nlevels))
    index_level_list.pop(slice_index)

    index_name_list = list(input_series.index.names)
    del index_name_list[slice_index]

    input_series = input_series.sort_index(level=index_level_list + [slice_index])
    index_level_list = index_level_list[0] if len(index_level_list) == 1 else index_level_list

    rle_df_list = []
    for group_index, group_series in input_series.groupby(level=index_level_list):
        group_series.index = group_series.index.droplevel(index_level_list)
        group_rle_df = rle_arbitrary_single_index_series(group_series, drop_run_values=drop_run_values)

        if not isinstance(group_index, tuple):
            group_index = (group_index,)
        index_rows = [tuple(group_index) + (index,) for index in group_rle_df.index]
        group_rle_df.index = pd.MultiIndex.from_tuples(tuples=index_rows, names=[*index_name_list, "start_index"])
        rle_df_list.append(group_rle_df)

    rle_df = pd.concat(rle_df_list)
    return rle_df


def rle_series(input_series, drop_run_values=tuple(), slice_index=None):
    """
    A convenient wrapper for rle_multi_index_series and rle_arbitrary_single_index_series. Determines whether the input
    series is a MultiIndex series or not based on the index and accordingly calls the appropriate function.

    See the documentation for rle_multi_index_series and rle_arbitrary_single_index_series for more information.

    :param input_series: An arbitrary series
    :type input_series: pd.Series
    :param drop_run_values: Values to drop from the output
    :type drop_run_values: tuple or list
    :param slice_index: The index to slice the MultiIndex by. Can also be a string representing the name. Only used for
        MultiIndex series. If passed with a single-index series, it will be ignored (a warning is shown).
    :return: A DataFrame with columns "end_index", "run_length", and "run_value" representing the run lengths of the
    input series. The index is either a single index or a multi-index. See the documentation for rle_multi_index_series
    and rle_arbitrary_single_index_series for more information on the index of the returned DataFrame.
    :rtype: pd.DataFrame
    """
    if isinstance(input_series.index, pd.MultiIndex):
        if slice_index is None:
            slice_index = -1
        return rle_multi_index_series(input_series, drop_run_values=drop_run_values, slice_index=slice_index)
    else:
        if slice_index is not None:
            warnings.warn("The slice_index parameter is only relevant for MultiIndex series. Ignoring slice_index.")
        return rle_arbitrary_single_index_series(input_series, drop_run_values=drop_run_values)


def expand_rle_df_to_series(input_rle_df, index_name="index", series_name="run_value"):
    """
    Expands a run-length encoded DataFrame to a Series. The run-length encoded DataFrame is expected to have columns
    "end_index", "run_length", and "run_value" representing the run lengths of the input series, and a "start_index"
    level at the end of its MultiIndex, or just a Index named "start_index". The returned Series will have the same
    index as the input run-length encoded DataFrame and will be named "run_value". This is the inverse operation of
    rle_series. As the index name and series name are lost during rle_series, they can be passed as parameters to this
    function.

    The internal expansion of the run-length encoded DataFrame is done using the expand_index_from_limits function.
    See the documentation for expand_index_from_limits for more information.

    :param input_rle_df: A run-length encoded DataFrame
    :type input_rle_df: pd.DataFrame
    :param index_name: The name of the Index, or the last level of the MultiIndex, of the returned Series. Defaults to
        "index".
    :type index_name: str
    :param series_name: The name of the returned Series. Defaults to "run_value".
    :type series_name: str
    :return: A Series representing the run-length encoded DataFrame
    :rtype: pd.Series
    """
    maintain_index = isinstance(input_rle_df.index, pd.MultiIndex)

    expanded_value_df = expand_index_from_limits(
        input_df=input_rle_df.reset_index("start_index"),
        start_col="start_index", end_col="end_index",
        index_name=index_name, end_inclusive=True,
        maintain_index=maintain_index)

    return expanded_value_df["run_value"].rename(series_name)
