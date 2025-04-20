import warnings

import pandas as pd
from .funcs import expand_index_from_limits


def multi_index_series_to_value_chunks(input_series, slice_index=-1, skip_values=tuple()):
    """
    Extracts chunks of values from a series with a MultiIndex, skipping values in skip_values. The chunks are returned
    as a DataFrame with columns "index_start", "index_end", "value", and "grouped_index". The grouped_index is the
    remaining index, excluding the slice_index. The "index_start" and "index_end" columns represent the start and end
    of a "slice_index" slice under the "grouped_index" with the same "value" throughout the slice.
    To ensure proper operation, the input_series must have a MultiIndex and will be sorted by all levels, with the
    slice_index being the last level to be sorted.

    :param input_series: The series to extract chunks from.
    :type input_series: pd.Series
    :param slice_index: The index to slice the MultiIndex by. Can also be a string representing the name of the index.
    :type slice_index: int or str
    :param skip_values: Values to skip when extracting chunks.
    :type skip_values: tuple or list
    :return: A DataFrame with columns "index_start", "index_end", "value", and "grouped_index".
    :rtype: pd.DataFrame
    """
    assert isinstance(input_series.index, pd.MultiIndex), "The input series must have a MultiIndex."

    if isinstance(slice_index, str):
        slice_index = input_series.index.names.index(slice_index)

    input_series = input_series.copy()
    index_level_list = list(range(input_series.index.nlevels))
    index_level_list.pop(slice_index)
    input_series = input_series.sort_index(level=index_level_list + [slice_index])

    index_level_list = index_level_list[0] if len(index_level_list) == 1 else index_level_list

    value_chunk_df_list = []
    for group_index, group_series in input_series.groupby(level=index_level_list):
        group_index = group_index if isinstance(group_index, tuple) else (group_index,)
        group_series.index = group_series.index.droplevel(index_level_list)
        group_chunks = flat_index_series_to_value_chunks(group_series, skip_values=skip_values)
        group_chunks["grouped_index"] = [group_index] * len(group_chunks)
        value_chunk_df_list.append(group_chunks)

    value_chunk_df = pd.concat(value_chunk_df_list)
    value_chunk_df["length"] = value_chunk_df["index_end"] - value_chunk_df["index_start"] + 1
    value_chunk_df = value_chunk_df.reset_index(drop=True)
    return value_chunk_df


def flat_index_series_to_value_chunks(input_series, skip_values=tuple()):
    """
    Extracts chunks of values from a series, skipping values in skip_values. Utilizes factorize to convert the series to
    numeric values and then replaces the numeric values with the original values after filtering out the skip values.
    Due to the utilization of factorize, this function is more efficient than the previous implementation, but it is not
    able to differentiate between different values with the same numeric representation, such as None and np.nan.

    :param input_series: The series to extract chunks from.
    :type input_series: pd.Series
    :param skip_values: Values to skip when extracting chunks.
    :type skip_values: tuple or list
    :return: A DataFrame with columns "index_start", "index_end", "value", and "length". Each row represents a chunk
        with limits.
    :rtype: pd.DataFrame
    """
    codes, uniques = input_series.factorize(use_na_sentinel=False)
    uniques_dict = dict(enumerate(list(uniques)))
    chunks = series_to_numeric_chunks(pd.Series(data=codes, index=input_series.index))
    chunks["value"] = chunks["value"].map(uniques_dict.get)
    chunks["length"] = chunks["index_end"] - chunks["index_start"] + 1
    filtered_chunks = chunks[~chunks["value"].isin(skip_values)]
    filtered_chunks = filtered_chunks.reset_index(drop=True)
    return filtered_chunks


def series_to_value_chunks(input_series, skip_values=tuple(), slice_index=None):
    """
    A wrapper function for flat_index_series_to_value_chunks and multi_index_series_to_value_chunks. Determines whether
    the input series has a MultiIndex or not and calls the appropriate function. Puts out a warning if the slice_index
    is explicitly given and the input_series is not a MultiIndex series.

    :param input_series: The series to extract chunks from.
    :type input_series: pd.Series
    :param skip_values: Values to skip when extracting chunks.
    :type skip_values: tuple or list
    :param slice_index: The index to slice the MultiIndex by. Can also be a string representing the name. Only relevant
        for input_series with a MultiIndex.
    :type slice_index: int or str
    :return: A DataFrame with columns "index_start", "index_end", "value", and "length". Each row represents a chunk
        with limits. If input_series is a MultiIndex series, this also includes a "grouped_index" column representing
        the remaining index, excluding the slice_index.
    :rtype: pd.DataFrame
    """

    if isinstance(input_series.index, pd.MultiIndex):
        if slice_index is None:
            slice_index = -1
        return multi_index_series_to_value_chunks(input_series, slice_index=slice_index, skip_values=skip_values)
    else:
        if slice_index is not None:
            warnings.warn("The slice_index parameter is only relevant for MultiIndex series. Ignoring slice_index.")
        return flat_index_series_to_value_chunks(input_series, skip_values=skip_values)


def series_to_numeric_chunks(input_series, skip_values=tuple()):
    """
    Extracts chunks of numeric values from a series, skipping values in skip_values.
    The chunks are returned as a DataFrame with columns "index_start", "index_end", and "value".

    :param input_series: The series to extract chunks from.
    :type input_series: pd.Series
    :param skip_values: Values to skip when extracting chunks.
    :type skip_values: tuple or list
    :return: A DataFrame with columns "index_start", "index_end", and "value". Each row represents a chunk with limits.
    :rtype: pd.DataFrame
    """
    transitions = ~input_series.diff().fillna(0).eq(0)
    transitions.iloc[0] = True
    values = input_series[transitions]
    value_chunk_df = pd.DataFrame({
        "index_start": values.index,
        "index_end": input_series.index[transitions.shift(-1, fill_value=False)].append(
            pd.Index([input_series.index[-1]])),
        "value": values
    })
    value_chunk_df = value_chunk_df[~value_chunk_df["value"].isin(skip_values)]
    return value_chunk_df


def boolean_series_to_chunks(boolean_series, skip_values=tuple()):
    group_chunks = boolean_series.astype(int).diff().fillna(0).astype(bool).cumsum()

    chunk_row_dicts = []
    for group_index, group_series in boolean_series.groupby(group_chunks):
        group_value = group_series.iloc[0]
        if group_value in skip_values:
            continue
        chunk_row_dicts.append({
            "index_start": group_series.index[0],
            "index_end": group_series.index[-1],
            "value": group_value
        })
    return pd.DataFrame(chunk_row_dicts)


def value_chunks_to_multi_index_series(value_chunk_df, slice_index_name="index",
                                       grouped_index_col=None, grouped_index_names=None):
    """
    Converts a chunks-DataFrame with columns "index_start", "index_end", and "value" (optionally "grouped_index") to a
    multi-index series. The "index_start" and "index_end" columns represent the start and end of a "slice_index" slice
    under the "grouped_index" with the same "value" throughout the slice.

    :param value_chunk_df: A DataFrame with columns "index_start", "index_end", and "value" (optionally "grouped_index")
    :type value_chunk_df: pd.DataFrame
    :param slice_index_name: The name of the slice index
    :type slice_index_name: str
    :param grouped_index_col: The name of the column in value_chunk_df to use as the grouped index
    :type grouped_index_col: str or None
    :param grouped_index_names: The names of the levels in the grouped index. Only relevant if grouped_index_col is not
        None.
    :type grouped_index_names: list or tuple
    :return: A multi-index series with the values from the value_chunk_df
    :rtype: pd.Series
    """
    out_series = value_chunk_df.copy()
    _maintain_index = grouped_index_col is not None
    if grouped_index_col is not None:
        out_series.index = pd.MultiIndex.from_tuples(out_series[grouped_index_col], names=grouped_index_names)

    expanded_value_df = expand_index_from_limits(
        input_df=out_series,
        start_col="index_start", end_col="index_end",
        index_name=slice_index_name, end_inclusive=True,
        maintain_index=_maintain_index)

    return expanded_value_df["value"]


def value_chunks_to_flat_index_series(value_chunk_df, slice_index_name="index"):
    """
    Converts a chunks-DataFrame with columns "index_start", "index_end", and "value" to a series. The "index_start" and
    "index_end" columns represent the start and end of a "slice_index" slice with the same "value" throughout the slice.
    Uses value_chunks_to_multi_index_series with grouped_index_col=None and grouped_index_names=None.

    :param value_chunk_df: A DataFrame with columns "index_start", "index_end", and "value"
    :type value_chunk_df: pd.DataFrame
    :param slice_index_name: The name of the slice index
    :type slice_index_name: str
    :return: A series with the values from the value_chunk_df
    :rtype: pd.Series
    """
    return value_chunks_to_multi_index_series(value_chunk_df, slice_index_name=slice_index_name,
                                              grouped_index_col=None, grouped_index_names=None)


def value_chunks_to_series(value_chunk_df, slice_index_name="index", grouped_index_col=None, grouped_index_names=None):
    """
    A wrapper around value_chunks_to_multi_index_series and value_chunks_to_flat_index_series. Determines whether the
    output series should be a MultiIndex series or not based on whether grouped_index_col is given and accordingly
    calls the appropriate function.

    :param value_chunk_df: A DataFrame with columns "index_start", "index_end", and "value" (optionally "grouped_index")
    :type value_chunk_df: pd.DataFrame
    :param slice_index_name: The name of the slice index. Defaults to "index".
    :type slice_index_name: str
    :param grouped_index_col: The name of the column in value_chunk_df to use as the grouped index. Defaults to None.
    Determines whether the output series should be a MultiIndex series or not.
    :param grouped_index_names: The names of the levels in the grouped index. Only relevant if grouped_index_col is not
    None. Defaults to None.
    :return: A series with the values from the value_chunk_df
    :rtype: pd.Series
    """
    if grouped_index_col is None:
        return value_chunks_to_flat_index_series(value_chunk_df, slice_index_name=slice_index_name)
    else:
        return value_chunks_to_multi_index_series(value_chunk_df, slice_index_name=slice_index_name,
                                                  grouped_index_col=grouped_index_col,
                                                  grouped_index_names=grouped_index_names)
