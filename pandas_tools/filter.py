import numpy as np
import pandas as pd

from socialscan_tools.data_processing import fit_series_to_frame


def get_modified_zscore(input_series):
    """
    From my old project.
    Get the modified zscore for each position in a pandas Series. The modified zscore is calculated according to
    https://hwbdocuments.env.nm.gov/Los%20Alamos%20National%20Labs/TA%2054/11587.pdf.
    :param input_series: Pandas Series of numeric values.
    :type input_series: pd.Series
    :return: Series of modified zscores
    :rtype: pd.Series
    """
    series_median = input_series.median(skipna=True)
    series_absolute_deviation = (input_series - series_median).abs()
    series_mad = series_absolute_deviation.median(skipna=True)  # mad = Median Absolute Deviation
    series_modified_zscore = 0.6745 * (input_series - series_median) / series_mad
    return series_modified_zscore


def get_modified_zscore_filter(input_series, threshold=3.5):
    """
    Get a boolean mask indicating whether each position in a pandas Series has a zscore above the threshold. Uses the
    modified zscore as calculated by get_modified_zscore.
    :param input_series: Pandas Series of numeric values.
    :type input_series: pd.Series
    :param threshold: The minimum zscore threshold
    :type threshold: float
    :return: Boolean mask, with True where zscore > threshold
    :rtype: pd.Series
    """
    series_modified_zscore = get_modified_zscore(input_series)
    series_filter = series_modified_zscore.abs().ge(threshold)
    return series_filter


def get_rolling_modified_zscore_filter(input_series, threshold=3, window_size=30, min_periods=1, nan_to_true=False):
    """
    Get a boolean mask indicating whether each position in a pandas Series has a rolling zscore above the threshold.
    :param input_series: Pandas Series of numeric values
    :type input_series: pd.Series
    :param threshold: The minimum zscore threshold
    :type threshold: float
    :param window_size: Size of the rolling window used to calculate the mean and standard deviation
    :type window_size: int
    :param min_periods: Minimum number of values required in a window to calculate the local zscore
    :type min_periods: int
    :param nan_to_true: Whether to translate any undefined z-scores to a True entry in the filter
    :type nan_to_true: bool
    :return: Boolean mask series, with True where local zscore > threshold
    :rtype: pd.Series
    """
    # shift suggested by https://stackoverflow.com/a/47165379
    series_median = input_series.rolling(window_size, min_periods=min_periods).median(skipna=True).shift(1)
    series_absolute_deviation = (input_series - series_median).abs()
    series_mad = series_absolute_deviation.rolling(window_size, min_periods=min_periods).median(
        skipna=True)  # mad = Median Absolute Deviation
    series_modified_zscore = 0.6745 * (input_series - series_median) / series_mad
    series_filter = series_modified_zscore.abs().ge(threshold)
    if nan_to_true:
        series_filter[series_modified_zscore.isnull().any(axis=1)] = True
    return series_filter


def filter_series_by_modified_zscore(input_series, z_threshold=3.5):
    """
    Helper function that applies the boolean mask from get_modified_zscore_filter to filter a pandas Series.
    :param input_series: Pandas Series to be filtered
    :type input_series: pd.Series
    :param z_threshold: Z-score above which the filter is to be applied
    :type z_threshold: float
    :return: Series filtered by the modified zscore < z_threshold
    :rtype: pandas.Series
    """
    filtered_series = input_series.where(~get_modified_zscore_filter(input_series, z_threshold))
    return filtered_series


def filter_series_by_rolling_modified_zscore(input_series, z_threshold=3.5, **kwargs):
    """
    Helper function that applies the boolean mask from get_modified_zscore_filter to filter a pandas Series.
    :param input_series: Pandas Series to be filtered
    :type input_series: pd.Series
    :param z_threshold: Z-score above which the filter is to be applied
    :type z_threshold: float
    :return: Series filtered by the modified zscore < z_threshold
    :rtype: pandas.Series
    """
    filtered_series = input_series.where(~get_rolling_modified_zscore_filter(input_series, z_threshold, **kwargs))
    return filtered_series


def filter_df_by_rolling_modified_zscore(input_df, cols_to_filter=None, z_threshold=3.5, **kwargs):
    output_df = input_df.copy()
    filter_cols = cols_to_filter if cols_to_filter is not None else input_df.columns
    for col in filter_cols:
        output_df[col] = filter_series_by_rolling_modified_zscore(input_df[col], z_threshold=z_threshold, **kwargs)
    output_df.loc[output_df[filter_cols].isna().any(axis=1), filter_cols] = np.nan
    return output_df


def filter_split_pd(input_pd, boolean_filter_series):
    boolean_filter = boolean_filter_series if isinstance(input_pd, pd.Series) else fit_series_to_frame(
        boolean_filter_series, input_pd)
    return input_pd.where(~boolean_filter), input_pd.where(boolean_filter)
