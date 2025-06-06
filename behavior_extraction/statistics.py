"""
This module contains functions related to the analysis of feature vector dataframes. The general assumption here is that
there is a Pandas dataframe with a "sample feature" MultiIndex and each row representing one samples features (the
columns). E.g. you take the behavior transition and proportion series created by bout_df_to_transition_series and
bout_df_to_proportion_series from the funcs module, and concatenate them for each sample. This module contains
functions to analyze the data in this dataframe.
"""
from sklearn.metrics import pairwise_distances, silhouette_samples
import pandas as pd
import numpy as np


def glass_delta(control_group, test_group):
    """
    Calculate the glass delta between two groups. The glass delta is defined as the difference in mean divided by the
    standard deviation of the control group. This is a measure of how much the test group differs from the control
    group, on average, and how that difference compares to the variability within the control group.
    :param control_group: A Pandas Series containing the values for the control group.
    :param test_group: A Pandas Series containing the values for the test group.
    :return: The glass delta between the two groups as a float.
    :rtype: float
    """
    return (np.mean(test_group) - np.mean(control_group)) / np.std(control_group, ddof=1)


def get_group_sizes(vector_df, group_index_levels=None, ignore_index_levels=None):
    """
    Get the number of elements in each group. A group is defined based on some of the index levels. Index levels can
    also be discarded before counting. That could be useful if you have a dataframe containing samples for a set of
    cohorts, days, treatments, animals, and timestamps, but you only care about the number of animals per treatment and
    day, not counting each timestamp as a separate sample in this context.

    :param vector_df: Pandas dataframe with a MultiIndex on the index and each row representing one sample's features.
    :type vector_df: pd.DataFrame
    :param group_index_levels: Index levels to use for grouping. If None, all index levels are used.
    :type group_index_levels: list
    :param ignore_index_levels: Index levels to ignore when counting. If None, no index levels are ignored.
    :type ignore_index_levels: list
    :return: Group sizes.
    :rtype: pd.Series
    """
    group_index_levels = vector_df.index.names if group_index_levels is None else group_index_levels
    ignore_index_levels = [] if ignore_index_levels is None else ignore_index_levels

    idx_frame = vector_df.index.to_frame().reset_index(drop=True)
    idx_frame = idx_frame.drop(ignore_index_levels, axis=1).drop_duplicates(keep="first")

    return idx_frame.value_counts(subset=group_index_levels).sort_index()


def get_group_aggregate(vector_df, group_index_levels=None, group_aggregate_func="median"):
    """
    Get the aggregate of a vector dataframe grouped by index levels.

    :param vector_df: Vector dataframe.
    :type vector_df: pd.DataFrame
    :param group_index_levels: Index levels to group by. If None, all index levels are grouped by.
    :type group_index_levels: list
    :param group_aggregate_func: Function to aggregate by, default is "median". See pandas.DataFrame.agg for more
    details and options.
    :type group_aggregate_func: str
    :return: Grouped aggregate.
    :rtype: pd.DataFrame
    """
    if group_index_levels is None:
        group_index_levels = vector_df.index.names
    return vector_df.groupby(group_index_levels).agg(group_aggregate_func)


def get_group_geometric_median(vector_df, group_index_levels=None):
    """
    Get the geometric median of a vector dataframe grouped by index levels.

    :param vector_df: A Vector dataframe.
    :type vector_df: pd.DataFrame
    :param group_index_levels: Index levels to group by. If None, all index levels are grouped by.
    :type group_index_levels: list
    :return: Grouped geometric median.
    :rtype: pd.DataFrame
    """
    return get_group_aggregate(vector_df, group_index_levels=group_index_levels, group_aggregate_func="median")


def get_pairwise_distance(vector_df1, vector_df2, distance_metric="cosine"):
    pairwise_distance_array = pairwise_distances(vector_df1, vector_df2, metric=distance_metric)
    pairwise_distance_df = pd.DataFrame(pairwise_distance_array, index=vector_df1.index, columns=vector_df2.index)
    return pairwise_distance_df


def get_silhouette_score_per_group(vector_df, group_index_levels=None, distance_metric="cosine"):
    if group_index_levels is None:
        group_index_levels = vector_df.columns.names

    X = vector_df
    y = vector_df.index.to_frame()[group_index_levels].apply(lambda x: str(tuple(x)), axis=1)

    silhouette_array = silhouette_samples(X, labels=y, metric=distance_metric)
    silhouette_df = pd.DataFrame({"group": y, "silhouette_score": silhouette_array}, index=X.index)
    return silhouette_df


def get_aggregated_group_distance(pairwise_distance_df, group_index_levels, group_aggregate_func="median"):
    if group_index_levels is None:
        group_index_levels = pairwise_distance_df.columns.names
    return pairwise_distance_df.apply(lambda x: x.groupby(group_index_levels).agg(group_aggregate_func), axis=1)


def get_closest_centroid_df(vector_df, group_index_levels=None):
    group_centroids = get_group_geometric_median(vector_df, group_index_levels=group_index_levels)
    centroid_distance_df = get_pairwise_distance(vector_df, group_centroids)

    closest_centroids = centroid_distance_df.idxmin(axis=1).rename("closest_centroid")
    expected_centroids = centroid_distance_df.index.to_frame().apply(
        lambda x: tuple([x[il] for il in group_index_levels]), axis=1).rename("expected_centroid")
    return closest_centroids.to_frame().join(expected_centroids)


def get_closest_centroid_confusion_matrix(vector_df, group_index_levels=None):
    centroid_confusion_df = get_closest_centroid_df(vector_df, group_index_levels=group_index_levels)
    centroid_confusion_df = centroid_confusion_df.value_counts(normalize=False, sort=False)
    centroid_confusion_df = centroid_confusion_df.unstack("expected_centroid", fill_value=0)
    centroid_confusion_df.index = pd.MultiIndex.from_tuples(centroid_confusion_df.index, names=group_index_levels)
    centroid_confusion_df.columns = pd.MultiIndex.from_tuples(centroid_confusion_df.columns, names=group_index_levels)
    return centroid_confusion_df.sort_index(axis=0).sort_index(axis=1)
