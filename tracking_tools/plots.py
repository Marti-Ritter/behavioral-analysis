import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from ..frame_pipeline.matplotlib_funcs import create_canvas, save_matplotlib_animation


def plot_keypoints(keypoint_df, ax=None, x="x", y="y", **plot_kwargs):
    """
    A function to plot keypoints from a dataframe. The dataframe is expected to have a simple index containing the keypoint names.
    The default assumption is that the dataframe has columns "x" and "y" for the x and y coordinates of the keypoints.
    As the plotting is done through the pandas plot method, additional plot_kwargs can be passed to customize the plot, potentially
    based on the columns of the dataframe.

    :param keypoint_df: A dataframe containing the keypoints to plot.
    :type keypoint_df: pd.DataFrame
    :param ax: The axis to plot on. If None, a new figure is created.
    :type ax: matplotlib.axes.Axes, optional
    :param plot_kwargs: Additional keyword arguments to pass to the plot method.
    :type plot_kwargs: dict
    :return: The axis the keypoints were plotted on.
    :rtype: matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    default_plot_kwargs = dict(x=x, y=y, kind="scatter", legend=False)
    extracted_plot_kwargs = {k: keypoint_df[v] for k, v in plot_kwargs.items() if isinstance(v, str) and v in keypoint_df.columns}
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs, **extracted_plot_kwargs}
    keypoint_df.plot(ax=ax, **plot_kwargs)
    return ax


def plot_keypoint_labels(keypoint_df, ax=None, **annotation_kwargs):
    """
    Plot labels for keypoints on a plot. The default assumption is that the dataframe has columns "x" and "y" for the x and y
    coordinates of the keypoints.
    Plotting is done through the matplotlib annotate method, so additional annotation_kwargs can be passed to customize the annotations.

    :param keypoint_df: A dataframe containing the keypoints to plot.
    :type keypoint_df: pd.DataFrame
    :param ax: The axis to plot on. If None, a new figure is created.
    :type ax: matplotlib.axes.Axes, optional
    :param annotation_kwargs: Additional keyword arguments to pass to the annotate method.
    :type annotation_kwargs: dict
    :return: The axis the keypoints were plotted on.
    :rtype: matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    default_annotation_kwargs = dict(x="x", y="y", xytext=(5, 5), textcoords="offset points")
    annotation_kwargs = {**default_annotation_kwargs, **annotation_kwargs}
    x_col, y_col = annotation_kwargs.pop("x"), annotation_kwargs.pop("y")
    for kp, row in keypoint_df.iterrows():
        ax.annotate(kp, (row[x_col], row[y_col]), **annotation_kwargs)


def plot_keypoint_skeleton(keypoint_df, skeleton_df, ax=None, x="x", y="y", **plot_kwargs):
    """
    Plots a skeleton based on the keypoints in the keypoint dataframe. The skeleton dataframe is expected to have a simple index
    containing the edge names or indices, and two columns indicating the start and end keypoint of the edge.
    Plotting happens through the pandas plot method, so additional plot_kwargs can be passed to customize the plot.
    The plot_kwargs can also be based on the columns of the skeleton dataframe.

    :param keypoint_df: A dataframe containing the keypoints to plot.
    :type keypoint_df: pd.DataFrame
    :param skeleton_df: A dataframe containing the skeleton edges to plot.
    :type skeleton_df: pd.DataFrame
    :param ax: The axis to plot on. If None, a new figure is created.
    :type ax: matplotlib.axes.Axes, optional
    :param plot_kwargs: Additional keyword arguments to pass to the plot method.
    :type plot_kwargs: dict
    """
    if ax is None:
        _, ax = plt.subplots()

    default_plot_kwargs = dict(x=x, y=y, kind="line", legend=False, color="black")

    edge_feature_cols = [col for col in skeleton_df.columns if not col.startswith("node_")]
    edge_feature_df = skeleton_df[edge_feature_cols]

    skeleton_df = skeleton_df[[col for col in skeleton_df.columns if col.startswith("node_")]]
    skeleton_keypoint_df = skeleton_df.stack("edge_feature").rename("keypoint_name")
    skeleton_keypoint_df = pd.merge(skeleton_keypoint_df, keypoint_df, left_on="keypoint_name", how="left", right_index=True)
    skeleton_keypoint_df = skeleton_keypoint_df.join(edge_feature_df, on="edge_index", lsuffix="_keypoint")

    for edge_index, edge_data in skeleton_keypoint_df.groupby(level="edge_index"):
        edge_features = edge_feature_df.loc[edge_index].to_dict()
        edge_plot_kwargs = {k: edge_features[v] for k, v in plot_kwargs.items() if isinstance(v, str) and v in edge_features}
        edge_plot_kwargs = {**default_plot_kwargs, **plot_kwargs, **edge_plot_kwargs}
        edge_data.plot(ax=ax, **edge_plot_kwargs)
    return ax


def plot_keypoint_instance(keypoint_df, skeleton_df=None, plot_labels=False, ax=None, keypoint_kwargs=None,
                           skeleton_kwargs=None, label_kwargs=None, **shared_kwargs):
    """
    A function to plot all details of a single instance of keypoints, potentially with a skeleton overlay, and optionally with labels.
    See the documentation of plot_keypoints, plot_keypoint_skeleton, and plot_keypoint_labels for more details on the individual plotting functions.
    The skeleton will be plotted first, followed by the keypoints, and then the labels if requested. This order can be manipulated by passing
    the zorder parameter in the keypoint_kwargs, skeleton_kwargs, and label_kwargs dictionaries.

    :param keypoint_df: A dataframe containing the keypoints to plot.
    :type keypoint_df: pd.DataFrame
    :param skeleton_df: A dataframe containing the skeleton edges to plot. If None, no skeleton is plotted.
    :type skeleton_df: pd.DataFrame, optional
    :param plot_labels: Whether to plot labels for the keypoints. Default is False.
    :type plot_labels: bool
    :param ax: The axis to plot on. If None, a new figure is created.
    :type ax: matplotlib.axes.Axes, optional
    :param keypoint_kwargs: Additional keyword arguments to pass to the plot_keypoints function.
    :type keypoint_kwargs: dict, optional
    :param skeleton_kwargs: Additional keyword arguments to pass to the plot_keypoint_skeleton function.
    :type skeleton_kwargs: dict, optional
    :param label_kwargs: Additional keyword arguments to pass to the plot_keypoint_labels function.
    :type label_kwargs: dict, optional
    :return: The axis the keypoints were plotted on.
    :rtype: matplotlib.axes.Axes
    """

    if ax is None:
        _, ax = plt.subplots()

    if keypoint_df.empty:
        return ax

    if keypoint_kwargs is None:
        keypoint_kwargs = {}
    if skeleton_kwargs is None:
        skeleton_kwargs = {}
    if label_kwargs is None:
        label_kwargs = {}

    if skeleton_df is not None:
        plot_keypoint_skeleton(keypoint_df, skeleton_df, ax=ax, **skeleton_kwargs, **shared_kwargs)
    plot_keypoints(keypoint_df, ax=ax, **keypoint_kwargs, **shared_kwargs)
    if plot_labels:
        plot_keypoint_labels(keypoint_df, ax=ax, **label_kwargs, **shared_kwargs)
    return ax


def plot_keypoint_instances(multi_instance_keypoint_df, ax=None, *args, **kwargs):
    if ax is None:
        _, ax = plt.subplots()

    instance_identifier_levels = multi_instance_keypoint_df.index.names[:-1]
    for identifier, single_keypoint_df in multi_instance_keypoint_df.groupby(instance_identifier_levels):
        single_keypoint_df = single_keypoint_df.droplevel(instance_identifier_levels, axis=0)

        plot_keypoint_instance(single_keypoint_df, ax=ax, *args, **kwargs)
    return ax


def normalize_keypoint_df_for_plotting(keypoint_df, target_size=256, cols_to_normalize=None):
    """
    Fit the keypoints of a dataframe to a target size. This can be useful for plotting keypoints inside a fixed-size image.
    The keypoints are normalized to the range [0, target_size] and centered in the image.
    A multi-indexed dataframe can also be supplied, in which case the normalization is applied to all levels of the index.
    This means that a multi-indexed dataframe containing multiple instances of keypoints can be normalized to the same target size.
    Supplying cols_to_normalize allows for a subset of the columns to be normalized. If None, all columns are normalized.

    :param keypoint_df: A dataframe containing the keypoints to normalize.
    :type keypoint_df: pd.DataFrame
    :param target_size: The size to normalize the keypoints to. Default is 256.
    :type target_size: int
    :param cols_to_normalize: The columns to normalize. If None, all columns are normalized.
    :type cols_to_normalize: list, optional
    :return: The normalized keypoint dataframe.
    :rtype: pd.DataFrame
    """
    if cols_to_normalize is None:
        cols_to_normalize = keypoint_df.columns
    keypoint_df = keypoint_df.copy()  # avoid modifying the original dataframe
    # shift the minimum to 0 (move all keypoints to the positive quadrant, in matplotlib bottom right)
    keypoint_df[cols_to_normalize] -= keypoint_df[cols_to_normalize].min(axis=0)
    keypoint_df[cols_to_normalize] /= keypoint_df[cols_to_normalize].max(axis=None)  # normalize to [0, 1]
    keypoint_df[cols_to_normalize] *= target_size  # scale to target size
    keypoint_df[cols_to_normalize] += (target_size - keypoint_df[cols_to_normalize].max(axis=0)) / 2  # center in the image, along the x-axis
    return keypoint_df


def plot_single_trajectory_overview(single_trajectory_df, skeleton_df=None, ax=None, **plot_kwargs):
    """
    Plot the standard overview for a single trajectory. The trajectory is expected to be a dataframe with a simple
    index containing the frame index, and columns "x" and "y" for the x and y coordinates of the keypoints. The skeleton
    is expected to be a dataframe with a simple index containing the edge names or indices, and two columns indicating
    the start and end keypoint of the edge.
    The plot will contain the given keypoints, potentially with a skeleton overlay. The first and last instance of the
    trajectory will be plotted with full opacity, while the instances in between will be plotted with reduced opacity.

    :param single_trajectory_df: The dataframe containing the trajectory to plot.
    :type single_trajectory_df: pd.DataFrame
    :param skeleton_df: The dataframe containing the skeleton edges to plot. If None, no skeleton is plotted.
    :type skeleton_df: pd.DataFrame, optional
    :param ax: The axis to plot on. If None, a new figure is created.
    :type ax: matplotlib.axes.Axes, optional
    :return: The axis the keypoints were plotted on.
    :rtype: matplotlib.axes.Axes
    """
    # matplotlib get coolwarm as cmap
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    if ax is None:
        _, ax = plt.subplots()

    keypoint_df = single_trajectory_df.stack("keypoint_name", future_stack=True)
    normed_kp = normalize_keypoint_df_for_plotting(keypoint_df, target_size=256)

    frames = sorted(keypoint_df.index.get_level_values("frame_index").unique())
    alpha_series = pd.Series([1] + [0.3] * (len(frames) - 2) + [1], index=frames)
    zorder_series = pd.Series(range(len(frames)), index=frames)
    zorder_series.iloc[0] = len(frames)-1
    color_series = pd.Series([cmap(i) for i in np.linspace(0, 1, len(frames))], index=frames)

    for frame in frames:
        plot_keypoint_instance(normed_kp.loc[frame], skeleton_df=skeleton_df, ax=ax, alpha=alpha_series[frame],
                               color=color_series[frame][:3], zorder=zorder_series[frame], **plot_kwargs)

    return ax


def create_track_gif(track_df, output_path, skeleton_df=None, title_text=None, clear_plots=True, overwrite=False,
                     target_size=256, plot_kwargs=None, save_kwargs=None, **animation_kwargs):
    """
    Create a gif of a single (syllable) trajectory. The trajectory is expected to be a "stacked keypoint" dataframe, where the first level
    of the index is the frame index, and the second level is the keypoint name. The dataframe is expected to have columns "x" and "y" for
    the x and y coordinates of the keypoints. The skeleton is expected to be a dataframe with a simple index containing the edge names or
    indices, and two columns indicating the start and end keypoint of the edge.

    :param single_trajectory_df: The dataframe containing the trajectory to plot.
    :type single_trajectory_df: pd.DataFrame
    :param output_path: The path to save the gif to.
    :type output_path: str
    :param skeleton_df: The dataframe containing the skeleton edges to plot. If None, no skeleton is plotted.
    :type skeleton_df: pd.DataFrame, optional
    :param title_text: The text to display as a title on the plot.
    :type title_text: str, optional
    :param clear_plots: Whether to clear every plot before the next one is plotted. Default is True.
    :type clear_plots: bool
    :param overwrite: Whether to overwrite the output file if it already exists. Default is False.
    :type overwrite: bool
    :param target_size: The size of the target canvas in pixels. Default is 256.
    :type target_size: int
    :param plot_kwargs: Additional keyword arguments to pass to the plot_keypoint_instance function. Can also be a callable that returns a 
    dictionary when called with the frame_index.
    :type plot_kwargs: dict or callable
    :param save_kwargs: Additional keyword arguments to pass to the save_matplotlib_animation function.
    :type save_kwargs: dict, optional
    :param animation_kwargs: Additional keyword arguments to pass to the animation function.
    :type animation_kwargs: dict
    """
    keypoint_df = track_df.stack("keypoint_name", future_stack=True)
    normalized_keypoint_df = normalize_keypoint_df_for_plotting(keypoint_df, target_size=target_size)
    save_kwargs = {} if save_kwargs is None else save_kwargs

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if not overwrite and os.path.exists(output_path):
        return

    canvas = create_canvas(target_size, target_size, dpi=100, alpha=1)
    ax = canvas.gca()

    # expected level of "frame_index" to be -2
    frames = sorted(normalized_keypoint_df.index.get_level_values("frame_index").unique())
    identifier_names = normalized_keypoint_df.index.names[:-2]
    identifier_count = len(identifier_names)

    if clear_plots:
        original_limits = ax.axis()

    def _animate(i):
        if clear_plots:
            ax.cla()
            ax.axis(original_limits)

        if plot_kwargs is None:
            additional_plot_kwargs = {}
        elif callable(plot_kwargs):
            additional_plot_kwargs = plot_kwargs(i)
        else:
            additional_plot_kwargs = plot_kwargs

        if identifier_count > 0:
            for _identifier, single_track in normalized_keypoint_df.groupby(identifier_names):
                single_track = single_track.droplevel(identifier_names)
                plot_keypoint_instance(single_track.loc[i], ax=ax, skeleton_df=skeleton_df, **additional_plot_kwargs)
        else:
            plot_keypoint_instance(normalized_keypoint_df.loc[i], ax=ax, skeleton_df=skeleton_df, **additional_plot_kwargs)

        if title_text is not None:
            ax.text(0.5, 0.95, title_text, horizontalalignment="center", verticalalignment="center",
                    transform=ax.transAxes, fontsize=12, fontweight="bold")

        return tuple()
    
    default_animation_kwargs = dict(frames=frames)
    updated_animation_kwargs = {**default_animation_kwargs, **animation_kwargs}

    save_matplotlib_animation(canvas, _animate, output_path, save_kwargs=save_kwargs, **updated_animation_kwargs)
    plt.close(canvas)
