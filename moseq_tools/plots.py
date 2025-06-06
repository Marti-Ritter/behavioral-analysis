import os
import warnings
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

from frame_pipeline.matplotlib_funcs import create_canvas, save_matplotlib_animation
from ..tracking_tools.plots import normalize_keypoint_df_for_plotting, plot_keypoint_instance

from .syllable_funcs import normalize_median_trajectories_for_plotting
from ..pandas_tools.funcs import pd_reindex_to_index_union, pd_flatten_indices
from ..visualization.networkx_funcs import plot_connections_from_pair_dict
from frame_pipeline.pil_funcs import stitch_animated_gif_list, ensure_pil_image


def _pd_flatten_unionize_sort_indices(input_pd_list):
    output_pd_list = pd_reindex_to_index_union([pd_flatten_indices(input_pd) for input_pd in input_pd_list])
    try:
        output_pd_list = [input_pd.sort_index(axis=0).sort_index(axis=1) if isinstance(input_pd, pd.DataFrame) else
                          input_pd.sort_index() for input_pd in output_pd_list]
    except TypeError:
        warnings.warn("Sorting indices failed. Continuing with unsorted pandas objects.")
    return output_pd_list


def _get_syllable_name_placeholder_dict(group_transition_frames_dict):
    first_transition_frame = next(iter(group_transition_frames_dict.values()))
    all_syllables = first_transition_frame.index.union(first_transition_frame.columns)
    return {ix: str(ix) for ix in all_syllables}


def visualize_transition_bigram(
        group_transition_frames_dict,
        syllable_name_dict=None,
        normalize="bigram",
        **fig_kwargs
):
    group_transition_frames_dict = dict(zip(group_transition_frames_dict.keys(),
                                            _pd_flatten_unionize_sort_indices(group_transition_frames_dict.values())))
    if syllable_name_dict is None:
        syllable_name_dict = _get_syllable_name_placeholder_dict(group_transition_frames_dict)

    n_col = 2 if len(group_transition_frames_dict) > 1 else 1
    n_row = int(np.ceil(len(group_transition_frames_dict) / n_col))

    default_fig_kwargs = dict(
        figsize=(12 if len(group_transition_frames_dict) != 1 else 6, 6 * n_row),
        sharex=False,
        sharey=True
    )
    fig_kwargs = {**default_fig_kwargs, **fig_kwargs}
    fig, ax = plt.subplots(n_row, n_col, **fig_kwargs)

    title_map = dict(bigram="Bigram", columns="Incoming", rows="Outgoing")
    color_lim = max([df.max(axis=None) for df in group_transition_frames_dict.values()])
    if len(group_transition_frames_dict) == 1:
        axs = [ax]
    else:
        axs = ax.flat
    for i, (group, trans_df) in enumerate(group_transition_frames_dict.items()):
        h = axs[i].imshow(trans_df, cmap="cubehelix", vmax=color_lim,)
        if i == 0:
            axs[i].set_ylabel("Incoming element/syllable")
            ynames = [syllable_name_dict[ix] for ix in trans_df.index]
            plt.yticks(range(len(ynames)), ynames)
        cb = fig.colorbar(h, ax=axs[i], fraction=0.046, pad=0.04)
        cb.set_label(f"{title_map[normalize]} transition probability")
        axs[i].set_xlabel("Outgoing element/syllable")
        axs[i].set_title(group)
        xnames = [syllable_name_dict[ix] for ix in trans_df.columns]
        axs[i].set_xticks(range(len(xnames)), xnames, rotation=90)

    return fig


def visualize_transition_difference_bigram(
        group_transition_frames_dict,
        syllable_name_dict=None,
        normalize="bigram",
        **fig_kwargs
):
    assert len(group_transition_frames_dict) > 1, "At least two groups are required for comparison."

    group_transition_frames_dict = dict(zip(group_transition_frames_dict.keys(),
                                            _pd_flatten_unionize_sort_indices(group_transition_frames_dict.values())))
    if syllable_name_dict is None:
        syllable_name_dict = _get_syllable_name_placeholder_dict(group_transition_frames_dict)

    # find combinations
    group_combinations = list(combinations(group_transition_frames_dict.keys(), 2))

    n_col = 2 if len(group_combinations) > 1 else 1
    n_row = int(np.ceil(len(group_combinations) / n_col))

    default_fig_kwargs = dict(
        figsize=(12 if len(group_combinations) != 1 else 6, 6 * n_row),
        sharex=False,
        sharey=True
    )
    fig_kwargs = {**default_fig_kwargs, **fig_kwargs}
    fig, ax = plt.subplots(n_row, n_col, **fig_kwargs)

    title_map = dict(bigram="Bigram", columns="Incoming", rows="Outgoing")
    if len(group_combinations) == 1:
        axs = [ax]
    else:
        axs = ax.flat
    for i, pair in enumerate(group_combinations):
        left_df = group_transition_frames_dict[pair[0]]
        right_df = group_transition_frames_dict[pair[1]]
        diff_df = left_df - right_df
        color_lim = max([diff_df.max(axis=None), -diff_df.min(axis=None)])
        h = axs[i].imshow(diff_df.values, cmap="coolwarm", vmax=color_lim, vmin=-color_lim)
        if i % 2 == 0:
            axs[i].set_ylabel("Incoming element/syllable")
            ynames = [syllable_name_dict[ix] for ix in diff_df.index]
            plt.yticks(range(len(ynames)), ynames)
        cb = fig.colorbar(h, ax=axs[i], fraction=0.046, pad=0.04)
        cb.set_label(f"{title_map[normalize]} transition probability difference")
        axs[i].set_xlabel("Outgoing element/syllable")
        axs[i].set_title(f"{pair[0]} - {pair[1]}")
        xnames = [syllable_name_dict[ix] for ix in diff_df.columns]
        axs[i].set_xticks(range(len(xnames)), xnames, rotation=90)

    return fig


def plot_frequencies(
        group_frequencies_dict,
        syllable_name_dict=None,
        ax=None,
        **plot_kwargs
):
    group_frequencies_dict = dict(zip(group_frequencies_dict.keys(),
                                      _pd_flatten_unionize_sort_indices(group_frequencies_dict.values()))
                                  )
    group_usages_series = pd.concat(group_frequencies_dict)
    group_usages_series.index.names = ["group", "syllable"]
    group_usages_series.name = "frequency"
    group_usages_df = group_usages_series.reset_index()

    if syllable_name_dict is None:
        all_syllables = group_usages_df["syllable"].unique()
        syllable_name_dict = {ix: str(ix) for ix in all_syllables}

    group_usages_series = pd.concat(group_frequencies_dict)
    group_usages_series.index.names = ["group", "syllable"]
    group_usages_series.name = "frequency"
    group_usages_df = group_usages_series.reset_index()

    default_plot_kwargs = dict(dodge=True, ci="sd", estimator=np.mean)
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    ax = sns.pointplot(group_usages_df, x="syllable", y="frequency", hue="group", ax=ax, **plot_kwargs)
    ax.set_xticklabels([syllable_name_dict[ix] for ix in group_usages_df["syllable"].unique()], rotation=90)
    return ax


def plot_transition_graph_group(
        group_transition_frames_dict,
        group_frequencies_dict=None,
        syllable_name_dict=None,
        node_scaling=2000,
        edge_scaling=100,
):
    group_transition_frames_dict = dict(zip(group_transition_frames_dict.keys(),
                                            _pd_flatten_unionize_sort_indices(group_transition_frames_dict.values())))
    if syllable_name_dict is None:
        syllable_name_dict = _get_syllable_name_placeholder_dict(group_transition_frames_dict)

    n_col = 2 if len(group_transition_frames_dict) > 1 else 1
    n_row = int(np.ceil(len(group_transition_frames_dict) / n_col))

    fig, all_axes = plt.subplots(n_row, n_col, figsize=(20, 10 * n_row))
    ax = all_axes.flat

    for i, (group, trans_df) in enumerate(group_transition_frames_dict.items()):
        transition_dict = trans_df.stack().to_dict()
        contained_nodes = trans_df.index.union(trans_df.columns).values

        if group_frequencies_dict is not None:
            reindexed_frequencies = group_frequencies_dict[group].reindex(contained_nodes, fill_value=0)
            node_sizes = reindexed_frequencies.values * node_scaling
        else:
            node_sizes = None

        _ = plot_connections_from_pair_dict(transition_dict,
                                            node_labels=syllable_name_dict,
                                            node_sizes=node_sizes,
                                            node_edge_color="k" if group_frequencies_dict is not None else "none",
                                            ax=ax[i], directed=True,
                                            positions=nx.circular_layout,
                                            outcome_widths=lambda x: np.abs(x) * edge_scaling,
                                            outcome_colors=lambda x: "k" if x != 0 else "none",
                                            outcome_labels=lambda x: np.round(x, 4) if x != 0 else None
                                            )
        ax[i].set_title(group)
    # turn off the axis spines
    for sub_ax in ax:
        sub_ax.axis("off")

    return fig


def _get_difference_graph_legend_elements(usage_diff_legend=True):
    return_labels = [
        Line2D([0], [0], color="r", lw=2, label=f"Up-regulated transition"),
        Line2D([0], [0], color="b", lw=2, label=f"Down-regulated transition")
    ]
    if usage_diff_legend:
        return_labels += [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Up-regulated usage",
                markerfacecolor="w",
                markeredgecolor="r",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Down-regulated usage",
                markerfacecolor="w",
                markeredgecolor="b",
                markersize=10,
            ),
        ]
    return return_labels


def plot_transition_graph_difference(
        group_transition_frames_dict,
        group_frequencies_dict=None,
        syllable_name_dict=None,
        node_scaling=2000,
        edge_scaling=100,
        node_color_func=None,
        edge_color_func=None,
        outcome_label_func=None,
        outcome_width_func=None,
        single_ax_size=(10, 10),
        **plot_kwargs
):
    assert len(group_transition_frames_dict) > 1, "At least two groups are required for comparison."

    group_transition_frames_dict = dict(zip(group_transition_frames_dict.keys(),
                                            _pd_flatten_unionize_sort_indices(group_transition_frames_dict.values()))
                                        )
    if syllable_name_dict is None:
        syllable_name_dict = _get_syllable_name_placeholder_dict(group_transition_frames_dict)

    # find combinations
    group_combinations = list(combinations(group_transition_frames_dict, 2))

    n_col = 2 if len(group_combinations) > 1 else 1
    n_row = int(np.ceil(len(group_combinations) / n_col))

    single_ax_width, single_ax_height = single_ax_size
    fig, all_axes = plt.subplots(n_row, n_col, figsize=(single_ax_width * n_col, single_ax_height * n_row))
    ax = all_axes.flat if len(group_combinations) > 1 else [all_axes]

    def _default_color_func(value):
        if value > 0:
            return "red"
        elif value < 0:
            return "blue"
        else:
            return "none"

    def _default_outcome_label_func(value):
        if value != 0:
            return np.round(value, 4)
        else:
            return None

    def _default_outcome_width_func(value):
        return np.abs(value) * edge_scaling

    if node_color_func is None:
        node_color_func = _default_color_func
    if edge_color_func is None:
        edge_color_func = _default_color_func
    if outcome_label_func is None:
        outcome_label_func = _default_outcome_label_func
    if outcome_width_func is None:
        outcome_width_func = _default_outcome_width_func

    for i, pair in enumerate(group_combinations):
        # left tm minus right tm
        diff_df = (group_transition_frames_dict[pair[0]] - group_transition_frames_dict[pair[1]])
        diff_dict = diff_df.stack().to_dict()

        contained_nodes = diff_df.index.union(diff_df.columns).values

        if group_frequencies_dict is not None:
            usages_diff = group_frequencies_dict[pair[0]] - group_frequencies_dict[pair[1]]
            normalized_usg_diff = (usages_diff / usages_diff.abs().sum())
            reindexed_normalized_usg_diff = normalized_usg_diff.reindex(contained_nodes, fill_value=0)
            node_sizes = reindexed_normalized_usg_diff.abs().values * node_scaling
            node_edgecolors = reindexed_normalized_usg_diff.apply(lambda x: node_color_func(x))
        else:
            node_sizes = None
            node_edgecolors = "none"

        default_plot_kwargs = dict(
            node_labels=syllable_name_dict,
            node_edgecolor=node_edgecolors,
            node_sizes=node_sizes,
            ax=ax[i],
            directed=True,
            positions=nx.circular_layout,
            outcome_widths=outcome_width_func,
            outcome_colors=edge_color_func,
            outcome_labels=outcome_label_func,
        )
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        plot_kwargs = {**default_plot_kwargs, **plot_kwargs}
        _ = plot_connections_from_pair_dict(diff_dict, **plot_kwargs)

        ax[i].set_title(pair[0] + " - " + pair[1])
    # turn off the axis spines
    for sub_ax in ax:
        sub_ax.axis("off")

    # add legend
    legend_elements = _get_difference_graph_legend_elements(usage_diff_legend=group_frequencies_dict is not None)
    plt.legend(handles=legend_elements, loc="upper left", borderaxespad=0)

    return fig


def create_single_trajectory_gif(single_trajectory_df, output_path, skeleton_df=None, title_text=None, add_trail=False, overwrite=False,
                                 target_size=256, save_kwargs=None, **animation_kwargs):
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
    :param add_trail: Whether to add a trail to the plot. Default is False.
    :type add_trail: bool
    :param overwrite: Whether to overwrite the output file if it already exists. Default is False.
    :type overwrite: bool
    :param target_size: The size of the target canvas in pixels. Default is 256.
    :type target_size: int
    :param save_kwargs: Additional keyword arguments to pass to the save_matplotlib_animation function.
    :type save_kwargs: dict, optional
    :param animation_kwargs: Additional keyword arguments to pass to the animation function.
    :type animation_kwargs: dict
    """
    normalized_trajectory_df = normalize_keypoint_df_for_plotting(single_trajectory_df, target_size=target_size)
    save_kwargs = {} if save_kwargs is None else save_kwargs

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if not overwrite and os.path.exists(output_path):
        return

    canvas = create_canvas(target_size, target_size, dpi=100, alpha=1)
    ax = canvas.gca()

    frames = sorted(normalized_trajectory_df.index.get_level_values(0).unique())

    if not add_trail:
        original_limits = ax.axis()
    else:
        alpha_series = pd.Series(np.linspace(0.1, 1, len(frames)), index=frames)

    def _animate(i):
        if not add_trail:
            ax.cla()
            ax.axis(original_limits)
            alpha = 1
        else:
            alpha = alpha_series.loc[i]

        plot_keypoint_instance(normalized_trajectory_df.loc[i], ax=ax, skeleton_df=skeleton_df, plot_labels=False,
                               keypoint_kwargs=dict(alpha=alpha), skeleton_kwargs=dict(alpha=alpha))

        if title_text is not None:
            ax.text(0.5, 0.95, title_text, horizontalalignment="center", verticalalignment="center",
                    transform=ax.transAxes, fontsize=12, fontweight="bold")

        return tuple()
    
    default_animation_kwargs = dict(frames=frames)
    updated_animation_kwargs = {**default_animation_kwargs, **animation_kwargs}

    save_matplotlib_animation(canvas, _animate, output_path, save_kwargs=save_kwargs, **updated_animation_kwargs)
    plt.close(canvas)


def _mangle_dataset(input_pd, spread=3, random_state=42):
    return input_pd * np.random.RandomState(random_state).rand(*input_pd.shape)*2*spread-spread


def _create_mangled_dataset(trans_frames_dict, usages_dict, random_state=42):
    rdm = np.random.RandomState(random_state)
    new_trans_frames = {k: _mangle_dataset(t_f, random_state=rdm.randint(0, 100)) for k, t_f in
                        trans_frames_dict.items()}
    new_usages = {k: _mangle_dataset(u, random_state=rdm.randint(0, 100)) for k, u in usages_dict.items()}
    return new_trans_frames, new_usages


def create_syllable_trajectory_gifs(median_trajectory_df, output_dir, overwrite=False, target_size=256,
                                    create_combined=True, save_kwargs=None, **animation_kwargs):
    normalized_trajectory_df = normalize_median_trajectories_for_plotting(median_trajectory_df, target_size=target_size)
    out_path_list = []
    for syllable, syllable_df in tqdm(normalized_trajectory_df.groupby("syllable")):
        out_path = os.path.join(output_dir, f"Syllable{syllable}.gif")
        out_path_list.append(out_path)

        create_single_trajectory_gif(syllable_df, out_path, title_text=f"Syllable {syllable}", overwrite=overwrite,
                                     target_size=target_size, save_kwargs=save_kwargs, **animation_kwargs)

        if not overwrite and os.path.exists(out_path):
            continue

    if create_combined and len(out_path_list) > 1:
        out_path = os.path.join(output_dir, "AllSyllables.gif")
        if not overwrite and os.path.exists(out_path):
            return

        pil_gifs = [ensure_pil_image(out_path) for out_path in out_path_list]
        stitched_gifs = stitch_animated_gif_list(*pil_gifs, columns=4)
        fps = animation_kwargs.get("fps", 30)
        stitched_gifs[0].save(out_path, save_all=True, append_images=stitched_gifs[1:], duration=int(1000 / fps),
                              loop=0)


def create_similarity_dendrogram(distance_df, ax=None):
    Z = linkage(squareform(distance_df.values), "complete")

    if ax is None:
        fig, ax = plt.subplots()

    dendrogram(Z, labels=[f"Syllable {x}" for x in distance_df.index], leaf_font_size=10, ax=ax, leaf_rotation=90)

    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("lightgray")
    ax.set_title("Syllable similarity")
    return ax
