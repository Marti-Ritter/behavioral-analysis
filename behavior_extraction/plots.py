"""
Plots related to behavioral analysis.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


def plot_grouped_heatmap(heatmap_df, ax=None, normalize_column_sum=False, annotate=True, group_index_levels=None,
                         ignore_groups=None, **kwargs):
    if group_index_levels is None:
        group_index_levels = heatmap_df.index.names
    ignore_groups = [] if ignore_groups is None else ignore_groups

    if normalize_column_sum:
        heatmap_df /= heatmap_df.sum(axis=0)
        annot_df = heatmap_df.map(
            lambda x: (str(round(x, 2))[::-1].zfill(4)[-2::-1] if 0 < x < 1 else ("1" if x == 1 else "")))
    else:
        annot_df = heatmap_df.map(lambda x: "" if x == 0 else x)

    if ax is None:
        _, ax = plt.subplots()

    default_kwargs = dict(square=True, cbar=True)
    kwargs = {**default_kwargs, **kwargs}

    if kwargs["cbar"]:
        divider = make_axes_locatable(ax)
        kwargs["cbar_ax"] = divider.append_axes("right", size="5%", pad=0.05)

    ax = sns.heatmap(heatmap_df, annot=annot_df if annotate else annotate, ax=ax, fmt="", **kwargs)
    group_widths = heatmap_df.groupby(group_index_levels).size()
    group_starts, group_ends = group_widths.cumsum().shift(1), group_widths.cumsum()

    for group, start in group_starts.items():
        if group in ignore_groups:
            continue
        ax.axvline(start, color="white", ls="--")
        ax.axvline(group_ends.loc[group], color="white", ls="--")
        ax.axhline(start, color="white", ls="--")
        ax.axhline(group_ends.loc[group], color="white", ls="--")
    return ax
