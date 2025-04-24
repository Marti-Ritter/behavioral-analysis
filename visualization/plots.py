from string import Template

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
from shapely.plotting import patch_from_polygon

from .matplotlib_funcs import create_canvas


def plot_polygon(polygon, name=None, ax=None, polygon_kwargs=None, plot_label=True, label_kwargs=None):
    """
    Plot a Shapely polygon onto a matplotlib.Axes

    :param ax: Axes that the polygon will be plotted to
    :type ax: plt.Axes
    :param name: Name of the polygon
    :type name: str
    :param polygon: Shapely polygon to plot
    :type polygon: shapely.geometry.Polygon
    :param polygon_kwargs: kwargs for the polygon outline, see matplotlib.lines.Line2D for more details
    :param plot_label: whether to plot the key/label onto the polygon
    :type plot_label: bool
    :param label_kwargs: kwargs for the text label, see matplotlib.text.Text for more details
    :type label_kwargs: dict
    :return: List of artists that were plotted
    :rtype: list
    """

    name = "" if name is None else name
    ax = plt.subplots()[1] if ax is None else ax

    default_polygon_kwargs = dict(color="green", alpha=1., linewidth=1., fill=False)
    polygon_kwargs = {**default_polygon_kwargs,
                      **polygon_kwargs} if polygon_kwargs is not None else default_polygon_kwargs

    plotted_artists = list()
    plotted_artists.append(ax.add_patch(patch_from_polygon(polygon, **polygon_kwargs)))

    if plot_label:
        default_label_kwargs = dict(ha="center", va="center", size=12, weight="bold",
                                    bbox=dict(facecolor='white', edgecolor='none'))
        label_kwargs = {**default_label_kwargs, **label_kwargs} if label_kwargs is not None else default_label_kwargs
        plotted_artists.append(ax.text(*polygon.centroid.coords[0], s=name, **label_kwargs))

    return plotted_artists


def plot_polygon_3d(polygon, name=None, figure=None, z=0, z_label=None, polygon_color="green", polygon_alpha=1.,
                    font_size=12, plot_label=True, line_width=1., fill=False):
    """
    Plot a Shapely polygon onto a plotly figure

    :param polygon:
    :type polygon:
    :param name:
    :type name:
    :param figure:
    :type figure:
    :param z:
    :type z: float
    :param z_label:
    :type z_label: float
    :param polygon_color:
    :type polygon_color:
    :param polygon_alpha:
    :type polygon_alpha:
    :param font_size:
    :type font_size:
    :param plot_label:
    :type plot_label:
    :param line_width:
    :type line_width:
    :param fill:
    :type fill:
    :return:
    :rtype:
    """

    name = "" if name is None else name
    figure = go.Figure() if figure is None else figure

    plotted_traces = []
    new_trace = go.Scatter3d(x=list(polygon.exterior.xy[0]), y=list(polygon.exterior.xy[1]),
                             z=[z] * len(polygon.exterior.xy[0]), mode='lines',
                             line=dict(color=polygon_color, width=line_width), opacity=polygon_alpha,
                             surfaceaxis=2 if fill else -1, showlegend=False)

    plotted_traces.append(new_trace)
    figure.add_trace(new_trace)

    if plot_label:
        z_label = z if z_label is None else z_label
        new_trace = go.Scatter3d(x=[polygon.centroid.coords[0][0]], y=[polygon.centroid.coords[0][1]], z=[z_label],
                                 mode='text', text=[name], textposition='middle center',
                                 marker=dict(color="black", size=font_size), showlegend=False)
        plotted_traces.append(new_trace)
        figure.add_trace(new_trace)

    return plotted_traces


def plot_polygons(polygon_dict, ax=None, polygon_color="green", polygon_alpha=1., line_width=1., plot_labels=True,
                  label_kwargs=None):
    """Plot Shapely polygons in a dict onto a matplotlib.Axes

    :param ax: Axes that the polygons will be plotted to
    :type ax: plt.Axes
    :param polygon_dict: Dict of polygons containing an assignment of name: polygon for arena elements
    :type polygon_dict: dict
    :param polygon_color: Color of the polygons, can be a dict of name: color or a single color for all polygons
    :type polygon_color: str or dict
    :param polygon_alpha: Opacity of the polygons can be a dict of name: alpha or a single alpha for all polygons
    :type polygon_alpha: float or dict
    :param line_width: Width of the polygon lines
    :type line_width: float
    :param plot_labels: whether to plot the keys/labels onto the polygons
    :type plot_labels: bool
    :param label_kwargs: kwargs for the text label, see matplotlib.text.Text for more details
    :type label_kwargs: dict
    """

    ax = plt.subplots()[1] if ax is None else ax

    if isinstance(polygon_color, str):
        polygon_color = {k: polygon_color for k in polygon_dict.keys()}
    if isinstance(polygon_alpha, float):
        polygon_alpha = {k: polygon_alpha for k in polygon_dict.keys()}

    default_label_kwargs = dict(ha="center", va="center", size=12, weight="bold",
                                bbox=dict(facecolor='white', edgecolor='none'))
    label_kwargs = {**default_label_kwargs, **label_kwargs} if label_kwargs is not None else default_label_kwargs

    plotted_artists = []

    def _plot_single_polygon(polygon, name, polygon_kwargs):
        plotted_artists.extend(plot_polygon(polygon, name=name, ax=ax, polygon_kwargs=polygon_kwargs,
                                            label_kwargs=label_kwargs, plot_label=plot_labels))

    for name, polygon in polygon_dict.items():
        polygon_kwargs = dict(color=polygon_color[name], alpha=polygon_alpha[name], linewidth=line_width)

        if isinstance(polygon, list):
            for p in polygon:
                _plot_single_polygon(p, name, polygon_kwargs)
        else:
            _plot_single_polygon(polygon, name, polygon_kwargs)

    return plotted_artists


def plot_polygons_3d(polygon_dict, figure=None, z=0, z_label=None, polygon_color="green", polygon_alpha=1.,
                     font_size=12, plot_labels=True, line_width=1., fill=False):
    """Plot Shapely polygons in a dict onto a plotly figure

    :param polygon_dict: Dict of polygons containing an assignment of name: polygon for arena elements
    :type polygon_dict: dict
    :param figure: Plotly figure that the polygons will be plotted to
    :type figure: plotly.graph_objects.Figure
    :param z: z-coordinate of the polygons
    :type z: float
    :param z_label: z-coordinate of the polygon labels
    :type z_label: float
    :param polygon_color: Color of the polygons
    :type polygon_color: str
    :param polygon_alpha: Opacity of the polygons
    :type polygon_alpha: float
    :param font_size: Size of the polygon labels
    :type font_size: int
    :param plot_labels: whether to plot the keys/labels onto the polygons
    :type plot_labels: bool
    :param line_width: Width of the polygon lines
    :type line_width: float
    :param fill: Whether to fill the polygons
    :type fill: bool
    :return: List of plotly traces
    :rtype: list
    """

    figure = go.Figure() if figure is None else figure

    plotted_traces = []
    for name, polygon in polygon_dict.items():
        try:
            for p in polygon:
                plotted_traces.extend(plot_polygon_3d(p, name=name, figure=figure, z=z, z_label=z_label,
                                                      polygon_color=polygon_color, polygon_alpha=polygon_alpha,
                                                      font_size=font_size, plot_label=plot_labels,
                                                      line_width=line_width, fill=fill))
        except TypeError:
            plotted_traces.extend(plot_polygon_3d(polygon, name=name, figure=figure, z=z, z_label=z_label,
                                                  polygon_color=polygon_color, polygon_alpha=polygon_alpha,
                                                  font_size=font_size, plot_label=plot_labels, line_width=line_width,
                                                  fill=fill))

    return plotted_traces


def plot_polygons_to_ax(ax, polygon_dict, **plot_kwargs):
    """Plot Shapely polygons in a dict onto a matplotlib.Axes

    :param ax: Axes that the polygons will be plotted to
    :type ax: plt.Axes
    :param polygon_dict: Dict of polygons containing an assignment of name: polygon for arena elements
    :type polygon_dict: dict
    :param plot_kwargs: kwargs for plot_polygons, see documentation there
    """

    return plot_polygons(polygon_dict, ax=ax, **plot_kwargs)


def plot_arena(polygon_dict, arena_size=None, dpi=100, polygon_color="black", polygon_alpha=1.,
               font_size=12, font_weight="bold", plot_labels=True, ax=None, **plot_kwargs):
    """Plot a dict of polygons describing an arena layout to a figure

    :param arena_size: Tuple or list of arena size in pixels
    :type arena_size: tuple or list
    :param dpi: Resolution of the canvas used to plot the arena
    :type dpi: int
    :param polygon_dict: Dict of polygons containing an assignment of name: polygon for arena elements
    :type polygon_dict: dict
    :param polygon_color: Color of the polygons
    :type polygon_color: str
    :param polygon_alpha: Opacity of the polygons
    :type polygon_alpha: float
    :param font_size: Size of the polygon labels
    :type font_size: int
    :param font_weight: Weight of the polygon labels
    :type font_weight: str
    :param plot_labels: tuple or list of labels for the polygons
    :type plot_labels: tuple or list
    :return: Figure showing the arena layout
    :rtype: plt.Figure
    """
    from shapely.ops import unary_union

    if arena_size is None:
        arena_bounds = unary_union(list(polygon_dict.values())).bounds
        arena_size = (arena_bounds[0] + arena_bounds[2], arena_bounds[1] + arena_bounds[3])

    ax = create_canvas(*arena_size, dpi).gca() if ax is None else ax
    plot_polygons_to_ax(ax, polygon_dict, polygon_color=polygon_color, polygon_alpha=polygon_alpha,
                        plot_labels=plot_labels, label_kwargs=dict(size=font_size, weight=font_weight), **plot_kwargs)
    return ax


def plot_chunks_timeline(input_chunk_df, center=0.5, height=1., alpha=0.1, color="blue", start_col="From Second",
                         end_col="To Second", ax=None):
    """Plot chunks onto a timeline, represented by slightly transparent spans. Overlaps become easily visible

    :param input_chunk_df: DataFrame containing information on the start and end of each chunk
    :type input_chunk_df: pandas.DataFrame
    :param center: y-position of the plotted spans
    :type center: float
    :param height: y-range of the plotted spans
    :type height: float
    :param alpha: opacity of the plotted spans. Keep this low to be able to see overlaps!
    :type alpha: float
    :param color: Color of the plotted spans
    :type color: str
    :param start_col: Column in the input_chunk_df that contains the starts of the spans
    :type start_col: str
    :param end_col: Column in the input_chunk_df that contains the ends of the spans
    :type end_col: str
    :param ax: Axes the spans will be plotted onto
    :type ax: matplotlib.axes.Axes or None
    :return: Axes with series of spans plotted onto
    :rtype: matplotlib.axes.Axes
    """
    local_ax = plt.subplots(figsize=(15, 5))[1] if ax is None else ax
    for _, row in input_chunk_df.iterrows():
        local_ax.axvspan(row[start_col], row[end_col], ymin=center - height / 2, ymax=center + height / 2, alpha=alpha,
                         color=color)
    return local_ax


def plot_chunks_additively_timeline(input_chunk_df, center=0.5, width=1., alpha=0.1, color="blue",
                                    start_col="From Second", end_col="To Second", ax=None):
    local_ax = plt.subplots(figsize=(15, 5))[1] if ax is None else ax
    for _, row in input_chunk_df.iterrows():
        local_ax.axvspan(row[start_col], row[end_col], ymin=center - width / 2, ymax=center + width / 2, alpha=alpha,
                         color=color)
    return local_ax


# https://stackoverflow.com/a/30536361
class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def plot_positions_to_arena(arena_fig, positions, kind="scatter"):
    ax = arena_fig.gca()
    if kind == "scatter":
        x, y = positions.T
        sns.scatterplot(x=x, y=y, ax=ax)


def get_arena_polygon_from_position(arena_polygon_dict, input_position, return_all=False, outside_behavior='return'):
    """
    Get the polygon(s) that contain(s) a given position. Can be set to only return the first one, e.g. if there are no
    intersections between polygons in the arena. Can raise an exception if a position is outside of all polygons in the
    arena if raise_outside is set to True.

    :param arena_polygon_dict: A dictionary mapping string names to shapely polygons.
    :type arena_polygon_dict: dict[str, shapely.geometry.Polygon]
    :param input_position: A single location at which to find the polygon.
    :type input_position: shapely.geometry.Point or tuple or list or numpy array
    :param return_all: If True, return all polygons that contain the input_position. If not, only the first polygon that
    contains the input_position will be returned.
    :type return_all: bool
    :param outside_behavior: Can be "return", "raise", or "closest". If "return" it will return an empty list if the
    input_position is outside of all polygons in the arena. If "raise", it will raise an exception if the input_position
    is outside of any polygon in the arena. If "closest" it will return the closest polygon to the input_position.
    :type outside_behavior: str
    :return: The polygon(s) that contain(s) the input_position. If return_all is True, a list of all polygons that
    contain the input_position will be returned. If return_all is False, a single polygon that contains the
    input_position will be returned or None if the input_position is outside of any polygon.
    :rtype: shapely.geometry.Polygon or list[shapely.geometry.Polygon] or None or list
    :raises ValueError: If the input_position is outside of any polygon and raise_outside is True.
    """
    from shapely.geometry import Point

    if not isinstance(input_position, Point):
        input_position = Point(input_position)

    position_polygons = []
    if outside_behavior != "closest":
        for polygon_name, polygon in arena_polygon_dict.items():
            if polygon.contains(input_position):
                position_polygons.append(polygon_name)
                if not return_all:
                    break
    else:
        polygon_distances = {k: v.distance(input_position) for k, v in arena_polygon_dict.items()}
        closest_polygon = min(polygon_distances, key=polygon_distances.get)
        position_polygons.append(closest_polygon)

    if not len(position_polygons) == 0:
        return position_polygons[0] if not return_all else position_polygons
    elif outside_behavior == "raise":
        raise ValueError("Position is outside of arena")
    else:  # outside_behavior is "return" or "closest"
        return None if not return_all else []


def save_arena_to_json(arena_dict, filename, overwrite=False):
    """
    Saves a given arena dictionary mapping polygon labels to their shapely Polygons to a JSON file.
    To accomplish this, the function first converts each polygon in the dictionary to a numpy array by accessing its exterior coordinates.

    :param arena_dict: A dictionary mapping polygon labels to their shapely Polygons.
    :type arena_dict: dict
    :param filename: The path of the JSON file to save the arena dictionary to.
    :type filename: str
    :param overwrite: Whether or not to overwrite an existing file with the same name. Defaults to False.
    :type overwrite: bool
    """
    import json
    import numpy as np
    array_arena_dict = {polygon_label: np.stack(polygon.exterior.xy).T.tolist() for polygon_label, polygon in
                        arena_dict.items()}
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"File {filename} already exists!")
    with open(filename, 'w') as out_file:
        json.dump(array_arena_dict, out_file)


def load_arena_from_json(filename):
    """
    Loads an arena from a JSON file. The loaded array is expected to contain a mapping of polygon labels to their
    corresponding exterior coordinates. The loaded array is converted into a list of Polygons using the `Polygon` class.

    :param filename: The path to the JSON file containing the arena data.
    :type filename: str
    :return: A dictionary containing the arena data.
    :rtype: dict
    """
    import json
    from shapely import Polygon
    with open(filename, "r") as json_file:
        array_arena_dict = json.load(json_file)
    return {polygon_label: Polygon(array) for polygon_label, array in array_arena_dict.items()}
