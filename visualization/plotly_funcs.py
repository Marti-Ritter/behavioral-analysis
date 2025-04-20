"""
Functions for plotting with plotly.
"""

from functools import reduce

import numpy as np
import pandas as pd
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from PIL import ImageColor
from plotly.subplots import make_subplots
from tqdm.auto import tqdm
from ..utility.builtin_classes.dicts import dicts_to_dict_of_tuples

from ..math_tools.geometry_funcs import get_sphere_coordinates


# taken from https://stackoverflow.com/a/67912302
def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)


# Identical to Adam's answer
def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    def hex_to_rgb(hex_color):
        """
        Returns the rgb string for the given hex color.

        :param hex_color: hex color
        :type hex_color: str
        :return: rgb string
        :rtype: str
        """
        return "rgb" + str(ImageColor.getcolor(hex_color, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    low_cutoff, low_color = 0, colorscale[0][1]
    high_cutoff, high_color = 1, colorscale[-1][1]
    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )


def make_sphere(center_point, radius, resolution=20):
    """
    Returns a plotly surface object of a sphere with center_point and radius.

    :param center_point: tuple of floats (x, y, z)
    :type center_point: tuple of float
    :param radius: radius of the sphere
    :type radius: float
    :param resolution: resolution of the sphere
    :type resolution: int
    :return: plotly surface object
    :rtype: go.Surface
    """

    (x_pns_surface, y_pns_surface, z_pns_surface) = get_sphere_coordinates(center_point, radius, resolution=resolution)
    return go.Surface(x=x_pns_surface, y=y_pns_surface, z=z_pns_surface, opacity=0.5)


def generate_arrows(x, y, z, u, v, w, head_length=0.1, line_width=3, c='black', legendgroup=None,
                    zero_length_return=False):
    """
    Generates arrows for a 3D plotly plot.

    :param x: x coordinates of the vector tails
    :type x: np.ndarray
    :param y: y coordinates of the vector tails
    :type y: np.ndarray
    :param z: z coordinates of the vector tails
    :type z: np.ndarray
    :param u: x components of the vector ends
    :type u: np.ndarray
    :param v: y components of the vector ends
    :type v: np.ndarray
    :param w: z components of the vector ends
    :type w: np.ndarray
    :param head_length: length of the arrow head relative to the length of the vector
    :type head_length: float
    :param line_width: width of the arrow lines
    :type line_width: float
    :param c: color of the arrows
    :type c: str
    :param legendgroup: legendgroup of the arrows
    :type legendgroup: str
    :param zero_length_return: if True, returns zero length vectors as well, shown as scatter points
    :type zero_length_return: bool
    :return: list of plotly objects
    :rtype: tuple of lists of go.Scatter3d and go.Cone and, optionally, go.Scatter3d
    """

    def _minmax_normalization(value, minimum=0, maximum=1):
        return (value - minimum) / (maximum - minimum)

    # Calculate the length of the vectors
    lengths = np.sqrt(u ** 2 + v ** 2 + w ** 2)

    # Filter out vectors with length zero
    nonzero_indices = np.where(lengths > 0)
    zero_indices = np.where(lengths == 0)
    nonzero_lengths = lengths[nonzero_indices]

    filtered_x, filtered_y, filtered_z = x[nonzero_indices], y[nonzero_indices], z[nonzero_indices]
    head_lengths = nonzero_lengths * head_length

    # Normalize the vectors
    u_norm = u[nonzero_indices] / nonzero_lengths
    v_norm = v[nonzero_indices] / nonzero_lengths
    w_norm = w[nonzero_indices] / nonzero_lengths

    # Calculate the end points of the lines
    end_x = filtered_x + u_norm * nonzero_lengths
    end_y = filtered_y + v_norm * nonzero_lengths
    end_z = filtered_z + w_norm * nonzero_lengths

    # Calculate the end points of the cones
    cone_x = end_x - head_lengths * u_norm
    cone_y = end_y - head_lengths * v_norm
    cone_z = end_z - head_lengths * w_norm

    # Create the line and cone traces
    line_traces = []
    cone_traces = []
    if isinstance(c, str):
        color = c
    else:
        color_values = c

    for i in range(len(filtered_x)):
        if isinstance(c, str):
            color = c
        if not isinstance(c, str):
            if not isinstance(c[i], str):
                color = get_color("viridis", _minmax_normalization(color_values[i],
                                                                   minimum=np.min(color_values),
                                                                   maximum=np.max(color_values)))
            else:
                color = c[i]

        line_trace = go.Scatter3d(x=[filtered_x[i], cone_x[i]], y=[filtered_y[i], cone_y[i]],
                                  z=[filtered_z[i], cone_z[i]],
                                  mode='lines', line=dict(width=line_width, color=color),
                                  showlegend=False if i > 0 else True, legendgroup=legendgroup, name=legendgroup)
        cone_trace = go.Cone(x=[cone_x[i]], y=[cone_y[i]], z=[cone_z[i]],
                             u=[u_norm[i]], v=[v_norm[i]], w=[w_norm[i]],
                             sizemode='scaled', sizeref=head_lengths[i],
                             anchor='tail', showlegend=False, showscale=False,
                             colorscale=[color, color], legendgroup=legendgroup)
        line_traces.append(line_trace)
        cone_traces.append(cone_trace)

    if zero_length_return:
        zero_length_traces = []
        for i, j in enumerate(zero_indices):
            color = None
            if isinstance(c, str):
                color = c
            if not isinstance(c, str):
                if not isinstance(c[i], str):
                    color = get_color("viridis", _minmax_normalization(color_values[i],
                                                                       minimum=np.min(color_values),
                                                                       maximum=np.max(color_values)))
                else:
                    color = c[i]
            showlegend = (i == 0) and not line_traces
            zero_length_trace = go.Scatter3d(x=x[j], y=y[j], z=z[j],
                                             mode='markers', marker=dict(size=line_width * 1, color=color),
                                             showlegend=showlegend, legendgroup=legendgroup, name=legendgroup)
            zero_length_traces.append(zero_length_trace)

        return line_traces, cone_traces, zero_length_traces

    else:
        return line_traces, cone_traces


def normalized_cone(x, y, z, u, v, w, sizeref=1):
    """
    Creates a cone trace from the given vector data. The vectors are normalized and the cone size is absolute.

    :param x: x coordinates of the vector base-points
    :type x: np.ndarray
    :param y: y coordinates of the vector base-points
    :type y: np.ndarray
    :param z: z coordinates of the vector base-points
    :type z: np.ndarray
    :param u: x components of the vector endpoints
    :type u: np.ndarray
    :param v: y components of the vector endpoints
    :type v: np.ndarray
    :param w: z components of the vector endpoints
    :type w: np.ndarray
    :param sizeref: reference for the cone size
    :type sizeref: float
    :return: plotly cone trace
    :rtype: go.Cone
    """
    # Calculate the length of the vector
    length = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    zero_length_filter = length != 0

    # Normalize the vector
    filtered_length = length[zero_length_filter]
    u_norm = u[zero_length_filter] / filtered_length
    v_norm = v / filtered_length
    w_norm = w / filtered_length

    # Create the cone trace
    cone_trace = go.Cone(x=[x], y=[y], z=[z],
                         u=u_norm, v=v_norm, w=w_norm,
                         sizemode='absolute', sizeref=sizeref, anchor='tail', showlegend=False, showscale=False)

    # Return the line and cone traces
    return cone_trace


def create_rgb_surface(rgb_img, depth_img, depth_cutoff=20, **kwargs):
    """
    Creates a surface plot from the given RGB and depth images. The depth image is used to create a depth map and the
    RGB image is used to create a color map. The depth map is used to create a surface plot and the color map is used
    to color the surface plot.

    Taken from https://github.com/plotly/plotly.py/issues/1827#issue-508669867 and
    https://www.kaggle.com/code/kmader/show-rgb-d-image-in-3d-plotly/notebook.

    :param rgb_img: RGB image
    :type rgb_img: np.ndarray
    :param depth_img: depth image
    :type depth_img: np.ndarray
    :param depth_cutoff: depth cutoff below which the depth values are set to NaN
    :type depth_cutoff: float
    :param kwargs: keyword arguments passed to the plotly surface trace
    :type kwargs: dict
    :return: plotly surface trace
    :rtype: go.Surface
    """
    rgb_img = rgb_img.swapaxes(0, 1)[:, ::-1]
    depth_img = depth_img.swapaxes(0, 1)[:, ::-1]
    eight_bit_img = Image.fromarray(rgb_img).convert('P', palette='WEB', dither=None)
    idx_to_color = np.array(eight_bit_img.getpalette()).reshape((-1, 3))
    colorscale = [[i / 255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    depth_map = depth_img.copy().astype('float')
    depth_map[depth_map < depth_cutoff] = np.nan
    return go.Surface(
        z=depth_map,
        surfacecolor=np.array(eight_bit_img),
        cmin=0,
        cmax=255,
        colorscale=colorscale,
        showscale=False,
        **kwargs
    )


def create_rgb_surface_frames(rgb_imgs, depth_imgs, frame_names=None, depth_cutoff=20, **kwargs):
    """
    Creates a list of plotly frames from the given RGB and depth images. The data of each frame is generated by the
    create_rgb_surface function using the zip of the RGB and depth images.

    :param rgb_imgs: A list of RGB images
    :type rgb_imgs: list of np.ndarray
    :param depth_imgs: A list of depth images
    :type depth_imgs: list of np.ndarray
    :param frame_names: A list of frame names
    :type frame_names: list of str or None
    :param depth_cutoff: depth cutoff below which the depth values are set to NaN
    :type depth_cutoff: float
    :param kwargs: keyword arguments passed to the create_rgb_surface function
    :return: A list of plotly frames
    :rtype: list of go.Frame
    """
    frames = []
    if frame_names is None:
        frame_names = [str(i) for i in range(len(rgb_imgs))]
    name_rgb_depth_imgs = list(zip(frame_names, rgb_imgs, depth_imgs))
    for (frame_name, rgb_img, depth_img) in tqdm(name_rgb_depth_imgs, total=len(name_rgb_depth_imgs)):
        frames.append(
            go.Frame(
                data=[create_rgb_surface(rgb_img, depth_img, depth_cutoff=depth_cutoff, **kwargs)],
                name=frame_name
            )
        )
    return frames


def get_xy_data_from_frame(plotly_frame):
    """
    Returns the x and y data from a plotly frame across all traces.

    :param plotly_frame: plotly frame
    :type plotly_frame: go.Frame
    :return: x and y data
    :rtype: tuple of np.ndarray
    """
    frame_data = plotly_frame.data
    x = np.array(dicts_to_dict_of_tuples(*frame_data)["x"]).squeeze()
    y = np.array(dicts_to_dict_of_tuples(*frame_data)["y"]).squeeze()
    return x, y


def autorange_frames_to_data(plotly_frames, padding=0.1, pad_mode="relative", range_mode="local"):
    """
    Sets the x and y-axis ranges of the given plotly frames to the data in the frames.

    :param plotly_frames: Plotly frames
    :type plotly_frames: list of go.Frame or go.Frames
    :param padding: Padding to add to the data range. If pad_mode is 'relative', this is a fraction of the data range.
    :type padding: float
    :param range_mode: Mode for setting the range. Can be 'local' or 'global'.
    :type range_mode: str
    :param pad_mode: Mode for padding. Can be 'relative' or 'absolute'.
    :type pad_mode: str

    """
    x_ranges = []
    y_ranges = []
    for frame in plotly_frames:
        x, y = get_xy_data_from_frame(frame)
        x_ranges.append(np.max(x) - np.min(x))
        y_ranges.append(np.max(y) - np.min(y))

    if range_mode == "global":
        x_ranges = [np.max(x_ranges)] * len(plotly_frames)
        y_ranges = [np.max(y_ranges)] * len(plotly_frames)

    for frame, x_range, y_range in zip(plotly_frames, x_ranges, y_ranges):
        x, y = get_xy_data_from_frame(frame)
        if pad_mode == "relative":
            x_padding = x_range * padding
            y_padding = y_range * padding
        elif pad_mode == "absolute":
            x_padding = padding
            y_padding = padding
        else:
            raise ValueError("pad_mode must be 'relative' or 'absolute'")
        frame.layout.update(xaxis_range=[np.min(x) - x_padding, np.max(x) + x_padding],
                            yaxis_range=[np.min(y) - y_padding, np.max(y) + y_padding])


def df_comparison_plot(*input_dfs, comparison_func=pd.concat, comparison_func_kwargs=None,
                       plot_func=px.scatter, **kwargs):
    """
    Plots the given dataframes in a comparison plot.

    :param input_dfs: Dataframes to plot
    :type input_dfs: pd.DataFrame
    :param comparison_func: Function to use for comparison
    :type comparison_func: function
    :param comparison_func_kwargs: Keyword arguments passed to the comparison function
    :type comparison_func_kwargs: dict
    :param plot_func: Function to use for plotting
    :type plot_func: function
    :param kwargs: Keyword arguments passed to the plotly figure
    :type kwargs: dict
    :return: plotly figure
    :rtype: go.Figure
    """
    comparison_func_kwargs = comparison_func_kwargs if comparison_func_kwargs is not None else {}
    comparison_df = comparison_func(input_dfs, **comparison_func_kwargs)
    return plot_func(comparison_df, **kwargs)


class AnimationButtons:
    """
    Convenience class for creating animation buttons for plotly figures.
    """
    # shamelessly taken from https://stackoverflow.com/q/66016464
    @staticmethod
    def play_scatter(frame_duration=500, transition_duration=300):
        """
        Creates a play button for a scatter plot.

        :param frame_duration: How long each frame should be displayed in milliseconds
        :type frame_duration: int
        :param transition_duration: How long the transition between frames should be in milliseconds
        :type transition_duration: int
        :return: The dict for the play button
        :rtype: dict
        """
        return dict(label="Play", method="animate", args=[None, {"frame": {"duration": frame_duration, "redraw": False},
                                                                 "fromcurrent": True,
                                                                 "transition": {"duration": transition_duration,
                                                                                "easing": "quadratic-in-out"}}])

    @staticmethod
    def play(frame_duration=1000, transition_duration=0):
        """
        Creates a play button for a plotly figure.

        :param frame_duration: How long each frame should be displayed in milliseconds
        :type frame_duration: int
        :param transition_duration: How long the transition between frames should be in milliseconds
        :type transition_duration: int
        :return: The dict for the play button
        :rtype: dict
        """
        return dict(label="Play", method="animate", args=[None, {"frame": {"duration": frame_duration, "redraw": True},
                                                                 "mode": "immediate", "fromcurrent": True,
                                                                 "transition": {"duration": transition_duration,
                                                                                "easing": "linear"}}])

    @staticmethod
    def pause():
        """
        Creates a pause button for a plotly figure.
        :return: The dict for the pause button
        :rtype: dict
        """
        return dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                                    "mode": "immediate",
                                                                    "transition": {"duration": 0}}])


def build_slider_for_frames(plotly_frames, slider_dict_kwargs=None, **frame_kwargs):
    """
    Builds a slider for the given plotly frames.

    :param plotly_frames: Plotly frames
    :type plotly_frames: tuple of go.Frame or list of go.Frame or go.Frames
    :param slider_dict_kwargs: Keyword arguments passed to the plotly slider
    :type slider_dict_kwargs: dict
    :param frame_kwargs: Keyword arguments passed to the plotly frame
    :type frame_kwargs: dict
    :return: plotly slider
    :rtype: dict
    """
    if slider_dict_kwargs is None:
        slider_dict_kwargs = {}

    default_frame_args = {"frame": {"duration": 0},
                          "mode": "immediate",
                          "fromcurrent": True,
                          "transition": {"duration": 0}, }

    frame_kwargs = {**default_frame_args, **frame_kwargs}

    steps = [
        {
            "args": [[f.name], frame_kwargs],
            "label": str(k),
            "method": "animate",
        }
        for k, f in enumerate(plotly_frames)
    ]

    # Create slider
    default_slider_dict = dict(pad={"b": 10, "t": 60}, len=1., x=0., y=0,)
    slider_dict = {**default_slider_dict, **slider_dict_kwargs, "steps": steps}

    return slider_dict


def append_or_update_layout_element(existing_elements, element_index=None, index_warn_message=None, **element_kwargs):
    """
    Appends or updates an element in the given list of elements. If the element index is greater than or equal to the
    number of existing elements, the element will be appended at the end. If the element index is less than the number
    of existing elements, the element at the given index will be updated.

    Originally, this function was intended to also allow insertion of new elements at a given position, but it turns out
    that the plotly update implementation does not support the mixed update of existing elements with parsed elements
    and dictionaries. Therefore, this function will only append or update elements.

    :param existing_elements: A list of elements from a plotly layout, such as buttons, sliders, or update menus
    :type existing_elements: list of plotly.basedatatypes.BaseLayoutHierarchyType
    :param element_index: Index of the element to append or update. Defaults to None, which will append the element at
    the end.
    :type element_index: int or None
    :param index_warn_message: Warning message to print if the element index is greater than the number of existing
    elements
    :type index_warn_message: str or None
    :param element_kwargs: Keyword arguments passed to the element
    :type element_kwargs: dict
    :return: The updated list of elements
    :rtype: list of plotly.basedatatypes.BaseLayoutHierarchyType
    """
    if element_index is None:
        element_index = len(existing_elements)

    existing_elements = list(existing_elements)

    if element_index >= len(existing_elements):
        if (element_index != len(existing_elements)) and (index_warn_message is not None):
            print(index_warn_message)
        existing_elements.append(element_kwargs)
    else:
        existing_elements[element_index].update(element_kwargs)
    return existing_elements


def add_frame_slider_to_figure(plotly_fig, slider_index=None, **slider_kwargs):
    """
    Adds a slider to the given plotly figure. This operates on the plotly figure in place.

    :param plotly_fig: Plotly figure
    :type plotly_fig: go.Figure
    :param slider_index: Index of the slider. If greater than the number of sliders in the figure, the slider will be
    appended at the end.
    :type slider_index: int or None
    :param slider_kwargs: Keyword arguments passed to the plotly slider
    :type slider_kwargs: dict
    """
    slider = build_slider_for_frames(plotly_fig.frames, **slider_kwargs)
    warn_message = "Warning: slider_index {} is greater than the number of sliders in the figure. " \
                   "Slider will be appended at position {}".format(slider_index, len(plotly_fig.layout["sliders"]))
    updated_sliders = append_or_update_layout_element(plotly_fig.layout["sliders"], slider_index,
                                                      index_warn_message=warn_message, **slider)
    plotly_fig.layout.update(sliders=updated_sliders)


def add_button_to_update_menu(update_menu, button_index=None, **button_kwargs):
    """
    Adds a button to the given update menu. This operates on the update menu in place.

    :param update_menu: A plotly update menu
    :type update_menu: go.layout._updatemenu.Updatemenu
    :param button_index: Index of the button. If greater than the number of buttons in the update menu, the button
    will be appended at the end.
    :type button_index: int or None
    :param button_kwargs: Keyword arguments passed to the plotly button
    :type button_kwargs: dict
    """
    warn_message = "Warning: button_index {} is greater than the number of buttons in the update menu. " \
                   "Button will be appended at position {}".format(button_index, len(update_menu["buttons"]))
    updated_buttons = append_or_update_layout_element(update_menu["buttons"], button_index,
                                                      index_warn_message=warn_message, **button_kwargs)
    update_menu.update(buttons=updated_buttons)


def add_update_menu_to_figure(plotly_fig, update_menu_index=None, **update_menu_kwargs):
    """
    Adds an update menu to the given plotly figure. This operates on the plotly figure in place.

    :param plotly_fig: A plotly figure
    :type plotly_fig: go.Figure
    :param update_menu_index: Index of the update menu. If greater than the number of update menus in the figure, the
    update menu will be appended at the end.
    :type update_menu_index: int or None
    :param update_menu_kwargs: Keyword arguments passed to the plotly update menu
    :type update_menu_kwargs: Any
    """
    warn_message = "Warning: update_menu_index {} is greater than the number of update menus in the figure. " \
                   "Update menu will be appended at position {}".format(update_menu_index,
                                                                        len(plotly_fig.layout["updatemenus"]))
    updated_update_menus = append_or_update_layout_element(plotly_fig.layout["updatemenus"], update_menu_index,
                                                           index_warn_message=warn_message, **update_menu_kwargs)
    plotly_fig.layout.update(updatemenus=updated_update_menus)


def add_update_button_to_figure(plotly_fig, update_menu_index=0, **button_kwargs):
    """
    Adds a button to the given plotly figure. This operates on the plotly figure in place.

    :param plotly_fig: A plotly figure
    :type plotly_fig: go.Figure
    :param update_menu_index: Index of the update menu. If greater than the number of update menus in the figure, a new
    update menu will be appended at the end.
    :type update_menu_index: int
    :param button_kwargs: Keyword arguments passed to the plotly button
    :type button_kwargs: dict
    """
    if update_menu_index >= len(plotly_fig.layout["updatemenus"]):
        add_update_menu_to_figure(plotly_fig, update_menu_index=update_menu_index)
        update_menu_index = len(plotly_fig.layout["updatemenus"]) - 1
    add_button_to_update_menu(plotly_fig.layout["updatemenus"][update_menu_index], **button_kwargs)


def add_play_button_to_figure(plotly_fig, button_kwarg_dict=None, scatter=True, **update_menu_kwargs):
    """
    Adds a play button to the given plotly figure. This operates on the plotly figure in place.

    :param plotly_fig: A plotly figure
    :type plotly_fig: go.Figure
    :param button_kwarg_dict: Keyword arguments passed to the play button
    :type button_kwarg_dict: dict
    :param scatter: If True, adds a play button for a scatter plot. Otherwise, adds a play button for a general plot.
    :type scatter: bool
    :param update_menu_kwargs: Keyword arguments passed to add_update_button_to_figure
    :type update_menu_kwargs: dict
    """
    if button_kwarg_dict is None:
        button_kwarg_dict = {}

    button_creation_func = AnimationButtons.play_scatter if scatter else AnimationButtons.play
    play_button_dict = button_creation_func(**button_kwarg_dict)
    add_update_button_to_figure(plotly_fig, **update_menu_kwargs, **play_button_dict)


def add_pause_button_to_figure(plotly_fig, **update_menu_kwargs):
    """
    Adds a pause button to the given plotly figure. This operates on the plotly figure in place.

    :param plotly_fig: A plotly figure
    :type plotly_fig: go.Figure
    :param update_menu_kwargs: Keyword arguments passed to add_update_button_to_figure
    :type update_menu_kwargs: dict
    """
    pause_button_kwargs = AnimationButtons.pause()
    add_update_button_to_figure(plotly_fig, **update_menu_kwargs, **pause_button_kwargs)


def add_play_pause_buttons_to_figure(plotly_fig, play_button_kwarg_dict=None, scatter=True, **update_menu_kwargs):
    """
    Convenience function for adding a pause and a play button to the given plotly figure. This operates on the plotly
    figure in place.

    :param plotly_fig: A plotly figure
    :type plotly_fig: go.Figure
    :param play_button_kwarg_dict: Keyword arguments passed to the play button
    :type play_button_kwarg_dict: dict or None
    :param scatter: If True, adds a play button for a scatter plot. Otherwise, adds a play button for a general plot.
    :type scatter: bool
    :param update_menu_kwargs: Keyword arguments passed to add_update_button_to_figure
    :type update_menu_kwargs: Any
    """
    if play_button_kwarg_dict is None:
        play_button_kwarg_dict = {}
    add_play_button_to_figure(plotly_fig, button_kwarg_dict=play_button_kwarg_dict, scatter=scatter,
                              **update_menu_kwargs)
    add_pause_button_to_figure(plotly_fig, **update_menu_kwargs)


def combine_figure_frames(*plotly_figs, names=None):
    """
    Combines the frames of the given plotly figures into a single list of frames. This function expects that the given
    figures have the same number of frames and are compatible in terms of the data they contain.

    :param plotly_figs: An arbitrary number of plotly figures
    :type plotly_figs: go.Figure
    :param names: Names of the frames
    :type names: list of str or None
    :return: list of frames
    :rtype: list of go.Frame
    """
    if names is None:
        names = [f.name for f in plotly_figs[0].frames]

    return [go.Frame(data=reduce(lambda x, y: x + y, [fig.frames[i].data for fig in plotly_figs]), name=names[i]) for
            i, f in enumerate(plotly_figs[0].frames)]


def combine_figs_as_subplots(*plotly_figs, positions=None, combine_frames=True, **kwargs):
    """
    A function that takes multiple plotly figures and combines them into a single figure with subplots.

    :param plotly_figs: An arbitrary number of plotly figures
    :type plotly_figs: go.Figure
    :param positions: Positions of the subplots
    :type positions: list of tuples of int
    :param combine_frames: If True, combines the frames of the given figures into a single list of frames
    :type combine_frames: bool
    :param kwargs: Keyword arguments passed to the plotly subplots
    :type kwargs: Any
    :return: plotly figure
    :rtype: go.Figure
    """

    combined_fig = make_subplots(**kwargs)

    if positions is None:
        # If no positions are given, use the default positions
        positions = list(combined_fig._get_subplot_coordinates())

    for i, fig in enumerate(plotly_figs):
        if combine_frames:
            combined_fig.add_traces(fig.frames[0].data, rows=positions[i][0], cols=positions[i][1])
        else:
            combined_fig.add_traces(fig.data, rows=positions[i][0], cols=positions[i][1])

    features_to_ignore = ["domain", "anchor"]  # features that should not be copied
    # copying axes layouts
    for i, fig in enumerate(plotly_figs):
        x_axis_features = [f for f in fig.layout.xaxis if f not in features_to_ignore]
        y_axis_features = [f for f in fig.layout.yaxis if f not in features_to_ignore]

        # copying axes layouts
        for feature in x_axis_features:
            combined_fig.layout["xaxis{}".format(i + 1)][feature] = fig.layout.xaxis[feature]
        for feature in y_axis_features:
            combined_fig.layout["yaxis{}".format(i + 1)][feature] = fig.layout.yaxis[feature]

    # combining frames
    if combine_frames:
        frames = combine_figure_frames(*plotly_figs)
        combined_fig.update(frames=frames)

    return combined_fig


def combine_figs_into_one(*plotly_figs, layout_source_index=0, combine_frames=True, **kwargs):
    """
    A function that takes multiple plotly figures and combines them into a single figure.

    :param plotly_figs: An arbitrary number of plotly figures
    :type plotly_figs: go.Figure
    :param layout_source_index:
    :param combine_frames: If True, combines the frames of the given figures into a single list of frames
    :type combine_frames: bool
    :param kwargs: Keyword arguments passed to the plotly figure
    :type kwargs: Any
    :return: plotly figure
    :rtype: go.Figure
    """

    combined_fig = go.Figure(**kwargs)

    for i, fig in enumerate(plotly_figs):
        if combine_frames:
            combined_fig.add_traces(fig.frames[0].data)
        else:
            combined_fig.add_traces(fig.data)

    combined_fig.update_layout(plotly_figs[layout_source_index].layout.to_plotly_json())

    # combining frames
    if combine_frames:
        frames = combine_figure_frames(*plotly_figs)
        combined_fig.update(frames=frames)

    return combined_fig


def animated_line_plot(pd_series, x_label="Time", y_label="Value", title="Animated Line Plot", series_name="Trace"):
    """
    Creates an animated line plot from the given pandas-series.
    Adapted from https://stackoverflow.com/a/66244663

    :param pd_series:
    :type pd_series:
    :param x_label:
    :type x_label:
    :param y_label:
    :type y_label:
    :param title:
    :type title:
    :param series_name:
    :type series_name:
    :return:
    :rtype:
    """
    pd_series = pd_series.sort_index()

    # Base plot
    fig = go.Figure(
        layout=go.Layout(
            updatemenus=[dict(type="buttons", direction="right", x=0.9, y=1.16), ],
            xaxis=dict(range=[pd_series.index.min(), pd_series.index.max()],
                       autorange=False, tickwidth=2,
                       title_text=x_label),
            yaxis=dict(range=[pd_series.min(), pd_series.max()],
                       autorange=False,
                       title_text=y_label),
            title=title,
        ))

    # Add traces
    fig.add_trace(
        go.Scatter(x=pd_series.iloc[[0]].index.values,
                   y=pd_series.iloc[[0]].values,
                   name=series_name,
                   visible=True,
                   mode='lines+markers'))

    # Animation
    frames = [go.Frame(
        data=[go.Scatter(x=pd_series.iloc[:k].index.values, y=pd_series.iloc[:k].values,
                         mode='lines+markers', name=str(k))]
    ) for k in range(len(pd_series))]
    fig.update(frames=frames)

    # Buttons
    add_play_pause_buttons_to_figure(fig, scatter=False, update_menu_index=0)
    add_frame_slider_to_figure(fig, slider_index=0, **build_slider_for_frames(frames))

    fig.update_layout(
        xaxis_range=[pd_series.index.min(), pd_series.index.max()],
        yaxis_range=[pd_series.min(), pd_series.max()],
    )

    return fig


def animated_px_line_plot(pd_df, x, y, remove_axes_assignment=False, **px_kwargs):
    """
    Creates an animated line plot from the given pandas-series. The animation consists of a line plot that is updated
    frame by frame (x-axis) and a frame slider that allows to select the frame up to which the plot is shown.
    Adapted from https://stackoverflow.com/a/66244663

    :param pd_df: Pandas dataframe
    :type pd_df: pd.DataFrame
    :param x: x-axis column
    :type x: str
    :param y: y-axis column
    :type y: str
    :param remove_axes_assignment: If True, removes the assignment of the x- and y-axes to the plotly figure. This is
    done automatically by plotly.express but is generally only needed if the trace is shown in a group of figures, e.g.
    in a subplot, a facet, or an inset. If you want to put this figure into a combined figure, you should set this to
    True, as it otherwise interferes with the axes assignment of the combined figure.
    :type remove_axes_assignment: bool
    :param px_kwargs: Keyword arguments passed to the plotly express line plot function
    :type px_kwargs: Any
    :return: plotly figure
    :rtype: go.Figure
    """
    pd_df = pd_df.set_index(x).sort_index()
    x_range = pd_df.index.min(), pd_df.index.max()
    y_range = pd_df[y].min(), pd_df[y].max()

    # Base plot
    fig = px.line(pd_df.loc[[x_range[0]]].reset_index(), x=x, y=y, **px_kwargs)

    # Animation
    frames = [go.Frame(data=px.line(pd_df.loc[:k].reset_index(), x=x, y=y, **px_kwargs).data, name=str(k)) for k in
              range(x_range[0], x_range[1] + 1)]
    fig.update(frames=frames)

    if remove_axes_assignment:
        update_data_and_frames(fig, {"xaxis": None, "yaxis": None})

    # Buttons
    add_update_menu_to_figure(fig, update_menu_index=0, type="buttons")
    add_play_pause_buttons_to_figure(fig, scatter=False, update_menu_index=0)
    add_frame_slider_to_figure(fig, slider_index=0, **build_slider_for_frames(frames))

    fig.update_layout(xaxis_range=x_range, yaxis_range=y_range)

    return fig


def update_data(plotly_fig, data_update_kwargs):
    """
    Updates the data of the given plotly figure with the given keyword arguments.

    :param plotly_fig: Plotly figure
    :type plotly_fig: go.Figure
    :param data_update_kwargs: Keyword arguments passed to the plotly figure data. Can be a single dict or a list of
    dicts with the same length as the frames or a list of lists of dicts with the same length as the frames and traces.
    Instead of a dict, a function can be passed that takes a trace as an argument and returns a dict.
    :type data_update_kwargs: (dict or function) or list of (dict or function) or list of (list of (dict or function))
    """
    for i, trace in enumerate(plotly_fig.data):
        if isinstance(data_update_kwargs, (list, tuple)):
            trace_update_kwargs = data_update_kwargs[i]
        else:
            trace_update_kwargs = data_update_kwargs

        if callable(trace_update_kwargs):
            trace_update_kwargs = trace_update_kwargs(trace)

        trace.update(trace_update_kwargs)


def update_frame_data(frames, data_update_kwargs):
    """
    Updates the data of the given frames with the given keyword arguments.

    :param frames: Plotly frames
    :type frames: list of go.Frame or tuple of go.Frame or go.Frames
    :param data_update_kwargs: Keyword arguments passed to the plotly frame data. Can be a single dict or a list of
    dicts with the same length as the frames or a list of lists of dicts with the same length as the frames and traces.
    Instead of a dict, a function can be passed that takes a trace as an argument and returns a dict.
    :type data_update_kwargs: (dict or function) or list of (dict or function) or list of (list of (dict or function))
    """
    for i, frame in enumerate(frames):
        if isinstance(data_update_kwargs, (list, tuple)):
            frame_update_kwargs = data_update_kwargs[i]
        else:
            frame_update_kwargs = data_update_kwargs

        for j, trace in enumerate(frame.data):
            if isinstance(frame_update_kwargs, (list, tuple)):
                trace_update_kwargs = frame_update_kwargs[j]
            else:
                trace_update_kwargs = frame_update_kwargs

            if callable(trace_update_kwargs):
                trace_update_kwargs = trace_update_kwargs(trace)

            trace.update(trace_update_kwargs)


def update_data_and_frames(plotly_fig, data_update_kwargs):
    """
    A convenience function for updating the data and frame data of a plotly figure.

    :param plotly_fig: Plotly figure
    :type plotly_fig: go.Figure
    :param data_update_kwargs: Keyword arguments passed to the plotly figure data. Can be a single dict or a list of
    dicts with the same length as the frames or a list of lists of dicts with the same length as the frames and traces.
    Instead of a dict, a function can be passed that takes a trace as an argument and returns a dict.
    :type data_update_kwargs: (dict or function) or list of (dict or function) or list of (list of (dict or function))
    """
    update_data(plotly_fig, data_update_kwargs)
    update_frame_data(plotly_fig.frames, data_update_kwargs)


def get_figure_aspect_ratio(plotly_fig):
    """
    Returns the aspect ratio of the given plotly figure. This function expects the figure to have an already set width
    and height. For the margin, the default value of zero are used if not set.

    :param plotly_fig: Plotly figure
    :type plotly_fig: go.Figure
    """
    width, height = plotly_fig.layout.width, plotly_fig.layout.height
    margin = plotly_fig.layout.margin.to_plotly_json()
    for margin_feature in ["l", "r", "b", "t"]:
        if margin_feature not in margin:
            margin[margin_feature] = 0

    return (width - margin["l"] - margin["r"]) / (height - margin["t"] - margin["b"])


def set_xy_ratio_once(plotly_fig, xy_ratio=1., set_y_range=True):
    """
    Sets the x and y range of the given plotly figure to the given ratio. This is useful for ensuring that the figure
    has a fixed aspect ratio. It should be kept in mind that any subsequent changes to the figure layout will overwrite
    the x and y range.

    This should theoretically work, but if not set by hand, the figure will have no width and height before being
    displayed, which will result in an error. It also appears that these two values are not always (re-)set even after
    the figure is displayed, so it is currently not recommended to use this function.

    :param plotly_fig: Plotly figure
    :type plotly_fig: go.Figure
    :param xy_ratio: Ratio of x and y axis
    :type xy_ratio: float
    :param set_y_range: If True, sets the y range to the given ratio. Otherwise, sets the x range to the given ratio.
    :type set_y_range: bool
    """
    x_range, y_range = plotly_fig.layout.xaxis.range, plotly_fig.layout.yaxis.range
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]

    aspect_correction = get_figure_aspect_ratio(plotly_fig) * xy_ratio

    if set_y_range:
        new_y_span = x_span / aspect_correction
        y_center = (y_range[0] + y_range[1]) / 2
        new_y_range = [y_center - new_y_span / 2, y_center + new_y_span / 2]
        if y_range[0] > y_range[1]:
            new_y_range = new_y_range[::-1]
        plotly_fig.update_layout(xaxis_range=x_range, yaxis_range=new_y_range)
    else:
        new_x_span = y_span * aspect_correction
        x_center = (x_range[0] + x_range[1]) / 2
        new_x_range = [x_center - new_x_span / 2, x_center + new_x_span / 2]
        if x_range[0] > x_range[1]:
            new_x_range = new_x_range[::-1]
        plotly_fig.update_layout(xaxis_range=new_x_range, yaxis_range=y_range)
