import pathlib
from tempfile import TemporaryDirectory

import matplotlib.axis
import matplotlib.pyplot as plt
import matplotlib.spines
import numpy as np
from PIL import Image
from matplotlib.animation import FuncAnimation
from matplotlib.image import imread
from matplotlib.offsetbox import HPacker, VPacker, AnnotationBbox, TextArea
from matplotlib.text import Text
from ..utility.builtin_classes.iterables import ensure_list
from ..utility.builtin_classes.objects import copy_object
from ..utility.builtin_classes.funcs import get_func_kwargs


def get_all_matplotlib_children(matplotlib_object, excluded_types=None):
    excluded_types = [] if excluded_types is None else excluded_types
    children = [child for child in matplotlib_object.get_children() if
                not any(isinstance(child, excluded_type) for excluded_type in excluded_types)]
    if children:
        return [grandchild for child in children for grandchild in ensure_list(get_all_matplotlib_children(child))]
    else:
        return matplotlib_object


def get_plot_elements(matplotlib_object):
    return get_all_matplotlib_children(matplotlib_object,
                                       excluded_types=(matplotlib.axis.Axis, matplotlib.spines.Spine,))


def create_canvas(shape_x, shape_y, dpi, alpha=0., show_after_creation=False):
    """Generates an empty figure with x by y pixels

    The generated canvas can be used to assign markers and lines accurately to specific pixel locations.

    :param shape_x: Width of the canvas in pixels
    :type shape_x: int
    :param shape_y: Height of the canvas in pixels
    :type shape_y: int
    :param dpi: Resolution of the canvas. Determines the final width and height in inches.
    :type dpi: int
    :param alpha: How opaque the canvas will be. Default is transparent. Value between 0 and 1.
    :type alpha: float
    :param show_after_creation: Whether to show the canvas after creation. Default is False.
    :type show_after_creation: bool
    :return: plt.Figure of width x and height y (in pixels)
    :rtype: plt.Figure
    """
    if not show_after_creation:
        plt.ioff()
    fig = plt.figure(figsize=(shape_x / dpi, shape_y / dpi), dpi=dpi)
    fig.patch.set_alpha(alpha)

    ax = plt.Axes(fig, (0., 0., 1., 1.), xlim=(0, shape_x), ylim=(0, shape_y))
    ax.set_axis_off()
    ax.autoscale(False)
    ax.invert_yaxis()
    fig.add_axes(ax)
    if not show_after_creation:
        plt.ion()

    return fig


def copy_figure(input_fig):
    """Copy a matplotlib figure

    :param input_fig: Figure to be copied
    :type input_fig: plt.Figure
    :return: Copy of the input figure
    :rtype: plt.Figure
    """
    fig_copy = copy_object(input_fig)

    # from https://stackoverflow.com/a/54579616
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig_copy
    fig_copy.set_canvas(new_manager.canvas)
    return fig_copy


def plot_scatter_sequence(ax, position_array, color="blue", alpha=1., label=None):
    x, y = position_array
    ax.scatter(x, y, label=label, color=color, s=1, alpha=alpha)
    diff_array = np.pad(np.diff(position_array), ((0, 0), (0, 1)), mode='constant')
    ax.quiver(x, y, *diff_array, scale_units='xy', angles="xy", scale=1, color=color, alpha=alpha)


# from https://stackoverflow.com/q/55703105
def fig2data(fig):
    """
    Convert a Matplotlib figure to a numpy array and return it.

    :param fig: a matplotlib figure
    :type fig: plt.Figure
    :return: a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    return buf


def fig2img(fig):
    """
    Convert a Matplotlib figure to a PIL Image in RGBA format and return it.

    :param fig: a matplotlib figure
    :type fig: plt.Figure
    :return: a Python Imaging Library (PIL) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombuffer("RGBA", (w, h), buf)


def fig2cv2(fig):
    """Convert a Matplotlib figure to a RGBA numpy array that can be used as an image in OpenCV2

    :param fig: a matplotlib figure
    :type fig: plt.Figure
    :return: a NumPy array
    :rtype: np.ndarray
    """
    return np.array(fig2img(fig))


def get_colors_from_cmap(cmap_name_or_cmap, num_colors_or_positions=None):
    """
    Returns a list of colors from a matplotlib colormap, given the name of the colormap or the colormap itself and
    either the number of colors or a list of positions. Alternatively, if num_colors_or_positions is None, the default
    is to return cmap.N colors.

    :param cmap_name_or_cmap: The name of the colormap or the colormap itself
    :type cmap_name_or_cmap: str or matplotlib.colors.Colormap
    :param num_colors_or_positions: Either the number of colors to return or a list of positions between 0 and 1, where
                                    0 is the first color in the colormap and 1 is the last color in the colormap. If
                                    None, the default is to return cmap.N colors.
    :type num_colors_or_positions: int or list or None
    :return: A list of colors
    :rtype: list
    """

    if isinstance(cmap_name_or_cmap, str):
        cmap = plt.get_cmap(cmap_name_or_cmap)
    else:
        cmap = cmap_name_or_cmap
    if num_colors_or_positions is None:
        return cmap(np.arange(cmap.N))
    elif isinstance(num_colors_or_positions, int):
        return cmap(np.linspace(0, 1, num_colors_or_positions))
    else:
        return cmap(num_colors_or_positions)


def qq_plot(sample1, sample2, n_quantiles=100, ax=None):
    """
    This function plots a quantile-quantile plot of two samples. The quantiles of both samples are calculated and then
    plotted against each other. If the samples are from the same distribution, then the points should lie on the
    diagonal. If the samples are from different distributions, then the points should lie off the diagonal.

    :param sample1: An array of values, to be fed to np.quantile
    :type sample1: np.ndarray
    :param sample2: An array of values, to be fed to np.quantile
    :type sample2: np.ndarray
    :param n_quantiles: The number of quantiles to calculate
    :type n_quantiles: int
    :param ax: The matplotlib axis on which to plot the qq plot. If None, then a new figure and axis are created.
    :type ax: plt.Axes or None
    :return: The matplotlib axis on which the qq plot was plotted
    :rtype: plt.Axes
    """
    # Calculate the quantiles of both samples
    sample1_quantiles = np.quantile(sample1, np.linspace(0, 1, n_quantiles))
    sample2_quantiles = np.quantile(sample2, np.linspace(0, 1, n_quantiles))

    if ax is None:
        _fig, ax = plt.subplots()

    ax.plot(sample1_quantiles, sample2_quantiles)

    diag_corners = max(sample1_quantiles.min(), sample2_quantiles.min()), min(sample1_quantiles.max(),
                                                                              sample2_quantiles.max())
    ax.plot(diag_corners, diag_corners, color="k")
    return ax


def align_axes(*plt_axes, align_x=True, align_y=True):
    """
    Aligns the x and/or y axes of all given axes. The axes are aligned to the minimum and maximum values of all the
    axes in the list. If align_x is True, then the x axes are aligned. If align_y is True, then the y axes are aligned.

    :param plt_axes: A list of axes
    :type plt_axes: plt.Axes
    :param align_x: Whether to align the x axes
    :type align_x: bool
    :param align_y: Whether to align the y axes
    :type align_y: bool
    """
    for ax in plt_axes:
        if align_x:
            ax.set_xlim(np.min([ax.get_xlim() for ax in plt_axes]), np.max([ax.get_xlim() for ax in plt_axes]))
        if align_y:
            ax.set_ylim(np.min([ax.get_ylim() for ax in plt_axes]), np.max([ax.get_ylim() for ax in plt_axes]))


def align_axes_from_dict(plt_ax_dict, align_x=True, align_y=True):
    """
    Aligns the x and/or y axes of a dictionary of axes. The axes are aligned to the minimum and maximum values of all
    the axes in the dictionary under the same key. If align_x is True, then the x axes are aligned. If align_y is True,
    then the y axes are aligned.

    :param plt_ax_dict: A dictionary mapping keys to lists of axes
    :type plt_ax_dict: dict[str, list[plt.Axes]]
    :param align_x: Whether to align the x axes
    :type align_x: bool
    :param align_y: Whether to align the y axes
    :type align_y: bool
    """
    for _ax_key, axes in plt_ax_dict.items():
        align_axes(*axes, align_x=align_x, align_y=align_y)


def get_figure_size(fig, dpi=100):
    """
    Returns the size of a matplotlib figure in inches.
    From https://kavigupta.org/2019/05/18/Setting-the-size-of-figures-in-matplotlib/

    :param fig: The matplotlib figure
    :type fig: plt.Figure
    :param dpi: The resolution of the figure
    :type dpi: int
    :return: The width and height of the figure in inches
    :rtype: tuple
    """
    with TemporaryDirectory() as temp_dir:
        # Use a temporary directory instead of a temporary file due to Windows not permitting the file to be opened by
        # another process
        temp_file_path = pathlib.Path(temp_dir) / "temp.png"
        fig.savefig(temp_file_path, bbox_inches='tight', dpi=dpi)
        height, width, _channels = imread(temp_file_path).shape
        return width / dpi, height / dpi


def set_figure_size(fig, size, dpi=100, eps=1e-4, give_up=2, _min_size_px=10):
    """
    Ensures that the size of a matplotlib figure is exactly the given size or at least very close to it. This is done by
    iteratively adjusting the size of the figure until it is close enough to the given size. If the figure cannot be
    adjusted to the given size, then the figure is adjusted to the closest possible size. _min_size_px is the minimum
    size of the figure in pixels, this is useful to avoid infinite loops where the algorithm attempts to achieve a
    height or width of 0.
    Taken from https://kavigupta.org/2019/05/18/Setting-the-size-of-figures-in-matplotlib/

    :param fig: A matplotlib figure
    :type fig: plt.Figure
    :param size: The desired size of the figure
    :type size: tuple
    :param dpi: The resolution of the figure
    :type dpi: int
    :param eps: The tolerance of the size of the figure, in inches
    :type eps: float
    :param give_up: The number of iterations to wait before giving up if the size does not change
    :type give_up: int
    :param _min_size_px: The minimum size of the figure in pixels, along each axis
    :type _min_size_px: int
    :return: Whether the size of the figure was adjusted to the given size
    :rtype: bool
    """
    target_width, target_height = size
    set_width, set_height = target_width, target_height  # reasonable starting point
    deltas = []  # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_figure_size(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < _min_size_px or set_height * dpi < _min_size_px:
            return False


def save_figs_from_dict(fig_dict, ensure_size=False):
    """
    Saves a dictionary mapping file paths to figures. If ensure_size is True, then the size of the figure will be
    adjusted to fit the size given by fig.get_size_inches(). This solution might be a bit hacky, but it works.

    :param fig_dict:
    :type fig_dict:
    :param ensure_size:
    :type ensure_size:
    :return:
    :rtype:
    """
    for fig_path, fig in fig_dict.items():
        if ensure_size:
            set_figure_size(fig, fig.get_size_inches())
            fig.savefig(fig_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        else:
            fig.savefig(fig_path, facecolor=fig.get_facecolor(), edgecolor='none')


def calculate_rows_cols_from_minimum_plots(minimum_plots, col_row_ratio=1.0):
    """
    Calculates the n_rows and n_rows of a plot grid with a given minimum number of plots and n_rows to n_rows ratio.

    :param minimum_plots: The minimum number of plots in the grid
    :type minimum_plots: int
    :param col_row_ratio: The ratio of n_cols to n_rows, the result may not be exactly this ratio
    :type col_row_ratio: float
    :return: The number of n_rows and n_rows
    :rtype: (int, int)
    """
    n_rows = np.ceil(np.sqrt(minimum_plots / col_row_ratio))
    n_cols = np.round(col_row_ratio * n_rows)
    return int(n_rows), int(n_cols)


def save_matplotlib_animation(initial_figure, animation_func, out_path="animation.gif", save_kwargs=None,
                              **animation_kwargs):
    """
    Saves an animation to file. The animation is created by calling animation_func with the given arguments.
    Can also be used to save the animation to a video file by changing the writer in save_kwargs and the file extension
    in out_path. For example, to save to an mp4 file, set save_kwargs={"writer": "ffmpeg"} and out_path="animation.mp4".

    :param initial_figure: A matplotlib figure used to create the initial frame of the animation
    :type initial_figure: plt.Figure
    :param animation_func: A function that generates the frames of the animation
    :type animation_func: function
    :param out_path: The path to save the animation to. Default is "animation.gif"
    :type out_path: str
    :param save_kwargs: The arguments to pass to FuncAnimation.save
    :type save_kwargs: dict
    :param animation_kwargs: The arguments to pass to FuncAnimation
    :type animation_kwargs: Any
    :return: None
    """
    save_kwargs = {} if save_kwargs is None else save_kwargs
    default_save_kwargs = dict(writer="imagemagick", fps=30)
    save_kwargs = {**default_save_kwargs, **save_kwargs}

    default_animation_kwargs = dict(interval=1000 / save_kwargs["fps"], blit=True, repeat=True)
    animation_kwargs = {**default_animation_kwargs, **animation_kwargs}
    animation = FuncAnimation(initial_figure, animation_func, **animation_kwargs)

    animation.save(out_path, **save_kwargs)


def add_identity_line(axes, *line_args, **line_kwargs):
    """
    Adds a line to the given axes that represents the line y = x. The line is updated whenever the x or y limits of the
    axes change.

    A wrapper around add_xy_line that sets xy_factor to 1 with _axes_transform=False.

    :param axes: A matplotlib axes object
    :type axes: plt.Axes
    :param line_args: The arguments to pass to axes.plot
    :type line_args: tuple
    :param line_kwargs: The keyword arguments to pass to axes.plot
    :type line_kwargs: dict
    :return: The matplotlib axes object
    :rtype: plt.Axes
    """
    return add_xy_line(axes, xy_factor=1, _axes_transform=False, *line_args, **line_kwargs)


def add_xy_line(axes, xy_factor=1, y_shift=0, x_limit=None, y_limit=None, _axes_transform=False, *line_args,
                **line_kwargs):
    """
    Adds a line to the given axes that represents the line y = x * xy_factor. The line is updated whenever the x or y
    limits of the axes change. If _axes_transform is True, then the line is drawn in axes coordinates, otherwise it is
    drawn in data coordinates.

    Adapted from https://stackoverflow.com/a/28216751.

    :param axes: A matplotlib axes object
    :type axes: plt.Axes
    :param xy_factor: The factor by which to multiply the x values to get the y values
    :type xy_factor: float
    :param y_shift: The shift to add to the y values
    :type y_shift: float
    :param x_limit: The x limits of the line. If None, the x limits are based on the x limits of the axes or data.
    :type x_limit: tuple or None
    :param y_limit: The y limits of the line. If None, the y limits are based on the y limits of the axes or data.
    :type y_limit: tuple or None
    :param _axes_transform: Whether to transform the line to axes coordinates. Default is False.
    :type _axes_transform: bool
    :param line_args: The arguments to pass to axes.plot
    :type line_args: tuple
    :param line_kwargs: The keyword arguments to pass to axes.plot. If "transform" is in line_kwargs, it will be
                        overwritten based on the value of _axes_transform.
    :return: The matplotlib axes object
    :rtype: plt.Axes
    """

    _transform = axes.transAxes if _axes_transform else axes.transData
    line_kwargs = {**line_kwargs, "transform": _transform}
    xy_line, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(_axes):
        """
        A callback function that updates the line whenever the x or y limits of the axes change.

        :param _axes: A matplotlib axes object
        :type _axes: plt.Axes
        :return: None
        """
        if not _axes_transform:
            low_x, high_x = _axes.get_xlim()
            low_y, high_y = _axes.get_ylim()
        else:
            low_x, high_x = 0, 1
            low_y, high_y = 0, 1
        if x_limit is not None:
            low_x, high_x = x_limit
            low_y, high_y = low_x * xy_factor, high_x * xy_factor
        if y_limit is not None:
            low_y, high_y = y_limit
            low_x, high_x = low_y / xy_factor, high_y / xy_factor
        low = max(low_x, low_y / xy_factor)
        high = min(high_x, high_y / xy_factor)
        xy_line.set_data([low, high], [low * xy_factor + y_shift, high * xy_factor + y_shift])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def add_xy_annotation(axes, text, relative_position=0.9, xy_factor=1, y_shift=0, x_limit=None, y_limit=None,
                      _axes_transform=False, *text_args, **text_kwargs):
    """
    Add a text annotation on the line given by xy_factor and y_shift, relative to the line length, defined by
    relative_position. The line is either drawn in axes or data coordinates, depending on _axes_transform.

    :param axes: A matplotlib axes object
    :type axes: plt.Axes
    :param text: The text to add
    :type text: str
    :param relative_position: The position of the annotation, as a fraction of the length of the line
    :type relative_position: float
    :param xy_factor: The factor by which to multiply the x values to get the y values
    :type xy_factor: float
    :param y_shift: The shift to add to the y values
    :type y_shift: float
    :param x_limit: The x limits of the line. If None, the x limits are based on the x limits of the axes or data.
    :type x_limit: tuple or None
    :param y_limit: The y limits of the line. If None, the y limits are based on the y limits of the axes or data.
    :type y_limit: tuple or None
    :param _axes_transform: Whether to transform the reference line to axes coordinates. Default is False.
    :type _axes_transform: bool
    :param text_args: The arguments to pass to axes.text
    :type text_args: tuple
    :param text_kwargs: The keyword arguments to pass to axes.text. If "transform" is in line_kwargs, it will be
                        overwritten based on the value of _axes_transform, if "rotation_mode" is not in text_kwargs it
                        will be ignored and replaced with "anchor".
    :return: The matplotlib axes object
    :rtype: plt.Axes
    """

    default_text_kwargs = dict(va="bottom", ha="right")
    _transform = axes.transAxes if _axes_transform else axes.transData
    text_kwargs = {**default_text_kwargs, **text_kwargs, "transform": _transform, "rotation_mode": "anchor"}
    text = axes.text(0, 0, text, *text_args, **text_kwargs)

    def callback(_axes):
        """
        A callback function that updates the line whenever the x or y limits of the axes change.

        :param _axes: A matplotlib axes object
        :type _axes: plt.Axes
        :return: None
        """
        if not _axes_transform:
            low_x, high_x = _axes.get_xlim()
            low_y, high_y = _axes.get_ylim()
        else:
            low_x, high_x = 0, 1
            low_y, high_y = 0, 1
        if x_limit is not None:
            low_x, high_x = x_limit
            low_y, high_y = low_x * xy_factor, high_x * xy_factor
        if y_limit is not None:
            low_y, high_y = y_limit
            low_x, high_x = low_y / xy_factor, high_y / xy_factor
        low = max(low_x, low_y / xy_factor)
        high = min(high_x, high_y / xy_factor)
        line_start, line_end = np.array([low, low * xy_factor + y_shift]), np.array([high, high * xy_factor + y_shift])
        line_vector = line_end - line_start
        text_location = line_start + relative_position * line_vector
        text_rotation = np.arctan2(line_vector[1], line_vector[0])
        text.set_position(text_location)
        text.set_rotation(np.rad2deg(text_rotation))

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def rainbow_text(source_text, ls, lc):
    """
    Adapted from https://stackoverflow.com/a/9185851 and
    https://discourse.matplotlib.org/t/partial-coloring-of-text-in-matplotlib/16551/14. To simplify the task of coloring
    text, first create a text object with the desired properties, then create a list of subtexts with the desired
    colors. This function will then create an AnnotationBbox with the subtexts and add it to the axes, replacing the
    original, uncolored text. Replacing means that the original text will be set to alpha=0 and the new text will be
    added on top of it.
    This works so far with any horizontal text (even multiline), but not with vertical text, or any rotated text.

    :param source_text: The original text object
    :type source_text: matplotlib.text.Text
    :param ls: A list of strings to split the text into
    :type ls: list[str]
    :param lc: A list of colors to assign to each string
    :type lc: list[str]
    :return: The AnnotationBbox containing the colored text
    :rtype: matplotlib.offsetbox.AnnotationBbox
    """

    # try to extract as many features as possible from source text (size, font, etc.)
    source_text_props = {k: getattr(source_text, f"get_{k}")() for k in get_func_kwargs(Text) if
                         hasattr(source_text, f"get_{k}") and k not in ("text", "color")}

    # build lines
    sub_lines = []
    sub_texts = []
    for s, c in zip(ls, lc):
        for i, line in enumerate(s.split("\n")):
            if i > 0:
                sub_lines.append(sub_texts)
                sub_texts = []
            sub_texts.append(TextArea(line, textprops={"color": c, **source_text_props}))

    # remember to append the last line
    if sub_texts:
        sub_lines.append(sub_texts)

    # a vertical text is one that is rotated by exactly 90 degrees (e.g. y-axis labels)
    # if the source_text is vertical, try hacky solution... just pack the lines vertically and whole text horizontally.
    # invert order of subtexts for vertical text because 90 degrees means left->right == bottom->top
    if source_text.get_rotation() == 90:
        sub_lines = [VPacker(children=s[::-1], align="center") for s in sub_lines]
        box_packer = HPacker
    else:
        sub_lines = [HPacker(children=s, align="center") for s in sub_lines]
        box_packer = VPacker

    # pack the text from the individual lines, then hide the original text and add the new rainbow text
    bbox = AnnotationBbox(box_packer(children=sub_lines, align="center"), xy=(0, 0), xybox=(0.5, 0.5),
                          xycoords=source_text, frameon=False)

    # if we would only want single-line & horizontal texts, then this would be far easier, just the lines below
    # sub_texts = [TextArea(s, textprops={"color":c, **source_text_props}) for s, c in zip(ls, lc)]
    # text_line = HPacker(children=sub_texts, align="center")
    # bbox = AnnotationBbox(text_line, xy=(0, 0), xybox=(0.5, 0.5), xycoords=source_text, frameon=False)

    source_text.set_alpha(0)
    source_text.get_figure().add_artist(bbox)

    return bbox


def get_function_added_artists(func, *args, reference_figure=None, return_func_return=False, **kwargs):
    """
    Calls a function with the given arguments and keyword arguments and returns the artists that were added to the
    reference_figure. Will only work if the function only adds artists to the reference_figure.
    To avoid unexpected output, the axis being plotted to should be already initialized before calling this function, 
    as otherwise there will also be all artists included that are generated on the first initialization of an axis.

    :param func: The function to call
    :type func: function
    :param args: The arguments to pass to the function
    :type args: tuple
    :param return_func_return: Whether to return the return value of the function. If True, the return value will be
    a tuple of the return value of the function and then the artists added to the figure. Default is False.
    :type return_func_return: bool
    :param kwargs: The keyword arguments to pass to the function
    :type kwargs: dict
    :return: A list of artists that were added to the current figure or a tuple of the return value of the function and
    the artists added to the figure
    :rtype: tuple or list
    """
    if reference_figure is None:
        reference_figure = plt.gcf()

    previous_elements = get_plot_elements(reference_figure)
    func_return = func(*args, **kwargs)
    added_elements = [element for element in get_plot_elements(reference_figure) if element not in previous_elements]

    if return_func_return:
        return func_return, added_elements
    else:
        return added_elements


def multiply_ax_limits(ax, limit_factor):
    """
    Multiplies the x and y limits of the given axes by the given factors. The factors can be given as either a single
    number or a tuple of two numbers. The factors should be greater than 0.
    If the factors are less than 1, then the limits will be reduced, if the factors are greater than 1, then the
    limits will be increased.

    :param ax: The matplotlib axes object
    :type ax: plt.Axes
    :param limit_factor: The factor by which to multiply the limits. If a single number, then both the x and y limits
    will be multiplied by this number. If a tuple, then the x limits will be multiplied by the first number and the y
    limits will be multiplied by the second number.
    :type limit_factor: int or float or tuple
    :return: None
    """
    if isinstance(limit_factor, (int, float)):
        x_factor = y_factor = limit_factor
    else:
        x_factor, y_factor = limit_factor

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    x_span, y_span = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]
    ax.set_xlim(x_lim[0] - x_span * (x_factor - 1) / 2, x_lim[1] + x_span * (x_factor - 1) / 2)
    ax.set_ylim(y_lim[0] - y_span * (y_factor - 1) / 2, y_lim[1] + y_span * (y_factor - 1) / 2)


def get_ticklabel_position_dict(input_axis):
    """
    Return a dict mapping each ticklabel to its position on the given axis (x or y) .

    :param input_axis: A matplotlib axis object.
    :type input_axis: matplotlib.axis.XAxis or YAxis
    :return: A dictionary mapping each tick label to its position on the axis.
    :rtype: dict[str, float]
    """
    return {ticklabel.get_text(): tick for tick, ticklabel in zip(input_axis.get_majorticklocs(), input_axis.get_majorticklabels())}