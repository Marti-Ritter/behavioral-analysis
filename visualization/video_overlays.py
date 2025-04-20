from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import is_color_like
from shapely.affinity import affine_transform
from shapely.ops import nearest_points

from ..lmt_tools.lmt_mask_funcs import LmtDetection
from .matplotlib_funcs import fig2cv2, get_colors_from_cmap
from .networkx_funcs import plot_undirected_pairwise_relationship_graph, plot_connections_from_pair_dict
from .plots import plot_polygons_to_ax
from ..math_tools.geometry_funcs import create_ellipsis, bbox_to_polygon, scale_and_pad_bbox, create_box, \
    get_bbox, get_rois_from_bbox_list
from ..math_tools.matrix_funcs import apply_transform_matrix
from ..socialscan_tools.data_processing import filter_split_pd
from ..tracking_tools.funcs import apply_transform_matrix_to_track_df
from ..tracking_tools.plots import plot_keypoint_instance
from ..visualization.matplotlib_funcs import get_function_added_artists

"""
                     |      |        |                           |           |
                    fig     |        |                           |           |
                     |      |        |                         video         |
                     |     data      |                           |           |
                     |      |        |                           |           |
                     |      |      colors                        |       parameters
                     |      |        |                           |           |
                     v      v        v                           v           v
                  +--------------------------+              +------------------------+
                  |*Overlay                  |              |show_video_in_notebook  |
                  +--------------------------+              +------------------------+
                  |Collect data, original    |              |Show a stream of frames |
                  |matplotlib figure and     |              |based on the given video|
                  |plotting parameters       |              |and parameters          |
                  |                          |              |                        |
                  +--------------------------+              +------------------------+
                  |get_frame_overlay(index)  +------------->|rgba_overlay_functions  |
                  |Return an array repre-    |              |Functions that can be   |
                  |senting the loaded data   |              |called to generate a    |
                  |at the given frame index  |              |figure that will be     |
                  |plotted to the input fig. |              |added to the stream.    |
                  |Can be added to CV2 video.|              |                        |
                  +--------------------------+              +------------+-----------+
                                                                         |
                                                                         v
                                                           stream of frames with overlay
"""


def get_label_positions(label_count, space_limits):
    label_positions = np.linspace(*space_limits, label_count + 1)
    return (label_positions - (label_positions[1] - label_positions[0]) / 2)[1:]


class FrameOverlayBase(ABC):
    """
    Base class for all frame overlays. Contains the initial figure and the data to be plotted, as well as any further
    parameters. The get_frame_overlay method should be called to get the frame overlay for a given frame index.
    Any artists added during the overlay process should be removed in the "finally" block of the get_frame_overlay
    method. The add_data_to_frame method should be implemented to add the data to the frame.
    """
    def __init__(self, initial_fig, data_dict, **parameters):
        """
        :param initial_fig: The initial figure that all created frame overlays are based on.
        Should be an empty canvas of the correct size to plot the data to a video frame.
        :type initial_fig: matplotlib.pyplot.Figure
        :param data_dict: A dict of data, for example pandas objects, from which the frame overlay is created.
        :type data_dict: dict
        :param parameters: Any parameters that are used in the subclass to create the overlay.
        :type parameters: Any
        """
        self.initial_fig = initial_fig
        self.data_dict = data_dict
        self.parameters = parameters
        self.store = self.preprocessing()

    def get_frame_overlay(self, frame_index):
        """
        Get the frame overlay for the given frame index. Any artists added during the overlay process will be removed
        afterward.

        :param frame_index: The index of the given frame, based on the values in the data_dict.
        :type frame_index: int
        :return: A cv2 compatible image of the frame overlay.
        :rtype: numpy.ndarray
        """
        new_artists = []
        try:
            new_artists = self.add_data_to_frame(frame_index)
            return fig2cv2(self.initial_fig)

        finally:
            for new_artist in new_artists:
                new_artist.remove()

    def preprocessing(self):
        """
        Preprocessing step that is called during initialization. Can be used to store data that is used in
        multiple frames. Should return the finished store-dict.
        """
        return {}

    def add_data_to_frame(self, frame_index):
        """
        Add the data to the frame. Should return a list of all artists that are added to the frame.

        :param frame_index:
        :type frame_index:
        :return:
        :rtype:
        """
        raise NotImplementedError("Subclass must implement abstract method")


class HeatmapOverlay(FrameOverlayBase):
    """
    Overlay a heatmap to the frame. The heatmap is scaled to the extent of the figure.
    """

    def __init__(self, initial_fig, data_dict, extent=None, cmap=None, vmin=None, vmax=None, alpha=0.5,
                 interpolation_mode=0, heatmap_alpha_span=(0, 1), **parameters):
        """
        :param initial_fig: The initial figure that all created frame overlays are based on.
        :type initial_fig: matplotlib.pyplot.Figure
        :param data_dict: A dict mapping frame indices to heatmaps.
        :type data_dict: dict
        :param extent: The extent of the heatmap. If None, the extent of the initial_fig's current axis will be used.
        :type extent: tuple
        :param cmap: The colormap to use for the heatmap. Defaults to matplotlib's "viridis".
        :type cmap: matplotlib.colors.Colormap or str
        :param vmin: The minimum value of the heatmap. If None, the minimum value of the heatmap per frame will be used.
        :type vmin: float
        :param vmax: The maximum value of the heatmap. If None, the maximum value of the heatmap per frame will be used.
        :type vmax: float
        :param alpha: The alpha value of the heatmap. If None, the alpha value will be calculated based on the
            heatmap_alpha_span and the heatmap value (bounded by vmin and vmax).
        :type alpha: float or None
        :param resampling_filter: The interpolation mode of the heatmap. Defaults to 0 (nearest). See documentation of
            PIL.Image for more information.
        :type resampling_filter: int or one of the PIL resampling filters
        :param heatmap_alpha_span: The relative span of the heatmap values that will be mapped to the alpha value.
            For example, if heatmap_alpha_span is (0, 1), the alpha value will be 0 for the minimum value of the
            heatmap and 1 for the maximum value of the heatmap.
        :type heatmap_alpha_span: tuple of floats
        :param parameters: Any parameters that are used in the subclass to create the overlay.
        :type parameters: Any
        """
        super().__init__(initial_fig, data_dict, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
                         interpolation_mode=interpolation_mode, heatmap_alpha_span=heatmap_alpha_span, **parameters)

    def preprocessing(self):
        from PIL import Image
        from matplotlib.colors import ListedColormap

        store = {}
        if self.parameters["extent"] is None:
            self.parameters["extent"] = (*self.initial_fig.gca().get_xlim(), *self.initial_fig.gca().get_ylim())

        extent = self.parameters["extent"]

        # resize heatmaps to fit the figure axis
        target_width = abs(int(extent[1] - extent[0]))
        target_height = abs(int(extent[3] - extent[2]))

        for frame, heatmap_data in self.data_dict.items():
            heatmap = np.array(Image.fromarray(heatmap_data).resize((target_width, target_height),
                                                                    resample=self.parameters["interpolation_mode"]))
            store[frame] = heatmap

        # calculate cmap depending on if alpha is None
        heatmap_cmap = get_colors_from_cmap(self.parameters["cmap"], num_colors_or_positions=None)
        if self.parameters["alpha"] is None:
            # approach from https://stackoverflow.com/a/37334212
            heatmap_cmap[:, -1] = np.linspace(*self.parameters["heatmap_alpha_span"], len(heatmap_cmap))
        heatmap_cmap = ListedColormap(heatmap_cmap)
        store["heatmap_cmap"] = heatmap_cmap

        return store

    def add_data_to_frame(self, frame_index):
        heatmap = self.initial_fig.gca().imshow(self.store[frame_index], extent=self.parameters["extent"],
                                                cmap=self.store["heatmap_cmap"], vmin=self.parameters["vmin"],
                                                vmax=self.parameters["vmax"], alpha=self.parameters["alpha"])
        return [heatmap]


class KeyPointOverlay(FrameOverlayBase):
    """
    Overlay that plots keypoints to the frame. The keypoints are stored in a pandas DataFrame with the columns "x" and
    "y". The index of the DataFrame should be the frame index. The keypoint parameters are a dict of dicts, where the
    keys are the names of the keypoints and the values are dicts of parameters that are passed to the plot function. The
    skeleton parameters are a dict of dicts, where the keys are tuples of the names of the keypoints that are connected
    by the skeleton and the values are dicts of parameters that are passed to the plot function.

    Important note: The current implementation leads to the alpha keyword argument being ignored if it is not 0 or 1.
    """
    def __init__(self, initial_fig, track_df, transform=None, skeleton_df=None, plot_labels=False,
                 keypoint_kwargs=None, skeleton_kwargs=None, label_kwargs=None, shared_kwargs=None):
        data_dict = dict(track_df=track_df, skeleton_df=skeleton_df)

        if shared_kwargs is None:
            shared_kwargs = {}
        
        super().__init__(initial_fig, data_dict, transform=transform, plot_labels=plot_labels, keypoint_kwargs=keypoint_kwargs, 
                         skeleton_kwargs=skeleton_kwargs, label_kwargs=label_kwargs, shared_kwargs=shared_kwargs)

    def preprocessing(self):
        store_dict = {}
        transform = self.parameters["transform"]
        
        if transform is not None:
            track_df = apply_transform_matrix_to_track_df(self.data_dict["track_df"], transform)
        else:
            track_df = self.data_dict["track_df"]
        store_dict["keypoint_df"] = track_df.stack("keypoint_name", future_stack=True)
        store_dict["track_identifiers"] = track_df.index.names[:-1]

        store_dict["skeleton_df"] = self.data_dict["skeleton_df"]
        store_dict["skeleton_identifiers"] = self.data_dict["skeleton_df"].index.names[:-1]

        if store_dict["skeleton_identifiers"]:
            assert store_dict["track_identifiers"] == store_dict["skeleton_identifiers"], "Track and skeleton identifiers do not match"

        return store_dict

    def add_data_to_frame(self, frame_index):
        new_artists = []

        for track_identifier, single_track_keypoints in self.store["keypoint_df"].groupby(self.store["track_identifiers"]):
            single_track_keypoints = single_track_keypoints.droplevel(level=self.store["track_identifiers"], axis=0)

            if frame_index not in single_track_keypoints.index:
                continue

            if self.store["skeleton_identifiers"]:
                if track_identifier in self.store["skeleton_df"].index:
                    skeleton_df = self.store["skeleton_df"].loc[track_identifier]
                else:
                    skeleton_df = None
            else:
                skeleton_df = self.store["skeleton_df"]

            instance_artists = get_function_added_artists(
                plot_keypoint_instance, keypoint_df=single_track_keypoints.loc[frame_index], skeleton_df=skeleton_df, 
                plot_labels=self.parameters["plot_labels"], keypoint_kwargs=self.parameters["keypoint_kwargs"], 
                skeleton_kwargs=self.parameters["skeleton_kwargs"], label_kwargs=self.parameters["label_kwargs"], 
                ax=self.initial_fig.gca(), **self.parameters["shared_kwargs"], 
                return_func_return=False, reference_figure=self.initial_fig)
            new_artists.extend(instance_artists)
        return new_artists



class KeyPointOverlayOld(FrameOverlayBase):
    """
    Overlay that plots keypoints to the frame. The keypoints are stored in a pandas DataFrame with the columns "x" and
    "y". The index of the DataFrame should be the frame index. The keypoint parameters are a dict of dicts, where the
    keys are the names of the keypoints and the values are dicts of parameters that are passed to the plot function. The
    skeleton parameters are a dict of dicts, where the keys are tuples of the names of the keypoints that are connected
    by the skeleton and the values are dicts of parameters that are passed to the plot function.
    """
    def __init__(self, initial_fig, data_dict, keypoint_color="k", transform=None,
                 keypoint_parameters=None, skeleton_parameters=None):
        if isinstance(keypoint_color, dict):
            keypoint_color_dict = keypoint_color
        elif is_color_like(keypoint_color):
            keypoint_color_dict = {keypoint: keypoint_color for keypoint in data_dict["keypoints"].keys()}
        else:
            raise ValueError("keypoint_color is not a valid color and not a dict")
        super().__init__(initial_fig, data_dict, keypoint_color_dict=keypoint_color_dict, transform=transform,
                         keypoint_parameters=keypoint_parameters, skeleton_parameters=skeleton_parameters)

    def preprocessing(self):
        store_dict = {}
        transform = self.parameters["transform"]

        for keypoint, position_df in self.data_dict["keypoints"].items():
            position_array = position_df[["x", "y"]].values
            store_dict[keypoint] = apply_transform_matrix(position_array, transform
                                                          ) if transform is not None else position_array
        return store_dict

    def add_data_to_frame(self, frame_index):
        new_artists = []

        for keypoint, position_array in self.store.items():
            plot_values = position_array[frame_index, :]
            keypoint_parameters = self.parameters["keypoint_parameters"]
            keypoint_parameters = keypoint_parameters if keypoint_parameters is not None else {}
            keypoint_parameters["color"] = self.parameters["keypoint_color_dict"][keypoint]
            new_artists.append(self.initial_fig.gca().scatter(*plot_values.T, **keypoint_parameters))

        skeleton = self.data_dict["skeleton"]
        for skeleton_line in skeleton:
            keypoint_start, keypoint_end = skeleton_line
            start_position, end_position = self.store[keypoint_start][frame_index, :], self.store[keypoint_end][
                                                                                       frame_index, :]
            skeleton_parameters = self.parameters["skeleton_parameters"]
            skeleton_parameters = skeleton_parameters if skeleton_parameters is not None else {}
            new_artists.extend(self.initial_fig.gca().plot(*np.vstack((start_position, end_position)).T,
                                                           **skeleton_parameters))
        return new_artists


class MultiIndividualKeyPointOverlay(FrameOverlayBase):
    def __init__(self, initial_fig, data_dict, individual_color=None, transform=None, keypoint_parameters=None,
                 skeleton_parameters=None):
        if individual_color is None:
            individual_color_dict = {individual: plt.get_cmap("tab10")(i) for i, individual in
                                     enumerate(data_dict["individuals"].keys())}
        elif isinstance(individual_color, str):
            individual_color_dict = {individual: individual_color for individual in data_dict["individuals"].keys()}
        elif isinstance(individual_color, dict):
            individual_color_dict = individual_color
        else:
            raise ValueError("individual_color must be a string or a dict")
        super().__init__(initial_fig, data_dict, individual_color_dict=individual_color_dict, transform=transform,
                         keypoint_parameters=keypoint_parameters, skeleton_parameters=skeleton_parameters)

    @classmethod
    def from_sleap_path(cls, initial_fig, sleap_tracking_h5_or_csv_path, individual_color=None, transform=None,
                        keypoint_parameters=None, skeleton_parameters=None):
        from ..tracking_tools.sleap_funcs import get_single_sleap_tracking_df
        from ..tracking_tools.sleap_funcs import get_sleap_analysis_h5_skeleton_df
        track_df = get_single_sleap_tracking_df(sleap_tracking_h5_or_csv_path)

        data_dict = dict(skeleton=get_sleap_analysis_h5_skeleton_df(sleap_tracking_h5_or_csv_path).values,
                         individuals={})
        for individual, individual_track_df in track_df.groupby("track"):
            individual_track_df = individual_track_df.droplevel("track", axis=0)
            track_keypoints = {k: v.unstack(level="subcol") for k, v in individual_track_df.drop(
                "score", axis=1, level=1).stack(level="subcol", future_stack=True).items()}
            data_dict["individuals"][individual] = dict(keypoints=track_keypoints)
        return cls(initial_fig, data_dict, individual_color=individual_color, transform=transform,
                   keypoint_parameters=keypoint_parameters, skeleton_parameters=skeleton_parameters)

    def preprocessing(self):
        store_dict = {}

        for individual, individual_data_dict in self.data_dict["individuals"].items():
            if "skeleton" not in individual_data_dict:
                if "skeleton" in self.data_dict:
                    individual_data_dict["skeleton"] = self.data_dict["skeleton"]
                else:
                    raise ValueError("No skeleton found in data_dict")

            store_dict[individual] = KeyPointOverlayOld(self.initial_fig, data_dict=individual_data_dict,
                                                     keypoint_color=self.parameters["individual_color_dict"][
                                                         individual],
                                                     transform=self.parameters["transform"],
                                                     keypoint_parameters=self.parameters["keypoint_parameters"],
                                                     skeleton_parameters=self.parameters["skeleton_parameters"])
        return store_dict

    def add_data_to_frame(self, frame_index):
        new_artists = []

        for individual, individual_overlay in self.store.items():
            new_artists.extend(individual_overlay.add_data_to_frame(frame_index))

        return new_artists

class TraceOverlay(FrameOverlayBase):
    """
    Overlay that plots a trace of the given data. The data_dict should contain a pandas DataFrame for each tag, with the
    x and y coordinates in the columns ("center", "x") and ("center", "y"). The color_dict should contain a color for
    each tag. The transform parameter can be used to transform the data before plotting. The trace_tail parameter can be
    used to set the length of the trace.
    """
    def __init__(self, initial_fig, data_dict, color_input_dict, transform=None, trace_tail=100):
        super().__init__(initial_fig, data_dict,
                         color_dict=color_input_dict, transform=transform, trace_tail=trace_tail)

    def preprocessing(self):
        store_dict = {}
        transform = self.parameters["transform"]
        for tag, trace_df in self.data_dict.items():
            trace_positions = trace_df[[("center", "x"), ("center", "y")]].values
            store_dict[tag] = apply_transform_matrix(trace_positions,
                                                     transform) if transform is not None else trace_positions
        return store_dict

    def add_data_to_frame(self, frame_index):
        new_artists = []
        for tag, trace_positions in self.store.items():
            plot_values = trace_positions[frame_index - self.parameters["trace_tail"]:frame_index, :]
            new_artists.extend(self.initial_fig.gca().plot(*plot_values.T, color=self.parameters["color_dict"][tag]))
        return new_artists


class RfidOverlay(FrameOverlayBase):
    """
    Overlay that plots the active rfid polygons for each frame.
    """
    def __init__(self, initial_fig, data_dict, color_input_dict, rfid_polygons, rfid_persistence=10):
        super().__init__(initial_fig, data_dict,
                         color_dict=color_input_dict, rfid_polygons=rfid_polygons, rfid_persistence=rfid_persistence)

    def add_data_to_frame(self, frame_index):
        new_artists = []
        for mouse, rfid_series in self.data_dict.items():
            rfid_events = rfid_series.loc[frame_index - self.parameters["rfid_persistence"]:frame_index]
            if rfid_events.empty:
                continue

            new_artists.extend(plot_polygons_to_ax(self.initial_fig.gca(),
                                                   {rfid_event: self.parameters["rfid_polygons"][rfid_event] for
                                                    rfid_event in rfid_events},
                                                   polygon_color=self.parameters["color_dict"][mouse]))
        return new_artists


class AnnotationOverlay(FrameOverlayBase):
    """
    Overlay that plots the tag annotations for each frame. If only_labels is set to True, only the labels will be
    plotted, without the arrows.
    """
    def __init__(self, initial_fig, data_dict, color_input_dict, only_labels=False, transform=None):
        super().__init__(initial_fig, data_dict,
                         color_dict=color_input_dict, only_labels=only_labels, transform=transform)

    def preprocessing(self):
        store_dict = {}

        x_positions = get_label_positions(len(self.data_dict), self.initial_fig.gca().get_xlim())
        y_position = 0.025 * self.initial_fig.gca().get_ylim()[0]
        store_dict["label_positions"] = {tag: (x_position, y_position) for tag, x_position in
                                         zip(self.data_dict.keys(), x_positions)}

        transform = self.parameters["transform"]
        store_dict["trace_positions"] = {}

        for tag, trace_df in self.data_dict.items():
            trace_positions = trace_df[[("center", "x"), ("center", "y")]].values
            store_dict["trace_positions"][tag] = pd.DataFrame(data=apply_transform_matrix(
                trace_positions, transform) if transform is not None else trace_positions,
                                                              index=trace_df.index)

        return store_dict

    def add_data_to_frame(self, frame_index):
        color_dict = self.parameters["color_dict"]
        label_positions = self.store["label_positions"]
        trace_positions_dict = self.store["trace_positions"]

        new_artists = []
        for tag, trace_positions in trace_positions_dict.items():
            if frame_index not in trace_positions.index:
                continue

            arrow_position = label_positions[tag] if any(np.isnan(trace_positions.loc[frame_index])) or self.parameters[
                "only_labels"] else trace_positions.loc[frame_index]

            new_artists.append(
                self.initial_fig.gca().annotate(tag, xy=arrow_position, xytext=label_positions[tag],
                                                ha="center", va="top",
                                                bbox=dict(facecolor=color_dict[tag], edgecolor='none'),
                                                arrowprops=dict(arrowstyle="->", color=color_dict[tag],
                                                                shrinkA=5, shrinkB=5)))
        return new_artists


class DirectionOverlay(FrameOverlayBase):
    """
    Overlay that plots the direction of the given data. The data_dict should contain a pandas DataFrame for each tag,
    with x and y denoting the center of the tag, and x_nose, and y_nose denoting the position of the "point" of the
    arrow,
    """
    def __init__(self, initial_fig, data_dict, color_input_dict, transform=None, arrow_kwargs=None):
        super().__init__(initial_fig, data_dict,
                         color_dict=color_input_dict, transform=transform)
        self.arrow_kwargs = dict({"length_includes_head": True, "width": 4, "edgecolor": "k", "linewidth": 0.1},
                                 **(arrow_kwargs if arrow_kwargs is not None else {}))

    def preprocessing(self):
        transform = self.parameters["transform"]

        store_dict = {"center_positions": {}, "nose_positions": {}}
        for tag, direction_df in self.data_dict.items():
            center_positions = direction_df[[("center", "x"), ("center", "y")]].values
            nose_positions = direction_df[[("nose", "x"), ("nose", "y")]].values

            store_dict["center_positions"][tag] = pd.DataFrame(data=apply_transform_matrix(
                center_positions, transform) if transform is not None else center_positions,
                                                               index=direction_df.index)
            store_dict["nose_positions"][tag] = pd.DataFrame(data=apply_transform_matrix(
                nose_positions, transform) if transform is not None else nose_positions,
                                                             index=direction_df.index)

        return store_dict

    def add_data_to_frame(self, frame_index):
        center_positions_dict = self.store["center_positions"]
        nose_positions_dict = self.store["nose_positions"]

        new_artists = []
        for tag, center_positions in center_positions_dict.items():
            nose_positions = nose_positions_dict[tag]
            if frame_index not in center_positions.index or frame_index not in nose_positions.index:
                continue

            center_position = center_positions.loc[frame_index]
            nose_position = nose_positions.loc[frame_index]

            new_artists.append(
                self.initial_fig.gca().arrow(*center_position, *(nose_position - center_position),
                                             facecolor=self.parameters["color_dict"][tag],
                                             **self.arrow_kwargs))

        return new_artists


class TraceFilterOverlay(FrameOverlayBase):
    """
    Overlay that plots the trace of the given data, with the option to filter the data. The data_dict should contain two
    columns x and y, and be able to be filtered by the filter_function. The filter_function should take a pandas
    DataFrame as input and return a boolean Series. The filter_color is the color of the filtered data.
    """
    def __init__(self, initial_fig, data_dict, filter_function, color_input_dict, filter_color="white",
                 transform=None, trace_tail=100):
        split_data_dict = {tag: dict(
            zip(["remaining", "filtered"], filter_split_pd(input_df, filter_function(input_df)))
        ) for tag, input_df in data_dict.items()}

        super().__init__(initial_fig, split_data_dict,
                         color_dict=color_input_dict, filter_function=filter_function, filter_color=filter_color,
                         transform=transform, trace_tail=trace_tail)

    def add_data_to_frame(self, frame_index):
        remaining_trace_positions_dict, filtered_trace_positions_dict = [
            {tag: trace_dfs[key].loc[
                  frame_index - self.parameters["trace_tail"]:frame_index, [("center", "x"), ("center", "y")]]
             for tag, trace_dfs in self.data_dict.items()} for key in ["remaining", "filtered"]]

        new_artists = []
        for tag, trace_positions in remaining_trace_positions_dict.items():
            plot_values = apply_transform_matrix(trace_positions.values,
                                                 self.parameters["transform"]
                                                 ) if self.parameters[
                                                          "transform"] is not None else trace_positions.values
            new_artists.extend(self.initial_fig.gca().plot(*plot_values.T, color=self.parameters["color_dict"][tag]))
        for tag, trace_positions in filtered_trace_positions_dict.items():
            plot_values = apply_transform_matrix(trace_positions.values,
                                                 self.parameters["transform"]
                                                 ) if self.parameters[
                                                          "transform"] is not None else trace_positions.values
            new_artists.extend(self.initial_fig.gca().plot(*plot_values.T, color=self.parameters["filter_color"]))
        return new_artists


class DistanceOverlay(FrameOverlayBase):
    """
    Overlay that plots the distance between the given data. The data_dict should contain a pandas DataFrame for each
    tag, with two columns x and y that are used to calculate the pairwise distances between the tags.
    """
    def __init__(self, initial_fig, data_dict, color_input_dict, transform=None, trace_tail=100):
        super().__init__(initial_fig, data_dict,
                         color_dict=color_input_dict, transform=transform, trace_tail=trace_tail)

    def preprocessing(self):
        store_dict = {}
        transform = self.parameters["transform"]
        store_dict["raw_positions"], store_dict["transformed_positions"] = {}, {}
        for tag, trace_df in self.data_dict.items():
            raw_positions = trace_df[[("center", "x"), ("center", "y")]].values
            store_dict["raw_positions"][tag] = raw_positions
            store_dict["transformed_positions"][tag] = apply_transform_matrix(
                raw_positions, transform) if transform is not None else raw_positions

        return store_dict

    def add_data_to_frame(self, frame_index):
        relevant_tags = {tag: position for tag, position in self.store["raw_positions"].items() if
                         not np.isnan(position).any()}

        new_artists = plot_undirected_pairwise_relationship_graph(list(relevant_tags.values()),
                                                                  lambda a, b: np.sqrt(
                                                                      (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2),
                                                                  positions=[self.store["transformed_positions"][tag]
                                                                             for tag in relevant_tags.keys()],
                                                                  ax=self.initial_fig.gca(),
                                                                  outcome_labels=lambda x: str(
                                                                      round(x, 2)) + "mm",
                                                                  outcome_widths=0.5, outcome_colors="beige",
                                                                  label_font_size=10, input_labels=[],
                                                                  node_colors=[self.parameters["color_dict"][tag] for
                                                                               tag in relevant_tags.keys()],
                                                                  node_sizes=50)[1]

        return new_artists


class OutlineOverlay(FrameOverlayBase):
    """
    Overlay that plots the outline of the given data. The data_dict should contain a pandas DataFrame for each tag,
    with an x and y column, an orientation column, and two columns for the axes of the ellipse/box, depending on the
    outline_mode.
    """

    def __init__(self, initial_fig, data_dict, color_input_dict, outline_mode="ellipsis", add_direction=False,
                 transform=None, polygon_kwargs=None):
        super().__init__(initial_fig, data_dict,
                         color_dict=color_input_dict, transform=transform, outline_mode=outline_mode,
                         add_direction=add_direction)
        self.polygon_kwargs = dict({"plot_labels": False, "line_width": 1.},
                                   **(polygon_kwargs if polygon_kwargs is not None else {}))

    def add_data_to_frame(self, frame_index):
        trace_positions_dict = {tag: trace_df.loc[frame_index, [("center", "x"), ("center", "y")]] for tag, trace_df in
                                self.data_dict.items() if frame_index in trace_df.index}

        relevant_tags = {tag: position.values for tag, position in trace_positions_dict.items() if
                         not position.isna().any() and not position.empty}

        poly_params_dict = {
            tag: self.data_dict[tag][[("axes", "0"), ("axes", "1")]].join(
                self.data_dict[tag]["orientation"]).loc[frame_index] for tag in relevant_tags.keys()}

        new_artists = []
        for tag, position in relevant_tags.items():
            poly_params = poly_params_dict[tag]
            orientation, axis0, axis1 = poly_params["orientation"], poly_params[("axes", "0")], poly_params[
                ("axes", "1")]
            if self.parameters["outline_mode"] == "box":
                poly = create_box(position, orientation, axis0, axis1)
            elif self.parameters["outline_mode"] == "ellipsis":
                poly = create_ellipsis(position, orientation, axis0, axis1)
            else:
                raise ValueError("Invalid outline mode")

            transform = self.parameters["transform"]
            poly = affine_transform(poly, [transform[0, 0], transform[0, 1],
                                           transform[1, 0], transform[1, 1],
                                           transform[0, 2], transform[1, 2]]
                                    ) if transform is not None else poly

            new_artists.extend(
                plot_polygons_to_ax(self.initial_fig.gca(), {tag: poly},
                                    polygon_color=self.parameters["color_dict"][tag],
                                    **self.polygon_kwargs))

            if self.parameters["add_direction"]:
                direction_origin = apply_transform_matrix(position, transform) if transform is not None else position
                direction_vector = np.array([np.cos(poly_params[0]), np.sin(poly_params[0])]) * poly_params[1]

                arrow_kwargs = {"width": 0.1, "head_width": 4, "linewidth": 0.1}
                new_artists.append(self.initial_fig.gca().arrow(*direction_origin, *direction_vector,
                                                                facecolor=self.parameters["color_dict"][tag],
                                                                **arrow_kwargs))

        return new_artists


class OutlineDistanceOverlay(FrameOverlayBase):
    """
    Overlay that plots the outline of the given data and the pairwise distances between the outlines.
    The data_dict should contain a pandas DataFrame for each tag, with an x and y column, an orientation column, and two
    columns for the axes of the ellipse/box, depending on the outline_mode.
    """
    def __init__(self, initial_fig, data_dict, color_input_dict, outline_mode="ellipsis", transform=None):
        super().__init__(initial_fig, data_dict,
                         color_dict=color_input_dict, transform=transform, outline_mode=outline_mode)

    def add_data_to_frame(self, frame_index):
        trace_positions_dict = {tag: trace_df.loc[frame_index, [("center", "x"), ("center", "y")]] for tag, trace_df in
                                self.data_dict.items()}
        poly_params_dict = {tag: trace_df.loc[frame_index, ["orientation", ("axes", "0"), ("axes", "1")]] for
                            tag, trace_df in self.data_dict.items()}

        polygon_dict = {}
        scaled_dict = {}
        new_artists = []

        for tag, trace_positions in trace_positions_dict.items():
            poly_params = poly_params_dict[tag]
            if not trace_positions.empty and not any(np.isnan(trace_positions.values)):
                if self.parameters["outline_mode"] == "box":
                    poly = create_box(trace_positions.values, poly_params[0], poly_params[1], poly_params[2])
                elif self.parameters["outline_mode"] == "ellipsis":
                    poly = create_ellipsis(trace_positions.values, poly_params[0], poly_params[1], poly_params[2])
                else:
                    raise ValueError("Invalid mode")

                polygon_dict[tag] = poly
                transform = self.parameters["transform"]
                scaled_dict[tag] = affine_transform(poly, [transform[0, 0], transform[0, 1],
                                                           transform[1, 0], transform[1, 1],
                                                           transform[0, 2], transform[1, 2]]
                                                    ) if transform is not None else poly

                new_artists.extend(plot_polygons_to_ax(self.initial_fig.gca(), {tag: scaled_dict[tag]},
                                                       polygon_color=self.parameters["color_dict"][tag],
                                                       plot_labels=False))

        distance_dict = {}
        positions_dict = {}
        for i, (tag1, poly1) in enumerate(polygon_dict.items()):
            for j, (tag2, poly2) in enumerate(polygon_dict.items()):
                if i >= j:
                    continue
                distance = poly1.distance(poly2)
                if distance > 0:
                    distance_dict[((tag1, tag2), (tag2, tag1))] = distance

                    connection_points = nearest_points(scaled_dict[tag1], scaled_dict[tag2])
                    positions_dict[(tag1, tag2)] = connection_points[0].coords[0]
                    positions_dict[(tag2, tag1)] = connection_points[1].coords[0]

            new_artists.extend(
                plot_connections_from_pair_dict(distance_dict, positions=positions_dict, ax=self.initial_fig.gca(),
                                                outcome_labels=lambda x: str(round(x, 2)) + "mm",
                                                outcome_widths=0.5, outcome_colors="beige",
                                                label_font_size=10, node_labels=None,
                                                node_sizes=50)[1])
        return new_artists


class BboxOverlay(FrameOverlayBase):
    """
    Overlay that plots the bounding box of the given data, per tag. The calculation_mode can be "box" or "ellipsis", and
    is used to calculate the bbox.
    """
    def __init__(self, initial_fig, data_dict, color_input_dict, calculation_mode="box", transform=None):
        super().__init__(initial_fig, data_dict,
                         color_dict=color_input_dict, transform=transform, calculation_mode=calculation_mode)

    def add_data_to_frame(self, frame_index):
        trace_positions_dict = {tag: trace_df.loc[frame_index, [("center", "x"), ("center", "y")]] for tag, trace_df in
                                self.data_dict.items()}
        poly_params_dict = {tag: trace_df.loc[frame_index, ["orientation", "axes/0", "axes/1"]] for
                            tag, trace_df in self.data_dict.items()}

        new_artists = []
        for tag, trace_positions in trace_positions_dict.items():
            poly_params = poly_params_dict[tag]
            if not trace_positions.empty and not any(np.isnan(trace_positions.values)):
                bbox = get_bbox(trace_positions.values, poly_params[0], poly_params[1], poly_params[2],
                                calculation_mode=self.parameters["calculation_mode"])
                bbox_poly = bbox_to_polygon(bbox)
                transform = self.parameters["transform"]
                bbox_poly = affine_transform(bbox_poly, transform.T.ravel()) if transform is not None else bbox_poly

                new_artists.extend(
                    plot_polygons_to_ax(self.initial_fig.gca(), {tag: bbox_poly},
                                        polygon_color=self.parameters["color_dict"][tag],
                                        plot_labels=False))
        return new_artists


class RoiOverlay(FrameOverlayBase):
    """
    Overlay that plots the ROI of the given data, per tag, joined if necessary. The bbox_mode can be "box" or
    "ellipsis", and is used to calculate the bbox per tag. These are then joined using the roi_mode, which can be
    "points" or "polygons".
    """
    def __init__(self, initial_fig, data_dict, size_factor=2, padding=0, bbox_mode="box", roi_mode="points",
                 roi_color="white", transform=None, plot_labels=True):
        super().__init__(initial_fig, data_dict, size_factor=size_factor, padding=padding, bbox_mode=bbox_mode,
                         roi_mode=roi_mode, roi_color=roi_color, transform=transform, plot_labels=plot_labels)

    def add_data_to_frame(self, frame_index):
        trace_positions_dict = {tag: trace_df.loc[frame_index, [("center", "x"), ("center", "y")]] for tag, trace_df in
                                self.data_dict.items()}
        poly_params_dict = {tag: trace_df.loc[frame_index, ["orientation", "axes/0", "axes/1"]] for
                            tag, trace_df in self.data_dict.items()}

        size_factor = self.parameters["size_factor"]
        transform = self.parameters["transform"]
        padding = self.parameters["padding"]

        all_bboxes = []
        new_artists = []
        for tag, trace_positions in trace_positions_dict.items():
            poly_params = poly_params_dict[tag]
            if not (trace_positions.empty or any(np.isnan(trace_positions.values))):
                bbox = get_bbox(trace_positions.values, poly_params[0], poly_params[1], poly_params[2],
                                calculation_mode=self.parameters["bbox_mode"])

                all_bboxes.append(scale_and_pad_bbox(bbox,
                                                     x_fact=size_factor, y_fact=size_factor,
                                                     x_pad=padding, y_pad=padding))

        rois = {}
        for i, roi in enumerate(get_rois_from_bbox_list(all_bboxes, mode=self.parameters["roi_mode"])):
            roi_poly = bbox_to_polygon(roi)
            rois[f"ROI {i}"] = affine_transform(roi_poly, transform.T.ravel()) if transform is not None else roi_poly

        new_artists.extend(
            plot_polygons_to_ax(self.initial_fig.gca(), rois, polygon_color=self.parameters["roi_color"],
                                plot_labels=self.parameters["plot_labels"]))
        return new_artists


class SortBboxOverlay(FrameOverlayBase):
    def __init__(self, initial_fig, data_dict, matplotlib_palette="viridis", transform=None, plot_labels=True,
                 label_kwargs=None):
        super().__init__(initial_fig, data_dict, matplotlib_palette=matplotlib_palette, transform=transform,
                         plot_labels=plot_labels, label_kwargs=label_kwargs)

    def preprocessing(self):
        palette = self.parameters["matplotlib_palette"]
        color_list = get_colors_from_cmap(palette, len(self.data_dict))
        color_dict = {tag: color for tag, color in zip(self.data_dict.keys(), color_list)}

        idx_limits_df = pd.DataFrame(index=pd.Index([], name="tag"),
                                     columns=pd.Index(["min_index", "max_index"], name="limits"))
        filtered_bbox_df_dict = {}
        for tag, bbox_df in self.data_dict.items():
            filtered_bbox_df = bbox_df.dropna(how="any", subset=["x1", "y1", "x2", "y2"], axis=0)
            idx_limits_df.loc[tag] = filtered_bbox_df.index.min(), filtered_bbox_df.index.max()
            filtered_bbox_df_dict[tag] = filtered_bbox_df

        return {"colors": color_dict, "idx_limits_df": idx_limits_df, "filtered_bbox_df_dict": filtered_bbox_df_dict}

    def add_data_to_frame(self, frame_index):
        idx_limits_df = self.store["idx_limits_df"]
        relevant_tags_series = (idx_limits_df["min_index"].le(frame_index) & idx_limits_df["max_index"].ge(frame_index))
        relevant_tags = relevant_tags_series[relevant_tags_series].index

        filtered_bbox_df_dict = self.store["filtered_bbox_df_dict"]
        bbox_corners_dict = {tag: filtered_bbox_df_dict[tag].loc[frame_index, ["x1", "y1", "x2", "y2"]].values for tag
                             in relevant_tags if
                             frame_index in filtered_bbox_df_dict[tag].index}

        transform = self.parameters["transform"]

        bbox_dict = {}
        for tag, bbox_limits in bbox_corners_dict.items():
            transformed_limits = bbox_limits.copy()
            if transform is not None:
                transformed_limits = np.hstack(apply_transform_matrix(transformed_limits.reshape((2, 2)), transform))
            transformed_limits[2:4] -= transformed_limits[0:2]

            bbox_dict[tag] = bbox_to_polygon(transformed_limits)

        new_artists = plot_polygons_to_ax(self.initial_fig.gca(), bbox_dict, polygon_color=self.store["colors"],
                                          plot_labels=self.parameters["plot_labels"],
                                          label_kwargs=self.parameters["label_kwargs"])
        return new_artists


class LmtOverlay(FrameOverlayBase):
    from ..lmt_tools.lmt_mask_funcs import LmtDetection
    def __init__(self, initial_fig, data_dict, color_input_dict, transform=None, polygon_kwargs=None):
        super().__init__(initial_fig, data_dict, color_dict=color_input_dict, transform=transform)
        self.polygon_kwargs = dict({"plot_labels": False, "line_width": 1.},
                                   **(polygon_kwargs if polygon_kwargs is not None else {}))

    def add_data_to_frame(self, frame_index):
        new_artists = []
        for tag, trace_df in self.data_dict.items():

            if frame_index not in trace_df.index:
                continue

            roi_mask = LmtDetection(trace_df.loc[frame_index, "data"]).roi.mask

            transform = self.parameters["transform"]
            roi_mask = affine_transform(roi_mask, [transform[0, 0], transform[0, 1],
                                                   transform[1, 0], transform[1, 1],
                                                   transform[0, 2], transform[1, 2]]
                                        ) if transform is not None else roi_mask

            new_artists.extend(
                plot_polygons_to_ax(self.initial_fig.gca(), {tag: roi_mask},
                                    polygon_color=self.parameters["color_dict"][tag],
                                    **self.polygon_kwargs))

        return new_artists
