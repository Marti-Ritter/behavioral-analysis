import json
import os
import warnings

import cv2
import h5py
import numpy as np
import pandas as pd
from ..math_tools.array_funcs import np_update
from ..pandas_tools.funcs import iter_series_to_df, expand_index_from_limits
from tqdm.asyncio import tqdm
from ..utility.builtin_classes.dicts import set_dict_reduction_to_mds
from ..utility.files.file_tools import ensure_directory
from ..utility.files.hdf5_tools import read_data_from_h5_file, write_data_to_h5_file, extract_frame_from_video, \
    get_hdf5_df_dataset, get_hdf5_json_string_dataset


def split_analysis_dict_into_track_dicts(analysis_dict, _fix_occupancy=True):
    if _fix_occupancy:
        # fixing track_occupancy orientation, to get tracks into 0th dimension, frames in 1st dimension
        analysis_dict["track_occupancy"] = np.transpose(analysis_dict["track_occupancy"])

    individual_series_keys = ["instance_scores", "track_occupancy", "tracking_scores"]
    individual_keys = individual_series_keys + ["point_scores", "track_names", "tracks"]
    universal_keys = ["edge_inds", "edge_names", "labels_path", "node_names", "provenance", "video_ind", "video_path"]

    track_dicts = [dict(zip(individual_keys, track_data)) for track_data in
                   zip(*[analysis_dict[ind_key] for ind_key in individual_keys])]
    track_dicts = [dict(**d, **{k: analysis_dict[k] for k in universal_keys}) for d in track_dicts]
    return track_dicts


def create_sleap_track_union(union_tracks):
    """
    Creates a union track from a list of tracks. The union track is a dictionary with the same keys as the tracks, but
    with the values being the combination of the values from the tracks. The combination is either achieved by summing
    the values (see "union_sum_keys" below) or by updating the values (see "union_update_keys" below).
    The "uniform_keys" below are keys that must be the same for all tracks, otherwise an AssertionError is raised.

    :param union_tracks: A list of tracks to create a union track from
    :type union_tracks: list of dict
    :return: A union track
    :rtype: dict
    """
    union_sum_keys = ["track_occupancy"]
    union_update_keys = ["instance_scores", "tracking_scores", "point_scores", "tracks"]
    uniform_keys = ["edge_inds", "edge_names", "labels_path", "node_names", "provenance", "video_ind", "video_path",
                    "track_names"]

    union_dict = {}

    for k in union_sum_keys:
        union_dict[k] = np.sum([trk[k] for trk in union_tracks], axis=0)

    for k in union_update_keys:
        for trk in union_tracks:
            if k not in union_dict.keys():
                union_dict[k] = trk[k]
            else:
                union_dict[k] = np_update(union_dict[k], trk[k], overwrite=True)

    for k in uniform_keys:
        for trk in union_tracks:
            if k not in union_dict.keys():
                union_dict[k] = trk[k]
            else:
                assert (np.equal(union_dict[k], trk[k]).all())

    return union_dict


def create_sleap_track_unions(track_dicts):
    track_names = np.unique([trk_dict["track_names"] for trk_dict in track_dicts])

    track_unions = []
    for track_name in track_names:
        union_tracks = [trk_dict for trk_dict in track_dicts if trk_dict["track_names"] == track_name]
        track_unions.append(create_sleap_track_union(union_tracks))
    return track_unions


def combine_track_dicts_into_analysis_dict(track_dicts, _fix_occupancy=True):
    individual_series_keys = ["instance_scores", "track_occupancy", "tracking_scores"]
    individual_keys = individual_series_keys + ["point_scores", "track_names", "tracks"]
    universal_keys = ["edge_inds", "edge_names", "labels_path", "node_names", "provenance", "video_ind", "video_path"]

    analysis_dict = {}
    for k in individual_keys:
        analysis_dict[k] = np.stack([trk_dict[k] for trk_dict in track_dicts])

    for k in universal_keys:
        analysis_dict[k] = track_dicts[0][k]
        for trk_dict in track_dicts:
            assert np.equal(analysis_dict[k], trk_dict[k]).all()

    if _fix_occupancy:
        # fixing track_occupancy orientation, to get tracks into 0th dimension, frames in 1st dimension
        analysis_dict["track_occupancy"] = np.transpose(analysis_dict["track_occupancy"])

    return analysis_dict


def load_and_clean_sleap_analysis_tracks_from_h5(h5_path, expected_n_animals=4, maximum_n_tracks=30):
    h5_dict = read_data_from_h5_file(h5_path, key="/")
    track_dicts = split_analysis_dict_into_track_dicts(h5_dict)
    uniform_tracks = create_sleap_track_unions(track_dicts)

    if len(uniform_tracks) > maximum_n_tracks:
        raise ValueError("Found {} tracks, but maximum is {}.".format(len(uniform_tracks), maximum_n_tracks))

    occupancy_dict = {trk["track_names"]: pd.Series(trk["tracking_scores"]).notna().astype(int) for trk in
                      uniform_tracks}
    occupancy_df = pd.DataFrame(occupancy_dict).sort_index(axis=1)
    occupancy_df = occupancy_df.loc[:, occupancy_df.ne(0).any(axis=0)]
    occupancy_set_dict = {col: set(occupancy_df.index[occupancy_df[col].astype(bool)]) for col in occupancy_df.columns}
    tracks_to_unify = set_dict_reduction_to_mds(occupancy_set_dict, allow_alternatives=False)

    for i, trk_names in enumerate(tracks_to_unify):
        for trk in uniform_tracks:
            if trk["track_names"] in trk_names:
                trk["track_names"] = "union_{}".format(i)
    merged_tracks = create_sleap_track_unions(uniform_tracks)
    merged_tracks = [trk for trk in merged_tracks if np.nan_to_num(trk["tracking_scores"], nan=0).sum() > 0]
    if len(merged_tracks) != expected_n_animals:
        raise ValueError("Expected {} animals, but found {}.".format(expected_n_animals, len(merged_tracks)))
    return combine_track_dicts_into_analysis_dict(merged_tracks)


def load_and_clean_sleap_analysis_h5_directory(directory_path, out_directory=None, expected_n_animals=4,
                                               maximum_n_tracks=30, skip_errors=True):
    """
    Utility func that loads all sleap h5 files in a directory, cleans them, and saves them to a new directory.

    :param directory_path: Path to the directory containing the h5 files
    :type directory_path: str
    :param out_directory: Path to the directory to save the cleaned h5 files to. If None, saves to the same directory
    as the input files.
    :type out_directory: str or None
    :param expected_n_animals: The expected number of animals in the h5 files. If the number of animals is not equal to
    this number, an error is raised.
    :type expected_n_animals: int
    :param maximum_n_tracks: The maximum number of tracks allowed in the h5 files. If the number of tracks is greater
    than this number, an error is raised.
    :type maximum_n_tracks: int
    :param skip_errors: If True, skips errors during file processing. If False, raises errors.
    :type skip_errors: bool
    """
    if out_directory is None:
        out_directory = directory_path

    ensure_directory(out_directory)

    h5_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".h5")]
    for h5_path in tqdm(h5_paths):
        try:
            out_path = os.path.join(out_directory, os.path.basename(h5_path).replace(".h5", "_union.h5"))

            if os.path.isfile(out_path):
                print("Skipping file {} because it already exists.".format(out_path))
                continue

            analysis_dict = load_and_clean_sleap_analysis_tracks_from_h5(h5_path, expected_n_animals=expected_n_animals,
                                                                         maximum_n_tracks=maximum_n_tracks)
            write_data_to_h5_file(out_path, analysis_dict)
        except Exception as e:
            if skip_errors:
                print("Skipping file {} due to error: {}".format(h5_path, e))
            else:
                raise e


def get_sleap_analysis_h5_additional_track_df(h5_path):
    """
    Parses additional tracking data contained in a sleap analysis h5 file into a pandas dataframe. The dataframe columns
    will be a MultiIndex with the track name as the first level, and the track score, and track occupancy as the second
    level. The index will be the frame index.

    :param h5_path: Path to the h5 file
    :type h5_path: str
    :return: The parsed dataframe
    :rtype: pd.DataFrame
    """
    h5_dict = read_data_from_h5_file(h5_path, key="/")
    h5_dict["track_occupancy"] = np.transpose(h5_dict["track_occupancy"])

    data_iterator = zip(*[h5_dict[k] for k in
                          ["track_names", "tracking_scores", "track_occupancy"]])

    additional_df_dict = {}
    for track_name, track_score, track_occupancy in data_iterator:
        frame_index = pd.RangeIndex(start=0, stop=track_score.shape[-1], name="frame_index")
        additional_df_dict[track_name] = pd.DataFrame({"track_score": track_score, "track_occupancy": track_occupancy},
                                                      index=frame_index)
    return pd.concat(additional_df_dict, names=["track"], axis=1)


def get_sleap_analysis_h5_skeleton_df(h5_path):
    """
    Parses the skeleton data contained in a sleap analysis h5 file into a pandas dataframe. The dataframe columns will
    be a MultiIndex with the edge name as the first level, and the node names as the second level. The index will be the
    edge index. Points not connected by an edge will not be included in the dataframe.

    :param h5_path: Path to the h5 file
    :type h5_path: str
    :return: The parsed dataframe
    :rtype: pd.DataFrame
    """
    h5_dict = read_data_from_h5_file(h5_path, key="/")
    skeleton_df = pd.DataFrame(h5_dict["edge_names"], columns=["node_0", "node_1"],
                               index=pd.RangeIndex(start=0, stop=len(h5_dict["edge_names"]), name="edge_index"))
    skeleton_df = skeleton_df.rename_axis("edge_feature", axis=1)
    return skeleton_df


def get_sleap_analysis_h5_metadata(h5_path):
    """
    Parses the metadata contained in a sleap analysis h5 file into a dictionary. This dictionary will contain the
    following keys:

    - "labels_path": The path to the labels file
    - "provenance": The provenance of the data
    - "video_ind": The index of the video
    - "video_path": The path to the video file

    :param h5_path: Path to the h5 file
    :type h5_path: str
    :return: The parsed metadata
    :rtype: dict
    """
    h5_dict = read_data_from_h5_file(h5_path, key="/")
    return {k: h5_dict[k] for k in ["labels_path", "provenance", "video_ind", "video_path"]}


def parse_sleap_analysis_h5(h5_path, return_tracks=True, return_additional=True, return_skeleton=True,
                            return_metadata=True):
    """
    Parses a sleap h5 file into a dictionary. The dictionary will contain the following keys:

    - "tracks": A pandas dataframe containing the track data
    - "additional": A pandas dataframe containing the additional track data
    - "skeleton": A pandas dataframe containing the skeleton data
    - "metadata": A dictionary containing the metadata

    The keys that are not requested will not be included in the dictionary.

    :param h5_path: Path to the h5 file
    :type h5_path: str
    :param return_tracks: If True, returns the track data. See get_sleap_analysis_h5_track_df for details.
    :type return_tracks: bool
    :param return_additional: If True, returns the additional track data. See get_sleap_analysis_h5_additional_track_df
        for details.
    :type return_additional: bool
    :param return_skeleton: If True, returns the skeleton data. See get_sleap_analysis_h5_skeleton_df for details.
    :type return_skeleton: bool
    :param return_metadata: If True, returns the metadata. See get_sleap_analysis_h5_metadata for details.
    :type return_metadata: bool
    :return: The parsed h5 data
    :rtype: dict[str, pd.DataFrame or dict]
    """
    parsed_data = {}
    if return_tracks:
        parsed_data["tracks"] = get_sleap_analysis_h5_track_df(h5_path)
    if return_additional:
        parsed_data["additional"] = get_sleap_analysis_h5_additional_track_df(h5_path)
    if return_skeleton:
        parsed_data["skeleton"] = get_sleap_analysis_h5_skeleton_df(h5_path)
    if return_metadata:
        parsed_data["metadata"] = get_sleap_analysis_h5_metadata(h5_path)
    return parsed_data


def parse_sleap_h5_to_df(slp_h5_path, prediction=False):
    """
    Parses a SLEAP .slp file to a pandas DataFrame. If prediction is True, parses the predictions. If False, parses the
    ground truth.

    :param slp_h5_path: The path to the .slp file
    :type slp_h5_path: str
    :param prediction: If True, parses the predictions. If False, parses the ground truth.
    :type prediction: bool
    :return: A pandas DataFrame of the predictions or ground truth.
    :rtype: pd.DataFrame
    :raises ValueError: If the .slp file is not a valid SLEAP file or if the .slp file contains an unexpected format for
    the video information.
    """

    metadata = json.loads(read_data_from_h5_file(slp_h5_path, key="/metadata", attribute="json"))

    node_index = pd.Series([x["id"] for x in metadata["skeletons"][0]["nodes"]], name="id").sort_values().index
    node_df = pd.DataFrame(metadata["nodes"], index=node_index).rename({"name": "bodyparts"}, axis=1).sort_index()

    videos_json_series = pd.Series(read_data_from_h5_file(slp_h5_path, key="/videos_json")).apply(
        lambda x: json.loads(x)["backend"])
    video_df = iter_series_to_df(videos_json_series)

    if video_df["filename"].eq(".").all() and not video_df["dataset"].eq("").all():
        video_df["video"] = video_df["dataset"].apply(lambda x: x.split("/")[0])
        video_df["video_index"] = video_df["video"].str.replace(r"\D", "", regex=True).astype(int)

        video_df = video_df.drop("filename", axis=1)
        video_path_dict = {video_name: json.loads(
            read_data_from_h5_file(slp_h5_path, key=f"/{video_name}/source_video", attribute="json")) for video_name in
            video_df["video"]}
        video_path_series = pd.Series(video_path_dict).apply(lambda x: x["backend"])
        video_path_df = iter_series_to_df(video_path_series)
        video_df = video_df.join(video_path_df[["filename", "grayscale", "bgr"]], on="video")

    elif "dataset" not in video_df or video_df["dataset"].eq("").all():
        video_df["video_index"] = range(len(video_df))

    else:
        raise ValueError("Could not parse video information from .slp file. Something unexpected happened.")

    points_df = read_data_from_h5_file(slp_h5_path, key="/points")
    pred_points_df = read_data_from_h5_file(slp_h5_path, key="/pred_points")
    frames_df = read_data_from_h5_file(slp_h5_path, key="/frames")

    instances_df = read_data_from_h5_file(slp_h5_path, key="/instances")
    training_instances_df = instances_df[instances_df["instance_type"].eq(0)]
    predicted_instances_df = instances_df[instances_df["instance_type"].eq(1)]

    expanded_instance_df = expand_index_from_limits(predicted_instances_df if prediction else training_instances_df,
                                                    start_col="point_id_start", end_col="point_id_end",
                                                    index_name="point_index")

    if not len(expanded_instance_df["instance_id"].unique()) == 1:
        expanded_instance_df["point_enum"] = expanded_instance_df.groupby(["frame_id", "instance_id"]).apply(
            lambda x: pd.Series(range(len(x)), index=x.index)).droplevel([0, 1])
    else:
        expanded_instance_df["point_enum"] = range(len(expanded_instance_df))

    expanded_instance_df = expanded_instance_df.join(node_df["bodyparts"], on="point_enum")

    expanded_frame_df = expand_index_from_limits(frames_df, start_col="instance_id_start", end_col="instance_id_end",
                                                 index_name="instance_index")

    # to actually ensure that we have the expected number of tracked instances, as every frame can have two full sets
    # of instances (one for training, one from prediction)
    expanded_frame_df = expanded_frame_df[expanded_frame_df.index.isin(expanded_instance_df["instance_id"])]

    if not len(expanded_frame_df["frame_id"].unique()) == 1:
        # if we have multiple frames, we need to ensure that the instance enumeration is starting from 0 in each frame
        expanded_frame_df["instance_enum"] = expanded_frame_df.groupby("frame_id").apply(
            lambda x: pd.Series(range(len(x)), index=x.index).astype(str)).droplevel(0)
    else:
        expanded_frame_df["instance_enum"] = range(len(expanded_frame_df))

    join_df = expanded_instance_df.join(pred_points_df if prediction else points_df, rsuffix="_points", how="inner")
    join_df = expanded_frame_df.join(join_df.set_index("instance_id"), rsuffix="_instance", how="inner")
    join_df = join_df.join(video_df.set_index("video_index")["filename"], on="video")

    return join_df.reset_index(drop=True)


def extract_frames_and_metadata_from_sleap():
    sleap_file = r"I:\20230412_tracking\sleap\labels_clean_videos.v006.pkg.slp"
    sleap = h5py.File(sleap_file, "r")
    metadata = json.loads(sleap["metadata"].attrs.get("json").decode())
    with open(r'I:\20230412_tracking\sleap\labels_clean_videos.v006.pkg.meta.json', 'w') as f:
        json.dump(metadata, f)

    node_index = pd.Series([x["id"] for x in metadata["skeletons"][0]["nodes"]], name="id").sort_values().index
    node_df = pd.DataFrame(metadata["nodes"], index=node_index).rename({"name": "bodyparts"}, axis=1)

    source_videos = get_sleap_source_videos(sleap)

    experimenter = "MR"
    output_root = r"I:\20230412_tracking\dlc_poc\labeled-data\test_data_slp"
    output_dir = "labeled-data"

    sleap_points = get_hdf5_df_dataset(sleap, "points")
    sleap_instances = get_hdf5_df_dataset(sleap, "instances")
    frame_df = get_hdf5_df_dataset(sleap, "frames")

    experimenter_instances = sleap_instances[sleap_instances["instance_type"].eq(0)]
    full_instance_df = expand_index_from_limits(experimenter_instances, start_col="point_id_start",
                                                end_col="point_id_end", index_name="instance_index")
    full_instance_df["point_num"] = full_instance_df.groupby(["frame_id", "instance_id"]).apply(
        lambda x: pd.Series(range(len(x)), index=x.index)).droplevel([0, 1])
    full_instance_df = full_instance_df.join(node_df["bodyparts"], on="point_num")

    full_frame_df = expand_index_from_limits(frame_df, start_col="instance_id_start", end_col="instance_id_end",
                                             index_name="instance_index")
    full_frame_df["individuals"] = full_frame_df.groupby("frame_id").apply(
        lambda x: "individual" + pd.Series(range(1, len(x) + 1), index=x.index).astype(str)).droplevel(0)

    full_df = full_instance_df.join(sleap_points)
    full_df = full_frame_df.join(full_df.set_index("instance_id"), rsuffix="_instance", how="inner")
    full_df = full_df.join(pd.Series(source_videos, name="video_path"), on="video")

    full_df["video_name"] = full_df["video_path"].apply(lambda x: os.path.basename(os.path.splitext(x)[0]))
    full_df["image_file"] = full_df.apply(lambda x: f"img{str(x['frame_idx']).zfill(6)}.png", axis=1)

    full_df["scorer"] = experimenter
    full_df["output_dir"] = output_dir

    full_df["output_path"] = full_df.apply(lambda x: os.path.join(x["output_dir"], x["video_name"], x["image_file"]),
                                           axis=1)

    full_df.loc[~full_df["visible"], ["x", "y"]] = np.nan

    ensure_directory(os.path.join(output_root, "labeled-data"))

    for video, video_df in tqdm(full_df.groupby("video")):
        dlc_video_df = sleap_df_to_dlc_format(video_df)
        dlc_video_df = dlc_video_df.reindex(
            dlc_video_df.columns.sortlevel(level=[0, 1, 2, 3], ascending=True, sort_remaining=True)[0], axis=1)
        video_path = video_df["video_path"].unique()[0]

        video_dir = os.path.join(output_root, output_dir, os.path.basename(os.path.splitext(video_path)[0]))
        ensure_directory(video_dir)

        for frame_idx, frame_df in video_df.groupby("frame_index"):
            output_path = frame_df["output_path"].unique()[0]
            frame_output_path = os.path.join(output_root, output_path.replace("/", "\\"))

            if os.path.isfile(frame_output_path):
                continue

            frame = extract_frame_from_video(video_path, frame_idx)
            cv2.imwrite(frame_output_path, frame)

        dlc_video_df.to_pickle(os.path.join(video_dir, f"CollectedData_{experimenter}.pkl"))


def sleap_predictions_to_dlc_format(sleap_file, output_dir, experimenter="SLEAP"):
    # sleap_file = r"I:\20230412_tracking\max_looming\CSIVideo_2023_4_17_17_8_24\loom1.predictions.slp"

    sleap = h5py.File(sleap_file, "r")
    metadata = json.loads(sleap["metadata"].attrs.get("json").decode())

    node_index = pd.Series([x["id"] for x in metadata["skeletons"][0]["nodes"]], name="id").sort_values().index
    node_df = pd.DataFrame(metadata["nodes"], index=node_index).rename({"name": "bodyparts"}, axis=1)

    source_videos = get_sleap_source_videos(sleap)

    sleap_points = get_hdf5_df_dataset(sleap, "pred_points")
    sleap_instances = get_hdf5_df_dataset(sleap, "instances")
    frame_df = get_hdf5_df_dataset(sleap, "frames")

    predicted_instances = sleap_instances[sleap_instances["instance_type"].eq(1)]
    full_instance_df = expand_index_from_limits(predicted_instances, start_col="point_id_start",
                                                end_col="point_id_end", index_name="instance_index")
    full_instance_df["point_num"] = full_instance_df.groupby(["frame_id", "instance_id"]).apply(
        lambda x: pd.Series(range(len(x)), index=x.index)).droplevel([0, 1])
    full_instance_df = full_instance_df.join(node_df["bodyparts"], on="point_num")

    full_frame_df = expand_index_from_limits(frame_df, start_col="instance_id_start", end_col="instance_id_end",
                                             index_name="instance_index")
    full_frame_df["individuals"] = full_frame_df.groupby("frame_id").apply(
        lambda x: "individual" + pd.Series(range(1, len(x) + 1), index=x.index).astype(str)).droplevel(0)

    full_df = full_instance_df.join(sleap_points, rsuffix="_point")
    full_df = full_frame_df.join(full_df.set_index("instance_id"), rsuffix="_instance", how="inner")
    full_df = full_df.join(pd.Series(source_videos, name="video_path"), on="video")

    full_df["video_name"] = full_df["video_path"].apply(lambda x: os.path.basename(os.path.splitext(x)[0]))
    full_df["image_file"] = full_df.apply(lambda x: f"img{str(x['frame_idx']).zfill(6)}.png", axis=1)

    full_df["scorer"] = experimenter

    output_dir_name = os.path.basename(output_dir)
    full_df["output_dir"] = output_dir_name

    full_df["output_path"] = full_df.apply(lambda x: os.path.join(x["output_dir"], x["video_name"], x["image_file"]),
                                           axis=1)

    full_df.loc[~full_df["visible"], ["x", "y"]] = np.nan

    ensure_directory(output_dir)

    for video, video_df in tqdm(full_df.groupby("video")):
        dlc_video_df = sleap_df_to_dlc_format(video_df, prediction=True)
        dlc_video_df = dlc_video_df.reindex(
            dlc_video_df.columns.sortlevel(level=[0, 1, 2, 3], ascending=True, sort_remaining=True)[0], axis=1)
        video_path = video_df["video_path"].unique()[0]

        video_dir = os.path.join(output_dir, os.path.basename(os.path.splitext(video_path)[0]))
        ensure_directory(video_dir)

        dlc_video_df.to_pickle(os.path.join(video_dir, f"CollectedData_{experimenter}.pkl"))


def sleap_df_to_dlc_format(sleap_df, prediction=False):
    if not prediction:
        dlc_df = sleap_df.pivot(columns=["bodyparts", "individuals", "scorer"],
                                index=["output_dir", "video_name", "image_file"], values=["x", "y"])
    else:
        dlc_df = sleap_df.pivot(columns=["bodyparts", "individuals", "scorer"],
                                index=["output_dir", "video_name", "image_file"], values=["x", "y", 'score_points'])
    dlc_df.columns = dlc_df.columns.rename("coords", level=0).reorder_levels([3, 2, 1, 0])
    dlc_df = dlc_df.reindex(dlc_df.columns.sortlevel(level=[0, 1, 2, 3], ascending=True, sort_remaining=True)[0], axis=1)
    dlc_df.index.names = [None, None, None]
    return dlc_df


def sleap_labels_to_dlc_formats(sleap_df, scorer="MR", output_dir="labeled-data"):
    sleap_df["scorer"] = scorer
    sleap_df["output_dir"] = output_dir

    sleap_df["video_name"] = sleap_df["filename"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    sleap_df["image_file"] = sleap_df["frame_index"].apply(lambda x: "img" + str(x).zfill(6) + ".png")
    sleap_df["individuals"] = sleap_df["instance_enum"].apply(lambda x: "individual" + str(int(x) + 1))

    sleap_df.loc[~sleap_df["visible"].astype(bool), ["x", "y"]] = np.nan

    dlc_df = sleap_df_to_dlc_format(sleap_df, prediction=False)

    return dlc_df


def get_sleap_video_path(slp_hdf5_object, video_index):
    return json.loads(slp_hdf5_object[f"video{video_index}"]["source_video"].attrs.get("json"))["backend"]["filename"]


def get_sleap_video_df(slp_hdf5_object):
    return pd.DataFrame([x["backend"] for x in get_hdf5_json_string_dataset(slp_hdf5_object, "videos_json")])


def get_sleap_source_videos(slp_hdf5_object):
    video_df = get_sleap_video_df(slp_hdf5_object)
    available_videos = video_df["dataset"].apply(lambda x: x.strip("video/"))
    if available_videos.eq("").all():
        return video_df["filename"].to_dict()
    return {int(video_index): get_sleap_video_path(slp_hdf5_object, video_index) for video_index in available_videos}


def pil_image_from_sleap_labeled_frame(sleap_labeled_frame):
    from PIL import Image

    frame_array = sleap_labeled_frame.image
    is_grayscale = frame_array.ndim == 2 or frame_array.shape[2] == 1

    if is_grayscale:
        squeezed_image = sleap_labeled_frame.image.squeeze()
        return Image.fromarray(squeezed_image, mode="L")
    else:
        has_bgr = hasattr(sleap_labeled_frame.video, "bgr")
        if has_bgr and sleap_labeled_frame.video.bgr:
            return Image.fromarray(sleap_labeled_frame.image[:, :, ::-1], mode="RGB")
        else:
            return Image.fromarray(sleap_labeled_frame.image, mode="RGB")


def sleap_file_to_extracted_labeled_frames(sleap_file_path, frame_out_dir):
    import sleap
    from sleap.io.video import HDF5Video

    sleap_source_dataset = sleap.load_file(str(sleap_file_path))

    for video in tqdm(sleap_source_dataset.videos):
        if isinstance(video.backend, HDF5Video):
            import warnings
            warnings.warn(f"Video {video.backend} is an HDF5 video. Frames will be extracted directly "
                          "from the SLEAP file and further cleanup of the directories may be needed.")

        video_dataset = sleap_source_dataset.find(video=video)
        video_path = video.filename
        video_name = os.path.basename(os.path.splitext(video_path)[0])
        if video.dataset:
            import re
            video_dataset_name = re.sub(r"\W+", "", video.dataset)  # sanitized dataset name
            video_name = f"{video_name}_{video_dataset_name}"
        video_out_dir = os.path.join(frame_out_dir, video_name)

        os.makedirs(video_out_dir, exist_ok=True)

        for frame in video_dataset:
            if not frame.has_user_instances:
                continue

            frame_name = f"img{str(frame.frame_idx).zfill(5)}.png"

            frame_output_path = os.path.join(video_out_dir, frame_name)
            if not os.path.isfile(frame_output_path):
                pil_image = pil_image_from_sleap_labeled_frame(frame)
                pil_image.save(frame_output_path)


def sleap_file_to_tracking_df(sleap_file_path, load_user_labels=True, show_progress=True):
    import sleap

    sleap_source_dataset = sleap.load_file(str(sleap_file_path))

    video_iterator = sleap_source_dataset.videos
    if show_progress:
        video_iterator = tqdm(video_iterator)

    video_df = None

    video_dict = {}
    for video in video_iterator:
        video_dataset = sleap_source_dataset.find(video=video)
        video_path = video.filename
        video_name = os.path.basename(os.path.splitext(video_path)[0])
        if video.dataset:
            import re
            video_dataset_name = re.sub(r"\W+", "", video.dataset)  # sanitized dataset name
            video_name = f"{video_name}_{video_dataset_name}"

        video_frame_dict = {}
        for frame in video_dataset:
            if not frame.has_user_instances and load_user_labels:
                continue

            anonymous_track_counter = 1

            frame_instance_dict = {}

            instances_to_load = frame.user_instances if load_user_labels else frame.instances

            for instance in instances_to_load:
                array_to_load, columns = (instance.points_array, ["x", "y"]) if load_user_labels else (
                    instance.points_and_scores_array, ["x", "y", "score"])
                points_df = pd.DataFrame(array_to_load, index=instance.skeleton.node_names,
                                         columns=columns)
                instance_name = instance.track.name if instance.track else f"individual{anonymous_track_counter}"
                instance_df = points_df.stack()

                if not load_user_labels:
                    instance_df.loc[("instance", "score")] = instance.score

                frame_instance_dict[instance_name] = instance_df
                anonymous_track_counter += 1

            frame_series = pd.concat(frame_instance_dict, axis=0,
                                     names=["individuals", "keypoint_name", "keypoint_feature"]).rename()
            video_frame_dict[frame.frame_idx] = frame_series

        if not video_frame_dict:
            continue
        video_df = pd.concat(video_frame_dict, axis=0, names=["frame_index"] + frame_series.index.names).unstack(
            level=[1, 0]).T
        video_dict[video_name] = video_df

    if video_df is None:
        raise ValueError("No videos could be processed in the provided dataset. "
                         "Did you provide the right load_user_labels?")

    tracking_df = pd.concat(video_dict, axis=0, names=["video_name"] + video_df.index.names, sort=False).sort_index()

    if len(sleap_source_dataset.skeletons) == 1:
        node_name_order = sleap_source_dataset.skeletons[0].node_names
        if load_user_labels:
            tracking_df = tracking_df.reindex(node_name_order, level=0, axis=1)
        else:
            tracking_df = tracking_df.reindex(["instance"] + node_name_order, level=0, axis=1)

    return tracking_df


def sleap_file_to_dlc_labels(sleap_file_path, dlc_out_dir, scorer="SLEAP", write_csv=True):
    sleap_file_to_extracted_labeled_frames(sleap_file_path, dlc_out_dir)
    sleap_tracking_df = sleap_file_to_tracking_df(sleap_file_path)

    _index_df = sleap_tracking_df.index.to_frame()
    _index_df["frame_index"] = _index_df["frame_index"].apply(lambda x: f"img{str(x).zfill(5)}.png")
    sleap_tracking_df.index = pd.MultiIndex.from_frame(_index_df)
    original_keypoint_order = sleap_tracking_df.columns.get_level_values("keypoint_name").unique()
    sleap_tracking_df = sleap_tracking_df.unstack("individuals").reorder_levels([2, 0, 1], axis=1).sort_index(
        level=[0, 2], axis=1).reindex(original_keypoint_order, level=1, axis=1)
    sleap_tracking_df = pd.concat({scorer: sleap_tracking_df},
                                  names=["scorer", "individuals", "bodyparts", "coords"], axis=1)
    sleap_tracking_df = pd.concat({"labeled-data": sleap_tracking_df}, names=[None, None, None], axis=0)

    for video_name, video_df in sleap_tracking_df.groupby(level=1, axis=0):
        video_out_dir = os.path.join(dlc_out_dir, video_name)

        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir, exist_ok=True)

        h5_output_path = os.path.join(video_out_dir, f"CollectedData_{scorer}.h5")
        if not os.path.isfile(h5_output_path):
            video_df.to_hdf(h5_output_path, key="keypoints")

            if write_csv:
                video_df.to_csv(h5_output_path.replace(".h5", ".csv"))


def copy_sleap_file_videos(sleap_file_path, video_out_dir):
    import sleap
    from sleap.io.video import MediaVideo
    import shutil
    sleap_source_dataset = sleap.load_file(str(sleap_file_path))

    os.makedirs(video_out_dir, exist_ok=True)

    for video in tqdm(sleap_source_dataset.videos):
        correct_backend = isinstance(video.backend, MediaVideo)
        if correct_backend:
            if not video.is_missing:
                video_path = video.filename
                out_path = os.path.join(video_out_dir, os.path.basename(video_path))
                if not os.path.isfile(out_path):
                    shutil.copy2(video_path, out_path)
            else:
                print(f"Video {video.filename} is missing.")
        else:
            print(f"Video {video.filename} has incorrect backend {video.backend} for copying.")


def parse_sleap_analysis_csv(csv_path):
    """
    Parses a csv file containing sleap analysis data into a pandas dataframe. The csv data will have an arbitrary number
    of keypoints columns (each containing an x, y, and score), and multiple index columns (usually track and frame_idx).
    The keypoint columns are a joined MultiIndex with the keypoint name and the x, y, and score as subcolumns.
    These subcolumns are indicated by a point separation like such: keypoint.x, keypoint.y, keypoint.score..

    :param csv_path: Path to the csv file
    :type csv_path: str
    :return: The parsed dataframe
    :rtype: pd.DataFrame
    """
    csv_df = pd.read_csv(csv_path)
    keypoint_cols = [col for col in csv_df.columns if "." in col]
    index_cols = [col for col in csv_df.columns if col not in keypoint_cols]

    keypoint_cols = [col.split(".") for col in keypoint_cols]
    keypoint_cols = pd.MultiIndex.from_tuples(keypoint_cols, names=["keypoint_name", "keypoint_feature"])

    csv_df = csv_df.set_index(index_cols)
    csv_df.columns = keypoint_cols

    # rename index "frame_idx" to "frame_index" for consistency
    if "frame_idx" in csv_df.index.names:
        csv_df.index.names = ["frame_index" if name == "frame_idx" else name for name in csv_df.index.names]

    return csv_df


def get_sleap_analysis_h5_track_df(h5_path):
    """
    Parses the tracks contained in a sleap analysis h5 file into a pandas dataframe. The dataframe columns will be a
    MultiIndex with the track name as the first level, keypoint name as the second level, and the x, y (possibly z)
    coordinates and score as the third level. The index will be the frame index.

    :param h5_path: Path to the h5 file
    :type h5_path: str
    :return: The parsed dataframe
    :rtype: pd.DataFrame
    """

    h5_dict = read_data_from_h5_file(h5_path, key="/")
    dimension_names = "xyz"
    node_names = h5_dict["node_names"]

    data_iterator = zip(*[h5_dict[k] for k in ["track_names", "instance_scores", "tracks", "point_scores"]])

    track_df_dict = {}
    for track_name, instance_score, track, point_scores in data_iterator:
        dim_dfs = []
        frame_index = pd.RangeIndex(start=0, stop=track.shape[-1], name="frame_index")
        for dim_name, track_dim in zip(dimension_names, track):
            dim_dfs.append(pd.DataFrame(track_dim.T,
                                        columns=pd.MultiIndex.from_product([node_names, [dim_name]],
                                                                           names=["keypoint_name", "keypoint_feature"]),
                                        index=frame_index))
        point_df = pd.DataFrame(point_scores.T,
                                columns=pd.MultiIndex.from_product([node_names, ["score"]],
                                                                   names=["keypoint_name", "keypoint_feature"]),
                                index=frame_index)
        track_df = pd.concat(dim_dfs + [point_df], axis=1).sort_index(axis=1)
        instance_score_df = pd.DataFrame(instance_score.T,
                                         columns=pd.MultiIndex.from_product([["instance"], ["score"]],
                                                                            names=["keypoint_name", "keypoint_feature"]),
                                         index=frame_index)
        track_df_dict[track_name] = pd.concat([instance_score_df, track_df], axis=1)

    return pd.concat(track_df_dict, names=["track"], axis=0)


def get_single_sleap_tracking_df(h5_or_csv_path, ignore_invalid_extension=False):
    """
    A wrapper func for get_sleap_analysis_h5_track_df and parse_sleap_analysis_csv. The function will load the tracking
    data from either a h5 file or a csv file generated by the sleap tracking pipeline. If the file extension is invalid,
    the function will raise a ValueError unless ignore_invalid_extension is set to True.

    :param h5_or_csv_path: The path to the h5 file or csv file.
    :type h5_or_csv_path: str
    :param ignore_invalid_extension: If True, the function will not raise an error if the file extension is invalid
    :type ignore_invalid_extension: bool
    :return: A dataframe containing the tracking data or None if the file extension is invalid and
    ignore_invalid_extension is True.
    :rtype: pd.DataFrame or None
    """
    if h5_or_csv_path.endswith(".csv"):
        track_df = parse_sleap_analysis_csv(h5_or_csv_path)
    elif h5_or_csv_path.endswith(".h5"):
        track_df = get_sleap_analysis_h5_track_df(h5_or_csv_path)
    elif ignore_invalid_extension:
        track_df = None
    else:
        raise ValueError("Invalid file extension, must be either .csv or .h5.")
    return track_df


def get_sleap_tracking_dict(h5_or_csv_dir):
    """
    Load a dictionary of tracking data from a directory containing csv files or h5 files generated by the sleap
    tracking pipeline. The returned dictionary of dataframes should have the general structure
    {recording_name: dataframe}. Multi-track data will be split into separate keys in the dictionary. If there are mixed
    h5 and csv files in the directory, the function will iterate over all files and load them accordingly, but throw a
    warning if the file types are mixed.

    :param h5_or_csv_dir: The path to the directory containing the csv files or h5 files.
    :type h5_or_csv_dir: str
    :return: A dictionary of dataframes containing the tracking data.
    :rtype: dict[str, pd.DataFrame]
    """

    def _file_name_to_key(file_name):
        return os.path.basename(file_name).rsplit(".", 1)[0]

    file_types = [f.rsplit(".", 1)[1] for f in os.listdir(h5_or_csv_dir)]
    if len(set(file_types)) > 1:
        warnings.warn(f"Directory contains mixed file types: {set(file_types)}")

    output_dict = {}
    for f in os.listdir(h5_or_csv_dir):
        track_df = get_single_sleap_tracking_df(os.path.join(h5_or_csv_dir, f), ignore_invalid_extension=True)
        if track_df is None:
            continue
        tracks = track_df.index.get_level_values("track").unique()
        for track in tracks:
            track_suffix = f"_{track.replace('_', '')}" if len(tracks) > 1 else ""
            output_dict[_file_name_to_key(f) + track_suffix] = track_df.loc[track, :]
    return output_dict


def get_sleap_tracking_df(h5_or_csv_dir):
    """
    A convenience function to use get_sleap_tracking_dict and merge the resulting dictionary of dataframes into a single
    dataframe.

    :param h5_or_csv_dir: The path to the directory containing the csv files or h5 files.
    :type h5_or_csv_dir: str
    :return: A single dataframe containing the tracking data.
    :rtype: pd.DataFrame
    """

    return pd.concat(get_sleap_tracking_dict(h5_or_csv_dir), names=["track", "frame_index"], axis=0)
