import json
import os

import cv2
import h5py
import numpy as np
import pandas as pd


def get_nodes_and_attributes_from_h5_file(tables_object_or_path, return_attributes=False):
    """
    Returns a dictionary of all nodes in a PyTables file. If return_attributes is True, also returns a dictionary of
    the nodes' attributes.

    :param tables_object_or_path: Either a path to a PyTables file or a PyTables file object (e.g. tables.File)
    :type tables_object_or_path: str or tables.File
    :param return_attributes: If True, also returns a dictionary of the nodes' attributes.
    :type return_attributes: bool
    :return: A dictionary of the nodes in the file, and optionally a dictionary of the nodes' attributes.
    :rtype: dict
    """
    if isinstance(tables_object_or_path, str):
        file = h5py.File(tables_object_or_path, 'r')
    else:
        file = tables_object_or_path

    nodes = {name: node for name, node in file.items()}
    if return_attributes:
        attributes = {name: dict(node.attrs) for name, node in file.items()}
        return nodes, attributes

    return nodes


def read_data_from_h5_file(h5py_object_or_path, key="/", attribute=None, read_group_contents=True,
                           unknown_node_type_error=False):
    """
    Reads data from a h5py file. If the key is a group, it returns a dictionary of all nodes in the group.
    If the key is a node, it returns the node's data. If the key is a node and an attribute is specified, it returns
    the node's attribute. If the key is a group and read_group_contents is True, it returns a dictionary of all nodes in
    the group.

    :param h5py_object_or_path: Either a path to a h5py file or a h5py file object (e.g. tables.File)
    :type h5py_object_or_path: str or tables.File
    :param key: The key of the node or group to read, defaults to "/", which is the root group.
    :type key: str
    :param attribute: The attribute to read. If None, reads the node's data. If not None, reads the node's attribute.
    :type attribute: str or None
    :param read_group_contents: If True, reads the contents of the group. If False, returns a dictionary of the group's
    nodes.
    :type read_group_contents: bool
    :param unknown_node_type_error: If True, raises an error if the node type is unknown. If False, returns the node
    :type unknown_node_type_error: bool
    :return: The node's data or attribute, or a dictionary of the group's nodes' data or the nodes themselves (depending
    on read_group_contents).
    :rtype: np.ndarray or pd.DataFrame or dict or tables.Node or bytes
    :raises ValueError: If the node type is unknown and unknown_node_type_error is True.
    """
    if isinstance(h5py_object_or_path, str):
        file = h5py.File(h5py_object_or_path, 'r')
    else:
        file = h5py_object_or_path

    try:
        if key in file:
            node = file[key]
            if attribute is not None:
                return node.attrs[attribute]

            if isinstance(node, h5py.Dataset):
                # check if node is a string. If it is, convert it to a string before returning it (later)
                is_string = h5py.check_string_dtype(node.dtype) is not None
                # check if node has a shape (i.e. is not a scalar)
                has_shape = node.shape
                # check if node is a compound dataset (i.e. table)
                is_compound = (node.dtype.names is not None) if has_shape else False

                if has_shape and is_compound:
                    return pd.DataFrame(np.array(node)).squeeze()  # convert data to DataFrame
                else:
                    return (node.asstr() if is_string else node)[()]  # convert to string if necessary, return data

            elif isinstance(node, h5py.Group):
                if read_group_contents:
                    return {name: read_data_from_h5_file(file, f"{key}/{name}") for name in node}
                else:
                    return {name: file[f"{key}/{name}"] for name in node}
            else:
                if unknown_node_type_error:
                    raise ValueError(f"Unknown node type: {type(node)}")
                else:
                    return node
        else:
            raise KeyError(f"No such key: {key}")

    finally:
        if isinstance(h5py_object_or_path, str):
            file.close()


def write_data_to_h5_file(h5py_object_or_path, data_dict, key="/", overwrite=False):
    """
    Inverse of read_data_from_h5_file. Writes data to a h5py file.
    This does not yet implement the writing of attributes.

    :param h5py_object_or_path: Either a path to a h5py file or a h5py file object (e.g. h5py.File)
    :type h5py_object_or_path: str or h5py.File
    :param data_dict: A dictionary of data to write to the file
    :type data_dict: dict
    :param key: The key of the node or group to write to, defaults to "/", which is the root group.
    :type key: str
    :param overwrite: If True, overwrites the data if it already exists. If False, raises an error if the data already
    exists.
    :type overwrite: bool
    :return: None
    :rtype: None
    """
    if isinstance(h5py_object_or_path, str):
        file = h5py.File(h5py_object_or_path, 'w')
    else:
        file = h5py_object_or_path

    try:
        target_node = file[key] if key in file else file.create_group(key)
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray) and v.dtype.type is np.str_:
                v = v.astype("S")
            if isinstance(v, pd.DataFrame):
                v = v.to_records(index=False)
            if k in target_node and overwrite:
                data = target_node[k]
                data[...] = v
            elif k in target_node and not overwrite:
                raise ValueError(f"Data already exists at key {key}/{k}")
            else:
                target_node[k] = v
    finally:
        file.close()


def translate_all_pkl_to_hdf(root_directory):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                translate_pkl_to_hdf(file_path, file_path.replace(".pkl", ".h5"))


def translate_pkl_to_hdf(pkl_path, hdf_path):
    pkl_df = pd.read_pickle(pkl_path)
    pkl_df = pkl_df.reindex(pkl_df.columns.sortlevel(level=[0, 1, 2, 3], ascending=True, sort_remaining=True)[0],
                            axis=1)
    pkl_df.to_hdf(hdf_path, key="df_with_missing", mode="w")


def extract_frame_from_video(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame


def get_hdf5_string_dataset(hdf5_object, dataset_name):
    return hdf5_object[dataset_name][()].astype(str)


def get_hdf5_df_dataset(hdf5_object, dataset_name):
    return pd.DataFrame(hdf5_object[dataset_name][()])


def get_hdf5_json_string_dataset(hdf5_object, dataset_name):
    string_dataset = get_hdf5_string_dataset(hdf5_object, dataset_name)
    return [json.loads(string) for string in string_dataset]


def series_to_dataframe(series, start, end):
    # Create a DataFrame with the Series values broadcasted to fill all rows and columns
    df = pd.DataFrame(series.values.reshape(1, -1).repeat(end - start, axis=0))
    # Set the index of the DataFrame to a RangeIndex with start and end as parameters
    df.index = pd.RangeIndex(start, end)
    # Set the columns of the DataFrame to the index of the Series and the matching dtypes
    df.columns = series.index
    # Return the resulting DataFrame
    return df
