import pandas as pd
from ..utility.builtin_classes.dicts import set_dict_value_by_graph, get_dict_value_by_graph, nested_dict_leaf_iterator

from .funcs import pd_prepend_column_levels_to_df, pd_sort_columns


class SubclassedDataFrame(pd.DataFrame):
    _metadata = ["extension_funcs"]
    default_funcs = {"b": (lambda x: x["a"] + 1)}

    def __init__(self, *args, **kwargs):  # cannot change this signature, otherwise dataframe breaks
        super(SubclassedDataFrame, self).__init__(*args, **kwargs)
        self.extension_funcs = self.default_funcs
        # new kwargs/args have to be checked like this, to avoid changing signature and thus breaking the frame
        if "auto_assignment_funcs" in kwargs:
            self.extension_funcs = {**self.extension_funcs, **kwargs["auto_assignment_funcs"]}

    def __getitem__(self, item):
        if item not in self and item in self.extension_funcs:
            self[item] = self.apply(lambda x: self.extension_funcs[item](x), axis=1)
        return super(SubclassedDataFrame, self).__getitem__(item)

    @property
    def _constructor(self):
        return SubclassedDataFrame


class PandasObjectDictMixin:
    """
    A mixin class for all dicts or MutableMappings that include pandas objects, such as pd.Series or pd.DataFrame.
    """
    keys = None
    __getitem__ = None
    clear = None

    def to_df(self, included_keys=None, excluded_keys=None, sort_columns=True, sort_index=True):
        """
        Converts all pandas objects in the dict to a single dataframe, with the keys as a single column level. If the
        dict contains non-pandas objects, they are converted to pd.Series with the key as the name.

        :param included_keys: Keys to include in the output dataframe. If None, all keys are included.
        :type included_keys: list or tuple
        :param excluded_keys: Keys to exclude from the output dataframe. If None, no keys are excluded. Exclude takes
        precedence over include.
        :type excluded_keys: list or tuple
        :param sort_columns: Whether to sort the columns according to the implementation in pd_sort_columns.
        :type sort_columns: bool
        :param sort_index: Whether to sort the index before concatenating any pandas objects. The sort is done with the
        default parameters in pd.Index.sort_values.
        :type sort_index: bool
        :return: A single dataframe containing all pandas objects in the dict.
        :rtype: pd.DataFrame
        """
        included_keys = self.keys() if included_keys is None else included_keys
        excluded_keys = tuple() if excluded_keys is None else excluded_keys
        contained_pd_objects = []
        non_pd_objects = {}
        for key in included_keys:
            if key in excluded_keys:
                continue
            elif isinstance(self[key], pd.DataFrame):
                contained_pd_objects.append(pd_prepend_column_levels_to_df(self[key], {"key": key}))
            elif isinstance(self[key], pd.Series):
                contained_pd_objects.append(self[key].rename(key))
            else:
                # print(f"Non-Pandas value \"{self[key]}\" found, translating to pd.Series")
                non_pd_objects[key] = self[key]
        if sort_index:
            contained_pd_objects = [pd_obj.sort_index() for pd_obj in contained_pd_objects]
        output_df = pd.concat(contained_pd_objects, axis=1)
        for key, value in non_pd_objects.items():
            output_df[key] = pd.Series(data=value, index=output_df.index, name=key, dtype=object)
        return pd_sort_columns(output_df) if sort_columns else output_df

    def from_df(self, input_df):
        """
        Overwrites this nested dict with data from a DataFrame, with the column levels as keys. If the dataframe
        contains non-pandas objects, they are converted to pd.Series with the column level as the name.

        :param input_df: The dataframe to convert to a dict of pandas objects.
        :type input_df: pd.DataFrame
        :rtype: None
        """
        join_cols = {}

        for col in input_df:
            if isinstance(col, tuple):
                try:
                    existing_cols = get_dict_value_by_graph(join_cols, col[:-1])
                except KeyError:
                    existing_cols = []
                set_dict_value_by_graph(join_cols, col[:-1], existing_cols + [input_df[col].rename(col[-1]), ])
            else:
                self[col] = input_df[col]

        for key, value, trace in nested_dict_leaf_iterator(join_cols, return_trace=True):
            set_dict_value_by_graph(self, trace + (key,), pd.concat(value, axis=1))

    def to_hdf5(self, file_path, included_keys=None, excluded_keys=None):
        """
        Converts all pandas objects in the dict to a single HDF5 file, with the keys used as path in the file. Strings
        are converted to bytes before writing to the file. This allows for them to be read as expected in HDFView.

        :param file_path: The path to the HDF5 file to write to.
        :type file_path: str
        :param included_keys: Keys to include in the output dataframe. If None, all keys are included.
        :type included_keys: list or tuple
        :param excluded_keys: Keys to exclude from the output dataframe. If None, no keys are excluded. Exclude takes
        precedence over include.
        :type excluded_keys: list or tuple
        :rtype: None
        """
        included_keys = self.keys() if included_keys is None else included_keys
        excluded_keys = tuple() if excluded_keys is None else excluded_keys

        # Create the HDF5 file
        with pd.HDFStore(file_path, mode="w") as store:
            for key, value, trace in nested_dict_leaf_iterator(self, return_trace=True):
                if key not in included_keys or key in excluded_keys:
                    continue

                where = "/" + "/".join(trace) + f"/{key}"
                # If the value is a pandas object, write it to the HDF5 file
                if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                    store.put(where, value)
                elif value is not None:
                    if isinstance(value, str):
                        value = value.encode("utf-8")
                    store.put(where, pd.Series(value, dtype=type(value)))  # convert to series to avoid errors
                else:
                    continue  # don't write empty dataframes

    def from_hdf5(self, file_path):
        """
        Overwrites this dict with the contents of a HDF5 file, with the path in the file as keys. Strings are converted
        from bytes to strings after reading from the file.

        :param file_path: The path to the HDF5 file to read from.
        :type file_path: str
        :rtype: None
        """
        # Create the HDF5 file
        with pd.HDFStore(file_path, mode="r") as store:
            # Loop through each key in the nested dictionary
            for k in store.keys():
                trace = k.strip("/").split("/")
                value = store.get(k).squeeze()

                if isinstance(value, (pd.Series, pd.DataFrame)) and value.empty:
                    value = None
                elif isinstance(value, bytes):
                    value = value.decode("utf-8")

                set_dict_value_by_graph(self, trace, value)
