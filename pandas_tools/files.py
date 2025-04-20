# built-in modules
import os
import re

# third-party modules
import pandas as pd
from tqdm.auto import tqdm
# local modules
from ..utility.builtin_classes.iterables import ensure_list
from ..utility.files.file_tools import find_files

from ..utility.regex_funcs import extract_regex_from_unstructured_string, extract_regex_from_string

# patterns as they are created by the current version of processing scripts
common_file_regex_patterns = {
    "trace_csv": r"^(?P<project_name>.+)_(?P<session_num>\d+)"
                 r"_TCR(?:_txtexport_notitle_Trace)?_Animal ?ID (?P<animal_id>\d+)(?:_Trace)?\.csv$",
    "event_csv": r"^(?P<project_name>.+)_box(?P<box_num>\d+)(?:_.*)?_(?P<session_num>\d+)"
                 r"_TCR(?:_txtexport_notitle_Events)?(?:_Animal ?ID (?P<animal_id>\d+)_Events)?\.csv$",
    "meta_json": r"^(?P<project_name>.+)_box(?P<box_num>\d+)_(?P<session_num>\d+)_TCR(?:_txtexport)?_meta\.json$",
    "rfid_csv": r"^(?P<project_name>.+) (?P<session_num>\d+) RFID\.csv$",
}


def build_regex_df_old(input_strings, pattern_string_or_dict, **extraction_kwargs):
    output_rows = []
    for input_str in input_strings:
        if isinstance(pattern_string_or_dict, dict):
            extraction_dict = extract_regex_from_unstructured_string(input_str, pattern_string_or_dict, **extraction_kwargs)
        else:
            search_result = re.search(pattern_string_or_dict, input_str, **extraction_kwargs)
            if search_result is None:
                continue
            extraction_dict = search_result.groupdict()
        extraction_dict["input"] = input_str
        output_rows.append(extraction_dict)
    return pd.DataFrame(output_rows)


def build_regex_df(input_strings, pattern_string_or_dict, return_spans=False, **extraction_kwargs):
    output_rows = []
    for input_str in input_strings:
        if isinstance(pattern_string_or_dict, dict):
            extraction_output = extract_regex_from_unstructured_string(input_str, pattern_string_or_dict,
                                                                       return_remaining=False, return_span_dict=True,
                                                                       **extraction_kwargs)
            if extraction_output is None:
                continue
            extraction_dict, span_dict = extraction_output
        else:
            extraction_output = extract_regex_from_string(input_str, pattern_string_or_dict, **extraction_kwargs)
            if extraction_output is None:
                continue

            extraction_result, span_result, _remaining_string = extraction_output

            if isinstance(extraction_result, dict):
                extraction_dict = extraction_result
                span_dict = span_result
            elif isinstance(extraction_result, tuple):
                extraction_dict = dict(enumerate(extraction_result))
                span_dict = dict(enumerate(span_result))
            else:
                extraction_dict = {"match": extraction_result}
                span_dict = {"match": span_result}

        extraction_dict["input"] = input_str
        span_dict = {str(k) + "_span": v for k, v in span_dict.items()}
        output_rows.append({**extraction_dict, **span_dict} if return_spans else extraction_dict)
    return pd.DataFrame(output_rows)


def build_file_df(root_directory, file_pattern_string_or_dict, file_extension=None,
                  find_kws=None, extraction_kws=None, return_spans=False, raise_empty=True):
    """
    Builds a DataFrame from a list of files in a directory, using a regex pattern to extract information from the file
    names. The DataFrame will have a column for each named group in the regex pattern, plus a column for the file path.

    :param root_directory: directory to search for files
    :type root_directory: str
    :param file_pattern_string_or_dict: file pattern to use for extracting information from the file names. If a string,
        it will be used as a regex pattern. If a dict, it will be used as the `pattern_dict` argument for
        `extract_from_unstructured_string`.
    :type file_pattern_string_or_dict: str or dict
    :param file_extension: optional file extension to search for (e.g. ".csv"). If None, all files will be searched.
    :type file_extension: str or None
    :param find_kws: optional keyword arguments to pass to `find_files`
    :type find_kws: dict or None
    :param extraction_kws: optional keyword arguments to pass to `extract_from_unstructured_string` or `re.search`
    :type extraction_kws: dict or None
    :param return_spans: whether to return the span of each match in addition to the match itself. If True, the
        DataFrame will have a column for each named group in the regex pattern, plus a column for the file path, plus a
        column for each named group in the regex pattern with "_span" appended to the name.
    :type return_spans: bool
    :param raise_empty: whether to raise an error if there are no matches for the pattern(s). If False, an empty
        DataFrame will be returned.
    :type raise_empty: bool
    :return: A DataFrame with a row for each file that matches the pattern(s), and a column for each named group in the
        pattern(s) and the file path.
    :rtype: pd.DataFrame
    """
    file_extension = "" if file_extension is None else file_extension
    find_kws = dict() if find_kws is None else find_kws
    extraction_kws = dict() if extraction_kws is None else extraction_kws

    found_files = find_files(root_directory, file_extension, **find_kws)
    relative_files = list(map(os.path.relpath, found_files, [root_directory]*len(found_files)))
    file_regex = build_regex_df(relative_files, file_pattern_string_or_dict, return_spans=return_spans,
                                **extraction_kws)
    if not file_regex.empty:
        file_regex = file_regex.rename({"input": "file_path"}, axis=1)
        file_regex["file_path"] = file_regex["file_path"].apply(lambda x: os.path.join(root_directory, x))
    elif raise_empty:
        raise RuntimeError(
            "There were zero matches for the pattern(s)! Input was {}.".format(file_pattern_string_or_dict))
    return file_regex


def build_multi_file_df(root_directory, file_type_patterns_dict, file_extension=None,
                        find_kws=None, extraction_kws=None, return_spans=False, raise_empty=True):
    """
    Builds a concatenated DataFrame from multiple regex patterns, each of which is applied to a subset of files
    in the root directory, as specified by the corresponding keys in the file_type_patterns_dict. Each sub-dataframe
    contains a column indicating the file type and is generated by `build_file_df`.

    :param root_directory: directory to search for files
    :type root_directory: str
    :param file_type_patterns_dict: dictionary mapping file type keys to file pattern(s) to use for extracting
        information from the file names. Each value can be either a string (used as a regex pattern), or a dict (used as
        the `pattern_dict` argument for `extract_from_unstructured_string`). The keys will be used as a column in the
        output DataFrame.
    :type file_type_patterns_dict: dict
    :param file_extension: optional file extension to search for (e.g. ".csv"). If None, all files will be searched.
    :type file_extension: str or None
    :param find_kws: optional keyword arguments to pass to `find_files`
    :type find_kws: dict or None
    :param extraction_kws: optional keyword arguments to pass to `extract_from_unstructured_string` or `re.search`
    :type extraction_kws: dict or None
    :param return_spans: whether to return the span of each match in addition to the match itself. If True, the
        DataFrame will have a column for each named group in the regex pattern, plus a column for the file path, plus a
        column for each named group in the regex pattern with "_span" appended to the name.
    :type return_spans: bool
    :param raise_empty: whether to raise an error if there are no matches for any of the patterns. If False, an empty
        DataFrame will be returned.
    :type raise_empty: bool
    :return: A concatenated DataFrame with a row for each file that matches any of the patterns, and a column for each
        named group in the pattern(s), the file type, and the file path.
    :rtype: pd.DataFrame
    """
    file_extension = "" if file_extension is None else file_extension
    find_kws = dict() if find_kws is None else find_kws
    extraction_kws = dict() if extraction_kws is None else extraction_kws

    # initialize empty list to store sub-dataframes
    sub_dataframes = []

    for file_type, pattern_string_or_dict in file_type_patterns_dict.items():
        # call build_file_df for each file pattern, and add a column with the corresponding file type
        sub_dataframe = build_file_df(root_directory, pattern_string_or_dict, file_extension=file_extension,
                                      find_kws=find_kws, extraction_kws=extraction_kws, return_spans=return_spans,
                                      raise_empty=raise_empty)
        sub_dataframe["file_type"] = file_type
        sub_dataframes.append(sub_dataframe)

    # concatenate sub-dataframes into final dataframe
    multi_file_df = pd.concat(sub_dataframes).reset_index(drop=True)

    return multi_file_df


def pd_get_read_func(file_extension):
    extension_to_func = {
        ".csv": pd.read_csv,
        ".xlsx": pd.read_excel,
        ".pkl": pd.read_pickle
    }
    return extension_to_func[file_extension.lower()]


def pd_get_write_func(file_extension):
    extension_to_func = {
        ".csv": pd.DataFrame.to_csv,
        ".xlsx": pd.DataFrame.to_excel,
        ".pkl": pd.DataFrame.to_pickle
    }
    return extension_to_func[file_extension.lower()]


def pd_read_file(input_path, **read_kws):
    return pd_get_read_func(os.path.splitext(input_path)[1])(input_path, **read_kws)


def pd_write_file(output_df, output_path, **write_kws):
    pd_get_write_func(os.path.splitext(output_path)[1])(output_df, output_path, **write_kws)


def pd_write_files(output_path_to_df_dict, **write_kws):
    for output_path, output_df in output_path_to_df_dict.items():
        pd_write_file(output_df, output_path, **write_kws)


def pd_read_and_accumulate_files(input_path_or_list, pd_accumulate_func=None, use_tqdm=False,
                                 read_kws=None, accumulate_kws=None):
    read_kws = dict() if read_kws is None else read_kws
    accumulate_kws = dict() if accumulate_kws is None else accumulate_kws

    input_path_list = ensure_list(input_path_or_list)
    iterator = tqdm(input_path_list) if use_tqdm else input_path_list

    input_df = pd.DataFrame()
    for input_path in iterator:
        path_df = pd_read_file(input_path, **read_kws)
        path_df["input_path"] = input_path
        input_df = pd.concat(
            [input_df, path_df if pd_accumulate_func is None else pd_accumulate_func(path_df, **accumulate_kws)])
    return input_df


def pd_transform(input_path_or_list, pd_accumulate_func=None, pd_transform_func=None, output_path_or_list=None,
                 read_kws=None, accumulate_kws=None, transform_kws=None, write_kws=None):
    transform_kws = dict() if transform_kws is None else transform_kws
    write_kws = dict() if write_kws is None else write_kws

    input_df = pd_read_and_accumulate_files(input_path_or_list,
                                            pd_accumulate_func=pd_accumulate_func,
                                            read_kws=read_kws, accumulate_kws=accumulate_kws)

    transform_result = input_df if pd_transform_func is None else pd_transform_func(input_df, **transform_kws)
    output_path_to_df_dict = transform_result if isinstance(transform_result, dict) else {p: df for p, df in zip(
        ensure_list(output_path_or_list), ensure_list(transform_result))}
    pd_write_files(output_path_to_df_dict, **write_kws)


def pd_multi_transform(input_output_paths_dict, use_tqdm=True, **kwargs):
    iterator = tqdm(input_output_paths_dict.items()) if use_tqdm else input_output_paths_dict.items()
    for input_path_or_list, output_path_or_list in iterator:
        pd_transform(input_path_or_list, output_path_or_list=output_path_or_list, **kwargs)


def label_from_regex_dict(input_string, label_regex_dict, regex_func=re.match):
    for label, regex_pattern in label_regex_dict.items():
        res = regex_func(regex_pattern, input_string)
        if res:
            return label


def label_series_from_regex_dict(input_series, label_regex_dict):
    return input_series.apply(lambda x: label_from_regex_dict(x, label_regex_dict))


def load_excel_workbook(workbook_path, file_kwargs=None, parse_kwargs=None):
    file_kwargs = dict() if file_kwargs is None else file_kwargs
    parse_kwargs = dict() if parse_kwargs is None else parse_kwargs
    excel_file = pd.ExcelFile(workbook_path, **file_kwargs)
    return {sheet_name: excel_file.parse(sheet_name, **parse_kwargs) for sheet_name in excel_file.sheet_names}


def create_file_df_from_regex(directory_path, regex_pattern, sep="_", file_suffixes=None, type_dict=None,
                              apply_dict=None, force_match=True):
    """

    :param directory_path:
    :type directory_path: str
    :param regex_pattern:
    :type regex_pattern: str
    :param sep:
    :type sep: str
    :param file_suffixes:
    :type file_suffixes: str
    :param type_dict:
    :type type_dict: dict
    :param apply_dict:
    :type apply_dict: dict
    :param force_match:
    :type force_match: bool
    :return:
    :rtype: pandas.DataFrame
    """
    if isinstance(directory_path, list):
        file_df = pd.DataFrame()
        for directory in directory_path:
            directory_df = create_file_df_from_regex(directory, regex_pattern, sep=sep, file_suffixes=file_suffixes,
                                                     type_dict=type_dict, apply_dict=apply_dict,
                                                     force_match=force_match)
            file_df = pd.concat((file_df, directory_df))
        return file_df

    required_suffixes = tuple(ensure_list(file_suffixes)) if file_suffixes is not None else None

    files = [f for f in os.listdir(directory_path) if
             (required_suffixes is None or f.endswith(required_suffixes)
              ) and (not force_match or re.match(regex_pattern, f))]
    file_df = pd.DataFrame(files, columns=["file_path"])

    if not file_df.empty:
        file_df = pd.concat((file_df, file_df["file_path"].str.extract(regex_pattern, expand=True)), axis=1)
        if type_dict is not None:
            file_df = file_df.astype(type_dict)
        if apply_dict is not None:
            for col, function in apply_dict.items():
                file_df[col] = file_df[col].apply(function)
    file_df["file_path"] = file_df["file_path"].apply(lambda x: os.path.join(directory_path, x))
    return file_df


def create_file_df_old(directory_path, fields, sep="_", file_suffixes=None):
    required_suffixes = tuple(ensure_list(file_suffixes)) if file_suffixes is not None else None

    files = [f for f in os.listdir(directory_path) if file_suffixes is None or f.endswith(required_suffixes)]
    file_df = pd.DataFrame(files, columns=["file_path"])

    if not file_df.empty:
        file_df[fields] = file_df["file_path"].apply(
            lambda x: os.path.splitext(x)[0]).str.split(sep, expand=True).iloc[:, :len(fields)]
    else:
        file_df[fields] = None

    return file_df
