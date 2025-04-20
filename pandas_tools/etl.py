# additional installs
import pandas as pd
from tqdm import tqdm
# dependencies
from ..utility.builtin_classes.iterables import ensure_list

# local modules
from .files import pd_read_file, build_file_df


def pd_read_and_accumulate_files(input_path_or_list, pd_accumulate_func=None, use_tqdm=False,
                                 read_kws=None, accumulate_kws=None):
    read_kws = dict() if read_kws is None else read_kws
    accumulate_kws = dict() if accumulate_kws is None else accumulate_kws

    input_path_list = ensure_list(input_path_or_list)
    iterator = tqdm(input_path_list) if use_tqdm else input_path_list

    output_pd = None
    for input_path in iterator:
        path_df = pd_read_file(input_path, **read_kws)
        path_df["input_path"] = input_path
        accumulation_result = path_df if pd_accumulate_func is None else pd_accumulate_func(path_df, **accumulate_kws)
        if output_pd is None:
            output_pd = accumulation_result
        else:
            output_pd = pd.concat([output_pd, accumulation_result])
    return output_pd


def pd_load_and_stitch_df_from_files(root_directory, file_pattern_string_or_dict, file_extension="csv",
                                     find_kws=None, extraction_kws=None, read_kws=None, _head=None):
    read_kws = dict() if read_kws is None else read_kws

    file_regex = build_file_df(root_directory, file_pattern_string_or_dict, file_extension,
                               find_kws=find_kws, extraction_kws=extraction_kws)

    def transfer_properties(input_df, properties_df):
        output_df = input_df.copy()
        properties_dict = properties_df.loc[input_df["input_path"].unique()[0]].to_dict()
        for key, value in properties_dict.items():
            output_df[key] = value
        return output_df

    paths_to_read = file_regex["file_path"].to_list() if _head is None else file_regex["file_path"].head(
        _head).to_list()

    stitched_df = pd_read_and_accumulate_files(paths_to_read,
                                               pd_accumulate_func=transfer_properties, read_kws=read_kws,
                                               accumulate_kws={"properties_df": file_regex.set_index("file_path")},
                                               use_tqdm=True)
    return stitched_df
