import os
from collections import ChainMap

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from ..utility.builtin_classes.funcs import split_chunks
from ..utility.builtin_classes.iterables import filter_consecutive_duplicates
from ..utility.builtin_classes.strings import int_sequence_to_string_synonym, string_synonym_to_int_sequence, CloseMatch

from .funcs import get_value_sequence
from .old_code import series_to_value_chunks, value_chunks_to_series


def get_single_sequence_occurrence_series(input_series, target_sequence, values_before=None, values_after=None):
    if values_before is None:
        values_before = 0
    if values_after is None:
        values_after = len(target_sequence) - values_before - 1
    sequence_series = get_value_sequence(input_series, values_before=values_before, values_after=values_after,
                                         include_nan=False)
    return sequence_series.apply(lambda x: x == target_sequence)


def get_sequence_occurrence_series(input_series, *target_sequences, values_before=None, values_after=None):
    sequence_occurrence_list = []
    for target_sequence in target_sequences:
        sequence_occurrence_list.append(
            get_single_sequence_occurrence_series(input_series, target_sequence, values_before=values_before,
                                                  values_after=values_after))
    sequence_occurence_df = pd.concat(sequence_occurrence_list, axis=1)
    return sequence_occurence_df.any(axis=1)


def get_single_pattern_occurrence_series(input_series, target_pattern):
    syllable_series = input_series
    syllable_chunks = series_to_value_chunks(syllable_series)

    values_before = 0
    values_after = len(target_pattern) - 1

    syllable_chunks["target_sequence_start"] = get_sequence_occurrence_series(
        syllable_chunks["value"], target_pattern, values_before=values_before, values_after=values_after)
    syllable_chunks["part_of_sequence"] = pd.DataFrame(
        get_value_sequence(syllable_chunks["target_sequence_start"], values_before=values_after,
                           values_after=values_before, include_nan=True).to_list()
    ).any(axis=1)
    sequence_value_chunks = syllable_chunks.loc[:, ["index_start", "index_end", "part_of_sequence"]].rename(
        {"part_of_sequence": "value"}, axis=1)
    return value_chunks_to_series(sequence_value_chunks, slice_index_name=input_series.index.name)


def get_pattern_occurrence_series(input_series, *target_patterns):
    pattern_occurrence_list = []
    for target_pattern in target_patterns:
        pattern_occurrence_list.append(get_single_pattern_occurrence_series(input_series, target_pattern))
    pattern_occurrence_df = pd.concat(pattern_occurrence_list, axis=1)
    return pattern_occurrence_df.any(axis=1)


def _value_fixer(value):
    _match_row_dicts = []
    for _match in value:
        parse_result, match_start, match_end = _match
        match, mismatches = tuple(parse_result)[0]
        _match_row_dicts.append(dict(match=match, mismatches=mismatches, match_start=match_start, match_end=match_end))
    return pd.DataFrame(_match_row_dicts, index=pd.RangeIndex(len(_match_row_dicts), name="match_idx"))


def _matches_chunk_to_df(matches_chunk, index_names=None):
    if index_names is None:
        index_names = ["pattern", "strain", "light_cycle", "mouse_id_string", "contact_frame", "track_id"]
    combined_matches = dict(ChainMap(*matches_chunk))

    fixed_combined_matches = {k: _value_fixer(v) for k, v in combined_matches.items()}
    combined_matches_df = pd.concat(fixed_combined_matches, names=index_names, axis=0)
    combined_matches_df["syllable_match"] = combined_matches_df["match"].apply(string_synonym_to_int_sequence)
    combined_matches_df["syllable_pattern"] = combined_matches_df.index.to_frame()["pattern"].apply(
        string_synonym_to_int_sequence)
    combined_matches_df["n_mismatches"] = combined_matches_df["mismatches"].apply(len)
    return combined_matches_df


def matches_to_match_df(matches, index_names=None):
    match_chunks = list(split_chunks(matches, chunk_size=100))
    with Parallel(n_jobs=os.cpu_count()) as p:
        match_dfs = p(delayed(_matches_chunk_to_df)(chunk, index_names=index_names) for chunk in tqdm(match_chunks))
    return pd.concat(match_dfs, axis=0)


def apply_close_match(string_pattern, string_sequences, hamming_distance):
    matcher = CloseMatch(string_pattern, hamming_distance)
    matches = {}
    for index, substring in string_sequences.items():
        result = list(matcher.scanString(substring, overlap=True))
        if result:
            matches[(string_pattern, *index)] = result
    return matches


def get_pattern_match_df(input_sequence_df, patterns_to_match=None, pattern_filter_func=None, length=None,
                         hamming_distance=0, _keep_debug_cols=False):
    assert (patterns_to_match is not None) or (
                length is not None), "Either included_patterns or length must be provided"

    input_sequence_length_df = input_sequence_df.apply(
        lambda x: pd.Series(filter_consecutive_duplicates(x, compute_lengths=True)), axis=1)
    input_sequence_length_df.columns = ["sequences", "lengths"]

    input_sequence_series = input_sequence_length_df["sequences"]
    input_length_series = input_sequence_length_df["lengths"]
    input_start_series = input_length_series.apply(lambda x: np.cumsum([0] + x[:-1])).rename(
        "sequence_start")
    input_end_series = input_length_series.apply(lambda x: np.cumsum(x) - 1).rename(
        "sequence_end")  # -1 for end non-inclusive

    if patterns_to_match is None:
        # first walk all substrings, get all unique N-character patterns
        patterns_to_match = set()
        for substring in input_sequence_series.values:
            for i in range(len(substring) - length):
                patterns_to_match.add(tuple(substring[i:i + length]))

    if pattern_filter_func is not None:
        patterns_to_match = {p for p in patterns_to_match if pattern_filter_func(p)}

    string_patterns = {int_sequence_to_string_synonym(p) for p in patterns_to_match}
    string_sequences = input_sequence_series.apply(lambda x: int_sequence_to_string_synonym(x))

    with Parallel(n_jobs=os.cpu_count()) as p:
        matches = p(delayed(apply_close_match)(pattern, string_sequences, hamming_distance) for pattern in
                    tqdm(sorted(string_patterns)))

    match_df = matches_to_match_df(matches, index_names=["pattern"] + input_sequence_df.index.names)
    match_df = match_df.reset_index(level="pattern").join(input_start_series).join(input_end_series)
    match_df = match_df.join(input_length_series)

    match_df["sequence_position_start"] = match_df.apply(lambda x: x["sequence_start"][x["match_start"]], axis=1)
    match_df["sequence_position_end"] = match_df.apply(lambda x: x["sequence_end"][x["match_end"]-1], axis=1)

    input_index = input_sequence_df.columns
    match_df["sequence_index_start"] = match_df.apply(lambda x: input_index[x["sequence_position_start"]], axis=1)
    match_df["sequence_index_end"] = match_df.apply(lambda x: input_index[x["sequence_position_end"]], axis=1)

    match_df["syllable_lengths"] = match_df.apply(lambda x: x["lengths"][x["match_start"]:x["match_end"]], axis=1)

    if not _keep_debug_cols:
        match_df = match_df.drop(
            labels=["sequence_start", "sequence_end", "lengths", "sequence_position_start", "sequence_position_end"],
            axis=1)

    return match_df.set_index("pattern", append=True).reorder_levels(
        ["pattern"] + input_sequence_df.index.names + ["match_idx"])


def match_df_to_occurrence_df(match_df):
    def _expand_occurrence(single_match_df):
        single_match_df = single_match_df[["sequence_index_start", "sequence_index_end"]].copy()
        single_match_df = single_match_df.rename(
            {"sequence_index_start": "index_start", "sequence_index_end": "index_end"}, axis=1)
        single_match_df["value"] = True
        return value_chunks_to_series(single_match_df, slice_index_name="index")

    occurrence_df = match_df.groupby(level=match_df.index.names).apply(_expand_occurrence)
    if len(match_df) > 1:
        # Necessary due to bug in pandas. See https://github.com/pandas-dev/pandas/issues/31063
        occurrence_df = occurrence_df.unstack(level="index")
    occurrence_df = occurrence_df.fillna(False).astype(bool)

    occurrence_df = occurrence_df.droplevel(["pattern", "match_idx"], axis=0)
    occurrence_df = occurrence_df.groupby(level=occurrence_df.index.names).any()
    return occurrence_df


def match_df_to_counts_df(match_df, contact_neighbourhood_df):
    contact_count = len(contact_neighbourhood_df)
    counts_df = match_df["syllable_pattern"].value_counts(normalize=False).rename("count").reset_index()
    counts_df["relative_frequency"] = counts_df["count"] / contact_count
    return counts_df


def match_df_to_chunks_df(match_df, keep_pattern_cols=True):
    index_names = match_df.index.names
    chunks_df = match_df.reset_index()
    chunks_df = chunks_df.rename(
        {"sequence_index_start": "index_start", "sequence_index_end": "index_end", index_names[0]: "value"}, axis=1)
    chunks_df["length"] = chunks_df["index_end"] - chunks_df["index_start"] + 1
    chunks_df["grouped_index"] = chunks_df[index_names[1:-1]].apply(tuple, axis=1)
    cols_to_keep = ["index_start", "index_end", "value", "length", "grouped_index"]
    if keep_pattern_cols:
        cols_to_keep.extend(["syllable_pattern", "syllable_lengths"])
    chunks_df = chunks_df[cols_to_keep]
    return chunks_df
