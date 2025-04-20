# dependencies

import pathlib

import cv2
import numpy as np
import pandas as pd
from social_arena.pandas_tools.files import build_file_df
from social_arena.pandas_tools.funcs import expand_index_from_limits, pd_limited_interpolation_series
from social_arena.pandas_tools.rle import rle_series, expand_rle_df_to_series
from social_arena.visualization.video_funcs import get_cv2_video_properties

root_dir = pathlib.Path(r"I:\2024_manuscript")

video_dir = root_dir / "videos"
raw_scores_filepath = root_dir / "ManualContactScoredRaw.csv"
unblinding_table_filepath = root_dir / "ManualContactScoredUnblinding.csv"

data_output_dir = root_dir / "data_output"
data_output_dir.mkdir(exist_ok=True)
# Manual Scoring Preprocessing
video_file_df = build_file_df(video_dir, r"(?P<recording_context>\w+)\\(?P<recording_date>\d+)_(?P<light_cycle>\w+)\\(?P<mouse_id_string>[\d&]+).*.mp4")
video_file_df["video_name"] = video_file_df["file_path"].str.rsplit("\\", n=1).str[1]
video_file_df[["fps", "frame_count"]] = video_file_df["file_path"].apply(lambda x: pd.Series(get_cv2_video_properties(x, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT)))
video_file_df
raw_score_df = pd.read_csv(raw_scores_filepath, header=0, index_col=0)
raw_score_df = raw_score_df.drop(["Treatment", "Code", "Apparatus", "Stage", "Trial"], axis=1)
raw_score_df = raw_score_df.rename({
    "Active Social Interaction : time pressed (s)": "active_social_duration",
    "Passive Social Interaction : time pressed (s)": "passive_social_duration",
}, axis=1)

hash_df = raw_score_df.value_counts(subset=["Test", "Animal"]).reset_index()
hash_df = hash_df[hash_df["count"].eq(1)]
hash_dict = hash_df.set_index("Animal")["Test"].to_dict()

unblinding_score_df = pd.read_csv(unblinding_table_filepath, header=0, index_col=0).set_index("Test")

raw_score_df = raw_score_df.join(unblinding_score_df, on="Test")
raw_score_df["video_name"] = raw_score_df["Mouse ID"] + " - " + raw_score_df["Date"].astype("datetime64[ns]").dt.strftime("%d-%m-%y") + ".mp4"
raw_score_df = raw_score_df.rename({"Type": "strain", "Cycle": "light_cycle", "Mouse ID": "mouse_id_string"}, axis=1)
raw_score_df["strain"] = raw_score_df["strain"].replace({"BL6J": "Bl6"})
raw_score_df["light_cycle"] = raw_score_df["light_cycle"].replace({"Normal": "norm", "Reverse": "reverse"})
raw_score_df
fps = 25.23  # target fps
maximum_frame = int((1200 + 5) * fps)  # 20 minutes plus 5 seconds

raw_score_df["bin_sec_start"] = raw_score_df["Segment of test"].str.extract("(\d+) - \d+ secs.").astype(int)
raw_score_df["bin_frame_start"] = (raw_score_df["bin_sec_start"] * fps).round().astype(int)
for state in ["active", "passive"]:
    raw_score_df[f"{state}_sec_length"] = raw_score_df[f"{state}_social_duration"].fillna(0)
    raw_score_df[f"{state}_sec_end"] = raw_score_df["bin_sec_start"] + raw_score_df[f"{state}_sec_length"]
    raw_score_df[[f"{state}_frame_end", f"{state}_frame_length"]] = (raw_score_df[[f"{state}_sec_end", f"{state}_sec_length"]] * fps).round().astype(int)

raw_contact_df = pd.DataFrame()
for state in ["active", "passive"]:
    hope_contact_series = None
    for (strain, light_cycle, mouse_id), grouped_df in raw_score_df.groupby(["strain", "light_cycle", "mouse_id_string"]):
        contact_index = expand_index_from_limits(grouped_df, "bin_frame_start", f"{state}_frame_end", end_inclusive=False, index_name="frame_index").index
        mouse_id_contact_series = pd.Series(data=False,
                                            index=pd.MultiIndex.from_product([[strain], [light_cycle], [mouse_id], range(0, maximum_frame)],
                                                                             names=["strain", "light_cycle", "mouse_id_string", "frame_index"]))
        mouse_id_contact_series.loc[pd.IndexSlice[strain, light_cycle, mouse_id, contact_index]] = True
        hope_contact_series = pd.concat([hope_contact_series, mouse_id_contact_series]) if hope_contact_series is not None else mouse_id_contact_series
    raw_contact_df[state] = hope_contact_series

raw_contact_df["active"].to_csv(data_output_dir / "ActiveContactSeries.csv")
raw_contact_df["passive"].to_csv(data_output_dir / "PassiveContactSeries.csv")
fill_limit = int(6 * fps)

interpolated_active_contact_series = raw_contact_df["active"].copy().replace(False, np.nan).astype(float)
interpolated_active_contact_series = pd_limited_interpolation_series(
    interpolated_active_contact_series, hard_limit=fill_limit, method="linear"
    ).fillna(0).astype(bool)

interpolated_passive_contact_series = raw_contact_df["passive"].copy().replace(False, np.nan).astype(float)
interpolated_passive_contact_series = pd_limited_interpolation_series(
    interpolated_passive_contact_series, hard_limit=fill_limit, method="linear"
    ).fillna(0).astype(bool)

interpolated_active_contact_series.to_csv(data_output_dir / "InterpolatedActiveContactSeries.csv")
interpolated_passive_contact_series.to_csv(data_output_dir / "InterpolatedPassiveContactSeries.csv")
rng = np.random.default_rng(seed=100)

interpolated_active_contact_rle = rle_series(interpolated_active_contact_series, drop_run_values=[False]).sort_index()
interpolated_active_contact_rle["distance_to_previous_contact"] = interpolated_active_contact_rle.reset_index("start_index").groupby(["strain", "light_cycle", "mouse_id_string"]).apply(
    lambda x: x["start_index"] - x["end_index"].shift(1)).values

randomized_chunk_row_dicts = []
for g, g_df in interpolated_active_contact_rle.groupby(["strain", "light_cycle", "mouse_id_string"], dropna=False):
    # shuffle lengths and spacings, select a random starting index from the spacings
    shuffled_lengths = g_df["run_length"].sample(frac=1, random_state=rng).values
    shuffled_spacings = g_df["distance_to_previous_contact"].dropna().sample(frac=1, random_state=rng).values
    next_start_index = 0

    randomized_chunk_row_dicts.append({
        "strain": g[0], "light_cycle": g[1], "mouse_id_string": g[2],
        "run_length": shuffled_lengths[0], "distance_to_previous_contact": np.nan,
        "start_index": int(next_start_index), "end_index": int(next_start_index + shuffled_lengths[0]),
        "run_value": True
    })
    next_start_index += shuffled_lengths[0]

    for length, spacing in zip(shuffled_lengths[1:], shuffled_spacings):
        next_start_index += spacing
        randomized_chunk_row_dicts.append({
            "strain": g[0], "light_cycle": g[1], "mouse_id_string": g[2],
            "run_length": length, "distance_to_previous_contact": spacing,
            "start_index": int(next_start_index), "end_index": int(next_start_index + length),
            "run_value": True
        })
        next_start_index += length

randomized_contact_rle = pd.DataFrame(randomized_chunk_row_dicts)
for g, g_df in randomized_contact_rle.groupby(["strain", "light_cycle", "mouse_id_string"], dropna=False):
    corresponding_active_contact_series = interpolated_active_contact_series.loc[g]
    reference_end_index = corresponding_active_contact_series.index.max()
    current_end_index = g_df["end_index"].max()
    random_shift = int(rng.uniform(0, reference_end_index - current_end_index))
    randomized_contact_rle.loc[g_df.index, ["start_index", "end_index"]] += random_shift

randomized_contact_rle = randomized_contact_rle.set_index(interpolated_active_contact_rle.index.names)
randomized_contact_rle["end_index"] -= 1
randomized_contact_series = expand_rle_df_to_series(randomized_contact_rle, index_name="frame_index", series_name="randomized")
randomized_contact_series = randomized_contact_series.reindex(interpolated_active_contact_series.index, fill_value=False)
randomized_contact_series.to_csv(data_output_dir / "RandomizedContactSeries.csv")