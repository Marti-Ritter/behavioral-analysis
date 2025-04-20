import pandas as pd


def check_track_df(input_track_df, print_report=True):
    track_identifiers = input_track_df.index.names[:-1]
    is_multi = len(track_identifiers) > 0
    
    data_index_name = input_track_df.index.names[-1]
    feature_level_names = input_track_df.columns.names

    data_index_name_correct = data_index_name == "frame_index"
    correct_feature_level_names = feature_level_names == ["keypoint_name", "keypoint_feature"]

    contained_keypoints = input_track_df.columns.get_level_values("keypoint_name").unique()
    features_per_keypoint = {k: list(input_track_df[k].columns.get_level_values("keypoint_feature").unique()) for k in contained_keypoints}

    if print_report:
        report_string = f"""
        The input track_df is a {'MULTI' if is_multi else 'SINGLE'} track dataframe.
        The data index name is {'correct' if data_index_name_correct else f'incorrect (expected "frame_index", got "{data_index_name}")'}.
        The feature level names are {'correct' if correct_feature_level_names else f'incorrect (expected ["keypoint_name", "keypoint_feature"], got "{feature_level_names}")'}.
        {"" if not input_track_df.empty else "WARNING: The input track dataframe is empty."}
        KEYPOINTS: FEATURES
    """

        for kp, features in features_per_keypoint.items():
            report_string += f"\t\t{kp}: {features}\n"

        if is_multi:
            identifier_df = input_track_df.index.to_frame().droplevel("frame_index").index
            identifier_df = identifier_df.drop_duplicates().to_frame().reset_index(drop=True)
            report_string += f"""

        MULTI:
        The track identifiers are: {track_identifiers}
        The number of unique identifiers is: {len(identifier_df)}
        Following tracks were detected:
        {identifier_df}
        """
        print(report_string)

    return data_index_name_correct and correct_feature_level_names and not input_track_df.empty


def check_keypoint_df(input_keypoint_df, print_report=True):
    track_identifiers = input_keypoint_df.index.names[:-1]
    is_multi = len(track_identifiers) > 0

    data_index_name = input_keypoint_df.index.names[-1]
    feature_level_names = input_keypoint_df.columns.names

    data_index_name_correct = data_index_name == "keypoint_name"
    correct_feature_level_names = feature_level_names == ["keypoint_feature"]

    contained_keypoints = input_keypoint_df.index.get_level_values("keypoint_name").unique()
    contained_features = input_keypoint_df.columns

    if print_report:
        report_string = f"""
        The input keypoint_df is a {'MULTI' if is_multi else 'SINGLE'} track keypoint dataframe.
        The data index name is {'correct' if data_index_name_correct else f'incorrect (expected "keypoint_name", got "{data_index_name}")'}.
        The feature level name is {'correct' if correct_feature_level_names else f'incorrect (expected "keypoint_feature", got "{feature_level_names}")'}.

        {"" if not input_keypoint_df.empty else "WARNING: The input keypoint dataframe is empty."}
        KEYPOINTS: {list(contained_keypoints)}
        FEATURES: {list(contained_features)}
        """


        if is_multi:
            identifier_df = input_keypoint_df.index.to_frame().droplevel("keypoint_name").index
            identifier_df = identifier_df.drop_duplicates().to_frame().reset_index(drop=True)
            report_string += f"""
        MULTI:
        The track identifiers are: {track_identifiers}
        The number of unique identifiers is: {len(identifier_df)}
        Following tracks were detected:
        {identifier_df}
        """
        print(report_string)

    return data_index_name_correct and correct_feature_level_names and not input_keypoint_df.empty


def check_skeleton_df(input_skeleton_df, print_report=True):
    track_identifiers = input_skeleton_df.index.names[:-1]
    is_multi = len(track_identifiers) > 0

    data_index_name = input_skeleton_df.index.names[-1]
    feature_level_names = input_skeleton_df.columns.names

    data_index_name_correct = data_index_name == "edge_index"
    correct_feature_level_names = feature_level_names == ["edge_feature"]

    contained_edges = input_skeleton_df.index.get_level_values("edge_index").unique()
    node_cols = [col for col in input_skeleton_df.columns if col.startswith("node_")]
    contained_features = input_skeleton_df.drop(node_cols, axis=1).columns

    if not is_multi:
        contained_edges = list(input_skeleton_df.index.get_level_values("edge_index").unique())
        contained_nodes = {n: list(input_skeleton_df.loc[n][node_cols]) for n in contained_edges}
    else:
        identifier_df = input_skeleton_df.index.to_frame().droplevel("edge_index").index
        identifier_df = identifier_df.drop_duplicates().to_frame().reset_index(drop=True)

        contained_edges = {track: list(input_skeleton_df.loc[track].index.get_level_values("edge_index").unique()) for track in identifier_df}
        contained_nodes = {track: {n: list(input_skeleton_df.loc[track].loc[n][node_cols]) for n in edges} for track, edges in contained_edges.items()}

    if print_report:
        report_string = f"""
        The input skeleton_df is a {'MULTI' if is_multi else 'SINGLE'} track skeleton dataframe.
        The data index name is {'correct' if data_index_name_correct else f'incorrect (expected "keypoint_name", got "{data_index_name}")'}.
        The feature level name is {'correct' if correct_feature_level_names else f'incorrect (expected "keypoint_feature", got "{feature_level_names}")'}.

        {"" if not input_skeleton_df.empty else "WARNING: The input keypoint dataframe is empty."}
        Contained features: {list(contained_features)}
    """

        if not is_multi:
            report_string += "EDGES: NODES\n"
            for edge in contained_edges:
                report_string += f"\t{edge}: {contained_nodes[edge]}\n"
        else:
            for track, edges in contained_edges.items():
                report_string += f"\tTRACK {track}:\n\t\tEDGES: NODES\n"
                for edge in edges:
                    report_string += f"\t\t{edge}: {contained_nodes[track][edge]}\n"

        if is_multi:
            report_string += f"""
        MULTI:
        The track identifiers are: {track_identifiers}
        The number of unique identifiers is: {len(identifier_df)}
        Following tracks were detected:
        {identifier_df}
        """
        print(report_string)

    return data_index_name_correct and correct_feature_level_names and not input_skeleton_df.empty
