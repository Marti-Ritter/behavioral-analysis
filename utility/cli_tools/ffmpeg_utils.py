import subprocess

import pandas as pd


def parse_ffmpeg_header(header):
    """
    Parses the header of the output of the `ffmpeg -codecs` command and returns a pandas DataFrame
    containing information about each feature that is supported by ffmpeg.

    The DataFrame has the following columns: index, value, and feature.

    :param header: The header of the output of the `ffmpeg -codecs` command.
    :type header: str
    :returns: A DataFrame containing information about each feature that is supported by ffmpeg.
    :rtype: pandas.DataFrame
    """
    feature_dict_rows = []
    header = header.replace("Codecs:", "", 1)

    for line in header.strip().split('\n'):
        feature_code, feature_description = line.strip().split(' = ', 1)

        for i, v in enumerate(feature_code):
            if v != ".":
                feature_dict_rows.append({"index": i, "value": v, "feature": feature_description})
                break

    return pd.DataFrame(feature_dict_rows)


def parse_ffmpeg_codecs(codec_lines):
    """
    Parses the output of the `ffmpeg -codecs` command and returns a pandas DataFrame containing information about
    each codec that is supported by ffmpeg.

    :param codec_lines: The output of the `ffmpeg -codecs` command.
    :type codec_lines: str
    :returns: A DataFrame containing information about each codec that is supported by ffmpeg.
    :rtype: pandas.DataFrame
    """
    codec_dict_rows = []
    for row in codec_lines.strip().split('\n'):
        feature_code, name, description = row.strip().split(maxsplit=2)
        codec_dict_rows.append({"feature_code": feature_code, "name": name, "description": description})

    return pd.DataFrame(codec_dict_rows)


def get_ffmpeg_codec_info():
    """
    Runs the `ffmpeg -codecs` command and returns two pandas DataFrames containing information about the
    features and codecs that are supported by ffmpeg.

    :returns: Two DataFrames containing information about the features and codecs that are supported by ffmpeg.
    :rtype: (pandas.DataFrame, pandas.DataFrame)
    """
    result = subprocess.run(['ffmpeg', '-codecs'], stdout=subprocess.PIPE)
    result.check_returncode()
    output = result.stdout.decode('utf-8')
    header, _, codec_lines = output.partition("-------")
    return parse_ffmpeg_header(header), parse_ffmpeg_codecs(codec_lines)


def parse_codec_feature_code(feature_code, parsed_header_df=None):
    """
    Parses a feature code and returns a pandas Series indicating which features are supported by the codec.

    :param feature_code: A feature code.
    :type feature_code: str
    :param parsed_header_df: A DataFrame containing information about the features that are supported by ffmpeg.
    :type parsed_header_df: pandas.DataFrame, optional
    :returns: A Series indicating which features are supported by the codec.
    :rtype: pandas.Series
    """
    parsed_header_df = get_ffmpeg_codec_info()[0] if parsed_header_df is None else parsed_header_df

    feature_series = pd.Series(data=False, index=parsed_header_df["feature"].values)
    for _index, row in parsed_header_df.iterrows():
        if feature_code[int(row["index"])] == row["value"]:
            feature_series[row["feature"]] = True

    return feature_series


def get_ffmpeg_codecs_df():
    """
    Get information about the codecs supported by ffmpeg and the features present in each codec.

    :returns: A DataFrame containing codec information and feature information
    :rtype: pandas.DataFrame
    """
    header_df, codecs_df = get_ffmpeg_codec_info()
    return codecs_df.join(
        codecs_df["feature_code"].apply(lambda code: parse_codec_feature_code(code, parsed_header_df=header_df)))
