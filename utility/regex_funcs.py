import re


def extract_regex_from_string(input_string, regex_pattern, flags=()):
    """
    Extracts a regex match from a string. If the regex pattern contains a named group, the output will be a dictionary
    with the group name as the key and the match as the value. If the regex pattern contains multiple named groups, the
    output will be a dictionary with the group names as the keys and a tuple of the matches as the value. If the regex
    pattern does not contain a named group, the output will be the match itself. If the regex pattern contains multiple
    groups, the output will be a tuple of the matches. If the regex pattern does not match the input string, None will
    be returned. If the regex pattern matches the input string, the match will be removed from the input string and the
    remaining string will be returned.

    :param input_string:
    :type input_string:
    :param regex_pattern:
    :type regex_pattern:
    :param flags:
    :type flags:
    :return:
    :rtype:
    """
    remaining_string = str(input_string)
    res = re.search(regex_pattern, remaining_string, *flags)
    if res is None:
        return None

    if res.groupdict():
        regex_output = res.groupdict()
        regex_span = {k: res.span(k) for k in res.groupdict().keys()}
    elif res.groups():
        regex_output = res.group(1) if len(res.groups()) == 1 else res.groups()
        regex_span = res.span(1) if len(res.groups()) == 1 else tuple(
            (res.span(i) for i in range(1, len(res.groups()) + 1)))
    else:
        regex_output = res.group()
        regex_span = res.span()

    remaining_string = remaining_string.replace(str(res.group()), "")

    return regex_output, regex_span, remaining_string


def extract_regex_from_unstructured_string(input_string, pattern_dict, no_match_value=None, strip_chars="_ -",
                                           return_remaining=True, return_span_dict=False, flags=()):
    """
    Extracts values from a string using a dictionary of regex patterns. The regex patterns are used to search the
    string for matches, and the values of the dictionary are used as keys in the output dictionary. The values of the
    output dictionary are the matches found in the string. If a match is not found for a given key, the value of the
    output dictionary will be set to `no_match_value`, unless that is set to None. If `remaining_key` is not None, the
    remaining string after all matches have been found will be added to the output dictionary with the key
    `remaining_key`. If `return_span_dict` is True, an additional span dictionary will contain a key for each key in the
    input dictionary, and the value of each key will be a tuple containing the start and end indices of the match in
    the input string. If `return_span_dict` is False, the span dictionary will not be returned.

    :param input_string: A string to search for matches.
    :type input_string: str
    :param pattern_dict: A dictionary of regex patterns to search for in the input string.
    :type pattern_dict: dict
    :param return_remaining: If True, return the remaining string after all matches have been found.
    :type return_remaining: bool
    :param no_match_value: The value to use for keys that do not have a match in the input string.
    :type no_match_value: object
    :param strip_chars: Characters to strip from the beginning and end of the remaining string.
    :type strip_chars: str
    :param return_span_dict: If True, return a span dictionary in addition to the output dictionary.
    :type return_span_dict: bool
    :param flags: Flags to pass to the re.search function. See https://docs.python.org/3/library/re.html#re.RegexFlag
    :type flags: tuple
    :return: A tuple containing the output dictionary and the span dictionary, if `return_span_dict` is True.
    :rtype: (dict, dict) or dict
    """
    remaining_string = str(input_string)
    output_dict = {}
    span_dict = {}
    for key, pattern in pattern_dict.items():
        extracted_value = extract_regex_from_string(remaining_string, pattern, flags=flags)
        if extracted_value is None:
            if no_match_value is not None:
                output_dict[key] = no_match_value
            continue

        if return_span_dict:
            output_dict[key], remaining_string, span_dict[key] = extracted_value
        else:
            output_dict[key], remaining_string = extracted_value

    output = (output_dict,)
    if return_remaining:
        output += remaining_string.strip(strip_chars),
    if return_span_dict:
        output += span_dict,

    return output if len(output) > 1 else output[0]


def replace_regex_dict(input_string, pattern_replacement_dict):
    for pattern, replacement in pattern_replacement_dict.items():
        input_string = re.sub(pattern, replacement, input_string)
    return input_string


def regex_search_with_fallback_value(regex_pattern, input_string, fallback_value=None):
    search_result = re.search(regex_pattern, input_string)
    if search_result is not None:
        return search_result.groups()[0]
    else:
        return fallback_value
