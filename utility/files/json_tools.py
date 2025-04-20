import json


def find_valid_json(s, raise_error=False, longest_only=True):
    """
    Finds and returns either the longest or all valid JSON-format substrings in a given string,
    sorted by length in descending order.

    :param s: The input string to search for JSON-format substrings.
    :type s: str
    :param raise_error: If True, raise a ValueError if no valid JSON substrings are found.
        If False (default), return an empty list if no valid JSON substrings are found.
    :type raise_error: bool
    :param longest_only: If True (default), return only the longest valid JSON substring found in `s`.
        If False, return all valid JSON substrings found in `s`.
    :type longest_only: bool
    :return: A list of strings representing either the longest or all valid JSON-format substrings found in `s`,
        depending on the value of `longest_only`, sorted by length in descending order.
        If no valid JSON substrings are found and `raise_error` is False, an empty list is returned.
    :rtype: list of str
    :raises ValueError: If `raise_error` is True and no valid JSON substrings are found.
    """
    valid_jsons = []
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            try:
                json_obj = json.loads(s[i:j])
                valid_jsons.append(s[i:j])
            except ValueError:
                pass
    if not valid_jsons and raise_error:
        raise ValueError("No valid JSON-format substrings found in input string.")

    valid_jsons.sort(key=len, reverse=True)
    if longest_only:
        if valid_jsons:
            return [valid_jsons[0]]
        else:
            return []
    else:
        return valid_jsons
