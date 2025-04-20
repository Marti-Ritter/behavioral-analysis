from typing import MutableMapping

from .sets import get_list_intersection, get_list_union, condense_sets_to_mds


def set_dict_value_by_graph(input_dict, graph, value, force_graph=True):
    """
    Set a value in a nested dict by a graph. If force_graph is True, the graph will be created if it does not exist. If
    force_graph is False, a KeyError will be raised if the graph does not exist.

    :param input_dict: A nested dict
    :type input_dict: dict
    :param graph: A list of keys to traverse the dict
    :type graph: list or tuple
    :param value: The value to set
    :type value: Any
    :param force_graph: Whether to create the graph if it does not exist
    :type force_graph: bool
    :rtype: None
    :raises KeyError: If force_graph is False and the graph does not exist
    """
    if not graph[0] in input_dict:
        if not force_graph:
            raise KeyError("Graph does not match dict structure!")
        input_dict[graph[0]] = {}
    if isinstance(input_dict, (dict, MutableMapping)) and len(graph) == 1:
        input_dict[graph[0]] = value
        return
    else:
        if not isinstance(input_dict[graph[0]], (dict, MutableMapping)):
            input_dict[graph[0]] = {}
        set_dict_value_by_graph(input_dict[graph[0]], graph[1:], value)


def get_dict_value_by_graph(input_dict, graph):
    """
    Get a value in a nested dict by a graph. If the graph does not exist, a KeyError will be raised.

    :param input_dict: A nested dict
    :type input_dict: dict
    :param graph: A list of keys to traverse the dict
    :type graph: list or tuple
    :return: The value at the end of the graph traversal
    :rtype: Any
    :raises KeyError: If the graph does not exist
    """
    if not graph[0] in input_dict:
        raise KeyError("Graph does not match dict structure!")
    if isinstance(input_dict, (dict, MutableMapping)) and graph[0] in input_dict and len(graph) == 1:
        return input_dict[graph[0]]
    else:
        if not isinstance(input_dict[graph[0]], (dict, MutableMapping)):
            raise KeyError("Graph does not match dict structure!")
        return get_dict_value_by_graph(input_dict[graph[0]], graph[1:])


def nested_dict_leaf_iterator(nested_dict, return_trace=False, _trace=tuple()):
    """
    Iterate over the leaves of a nested dict. If return_trace is True, the trace of the leaf will be returned as well.

    :param nested_dict: A nested dict
    :type nested_dict: dict
    :param return_trace: Whether to return the trace of the leaf
    :type return_trace: bool
    :param _trace: The trace of the leaf
    :type _trace: tuple
    :return: A tuple of the key, value, and trace of the leaf if return_trace is True, otherwise a tuple of the key and
    value of the leaf
    :rtype: (Any, Any, tuple) or (Any, Any)
    """
    for key, value in nested_dict.items():
        if isinstance(value, (dict, MutableMapping)):
            yield from nested_dict_leaf_iterator(value, return_trace=return_trace, _trace=(*_trace, key))
        else:
            yield (key, value, _trace) if return_trace else (key, value)


def nested_dict_walk(nested_dict, bottom_up=False):
    """
    A function mimicking os.walk for nested dicts. If bottom_up is True, the nested dict will be traversed bottom-up,
    i.e. the deepest leaves will be returned first. Calls nested_dict_leaf_iterator initially, then iterates over the
    leaves and returns the graph and value of each leaf.

    :param nested_dict: A nested dict
    :type nested_dict: dict
    :param bottom_up: Whether to traverse the nested dict bottom-up
    :type bottom_up: bool
    :return: A tuple of the graph and value of each leaf
    :rtype: (tuple, Any)
    """
    iterated_leafs = list(nested_dict_leaf_iterator(nested_dict, return_trace=True))

    if bottom_up:
        iterated_leafs = sorted(iterated_leafs, key=lambda x: len(x[2]), reverse=True)
    else:
        iterated_leafs = sorted(iterated_leafs, key=lambda x: len(x[2]))

    for key, value, trace in iterated_leafs:
        yield (*trace, key), value


def check_dict_equality(dict_list):
    """
    Check the equality of a list of dicts. Returns a tuple of shared keys, common values, unique keys, and variable
    keys. Shared keys are keys that are present in all dicts. Common values are values that are the same for all dicts.
    Unique keys are keys that are present in only one dict. Variable keys are keys that are present in more than one dict
    but have different values.

    :param dict_list: A list of dicts
    :type dict_list: list of dict
    :return: A tuple of shared keys, common values, unique keys, and variable keys
    :rtype: (set, list, dict, list)
    """
    shared_keys = get_list_intersection([set(d.keys()) for d in dict_list])
    common_values = {key: value for key, value in dict_list[0].items() if
                     key in shared_keys and all(d[key] == dict_list[0][key] for d in dict_list)}
    unique_keys = [
        set(dict_list[i].keys()).difference(get_list_union([set(d.keys()) for d in dict_list[:i] + dict_list[i + 1:]]))
        for i in range(len(dict_list))]
    variable_keys = [key for key in shared_keys if key not in common_values.keys()]

    return shared_keys, common_values, unique_keys, variable_keys


def update_default_dict(default_dict, update_dict=None, ignore_values=(None,), allow_new_attributes=False):
    """
    How to do update a dict of default values with an optionally incomplete dict.
    Useful at the start of a function that has hard-coded defaults and may be supplied with a set of ignored values,
    such as None, and a boolean flag whether to limit the return to keys included in the default dict.

    :param default_dict: dict with default values for keys
    :type default_dict: dict
    :param update_dict: dict with keys to update and their updated values
    :type update_dict: dict
    :param ignore_values: iterable with values that will cause the entry in update_dict to be ignored
    :type ignore_values: tuple or list or set
    :param allow_new_attributes: whether to allow keys from update_dict when they are not included in default_dict
    :type allow_new_attributes: bool
    :return: default_dict updated with updated dict, with ignore_values staying at default/being ignored,
    and unknown keys optionally ignored
    :rtype: dict
    """

    update_dict = update_dict if update_dict is not None else {}
    filtered_update_dict = {key: value for key, value in update_dict.items() if value not in ignore_values}
    permitted_keys = {*default_dict.keys(), *(filtered_update_dict.keys() if allow_new_attributes else [])}
    return {key: filtered_update_dict[key] if key in filtered_update_dict else default_dict[key] for key
            in permitted_keys}


def mirror_relationship_dict(relationship_dict):
    """
    Mirrors a dict of relationships between objects, e.g. {"A": {"B": 1, "C": 2}, "B": {"C": 3}} to
    {"B": {"A": 1}, "C": {"A": 2, "B": 3}}.

    :param relationship_dict: dict with relationships between objects, e.g. {"A": {"B": 1, "C": 2}, "B": {"C": 3}}
    :type relationship_dict: dict
    :return: dict with all relationships mirrored
    :rtype: dict
    """
    output_dict = relationship_dict.copy()
    for level1, sub_dict in relationship_dict.items():
        for level2, value in sub_dict.items():
            if level1 == level2:
                continue
            if level2 not in output_dict:
                output_dict[level2] = {}
            output_dict[level2][level1] = value
    return output_dict


def dicts_to_dict_of_tuples(*input_dicts):
    """
    Converts a list of dicts to a dict of tuples, where the keys of the dicts are the keys of the output dict, and the
    values of the dicts are the values of the output dict.

    :param input_dicts: A list of dicts
    :type input_dicts: list
    :return: A dict of tuples
    :rtype: dict
    """
    all_keys = set().union(*input_dicts)
    return {key: tuple(input_dict[key] for input_dict in input_dicts if key in input_dict) for key in all_keys}


def numerize_string_values_in_dict(input_dict):
    """
    Converts all string values in a dictionary to their numerical equivalents, if possible. If the string value cannot
    be converted to a number, then the string value is left as is.

    :param input_dict: A dictionary
    :type input_dict:  dict
    :return: The dictionary with all string values converted to their numerical equivalents, if possible
    :rtype: dict
    """
    output_dict = {}
    for graph, value in nested_dict_walk(input_dict):
        if isinstance(value, str):
            for numerical_type in (int, float):
                try:
                    value = numerical_type(value)
                    break
                except ValueError:
                    pass
            set_dict_value_by_graph(output_dict, graph, value)
        else:
            set_dict_value_by_graph(output_dict, graph, value)
    return output_dict


def _update_default_dict_old(update_dict=None):
    """
    How to do update a dict of default values with an optionally incomplete set of keys and values.
    Useful at the start of a function that has hard-coded defaults and may be supplied with a alternative dict as arg.
    :param update_dict:
    :type update_dict:
    :return:
    :rtype:
    """
    default_dict = {"a": 1}
    return dict(default_dict, **(update_dict if update_dict is not None else {}))


def invert_flat_dict(input_dict, unpack_single_values=False, unpack_list_values=True):
    """
    Inverts a passed flat (not nested) dictionary (key: value) and returns an inverted mapping (value: key). If the
    initial mapping is to a list, and unpack_list_values is True, then the output will map each of the list elements to
    the initial key. If unpack_single_values is True, the output dictionary will be scanned for len == 1 values and
    those values will be unpacked/ their first element will take their place.

    :param input_dict: A dictionary
    :type input_dict: dict
    :param unpack_single_values: Whether to select the first element of values of length 1 before returning output_dict
    :type unpack_single_values: bool
    :param unpack_list_values: Whether to unpack list values in the passed dict to multiple keys in the output_dict
    :type unpack_list_values: bool
    :return: A dict with inverted mapping
    :rtype: dict
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list) and unpack_list_values:
            output_dict.update({val: key for val in value})
        else:
            output_dict.update({value: [key]} if value not in output_dict else {value: [*output_dict[value], key]})
    return {k: v[0] if len(v) == 1 else v for k, v in output_dict.items()} if unpack_single_values else output_dict


def ensure_keys_in_dict(input_dict, expected_key_list, default_value=None):
    return {**input_dict, **{key: default_value for key in expected_key_list if key not in input_dict}}


def get_key_by_value(input_dict, value, return_first=True):
    """
    Get the key of a value in a dict. If the value is not in the dict, a ValueError is raised. Returns the first key
    found with the value if return_first is True, otherwise returns a list of all keys with the value. If the value is
    not in the dict, a ValueError is raised.
    Implementation from https://stackoverflow.com/a/8023329

    :param input_dict: A dict
    :type input_dict: dict
    :param value: A value
    :type value: Any
    :param return_first: Whether to return only the first key found with the value
    :type return_first: bool
    :return: The key of the value in the dict
    :rtype: Any
    :raises ValueError: If the value is not in the dict
    """
    if return_first:
        res = next((key for key, val in input_dict.items() if val == value), None)
    else:
        res = [key for key, val in input_dict.items() if val == value]
    if res is None or len(res) == 0:
        raise ValueError("Value not found in dict!")
    return res


def set_dict_reduction_to_mds(set_dict, allow_alternatives=False):
    condensed_mds = condense_sets_to_mds(list(set_dict.values()), allow_alternatives=allow_alternatives)

    # translate the condensed mds to a list of keys from the original set_dict
    mds = []
    for mds_merge in condensed_mds:
        mds.append([get_key_by_value(set_dict, c, return_first=True) for c in mds_merge])
    return mds
