import inspect
import math
import sys
from functools import wraps


def convert_to_func(input_object):
    """
    Convert an object to a function. If the object is a dict, return a function that returns the value of the dict
    corresponding to the passed key. If the object is callable, return the object. Otherwise, return a function that
    returns the passed object. Useful for cases where a function, dict, or other object may be passed to a function, and
    the function needs to be able to handle all cases. E.g. a plot function needs to figure out the color to assign to
    a value, and the assignment can be either discrete (dict) or continuous (function) or static (object, str here).

    :param input_object: An object
    :type input_object: Any
    :return: A function
    :rtype: function
    """
    if isinstance(input_object, dict):
        return input_object.get
    elif callable(input_object):
        return input_object
    else:
        return lambda x: input_object


def get_function_dict_of_module(module_name):
    """
    Taken from https://stackoverflow.com/a/63413129.

    :param module_name:
    :type module_name: str
    :return:
    :rtype: dict
    """
    return {name: obj for name, obj in inspect.getmembers(sys.modules[module_name])
            if (inspect.isfunction(obj) and obj.__module__ == module_name)}


def make_string_alphanum(input_string, permitted_characters="._- ", invalid_replacement=None):
    """
    Drop all characters from a string that are not alphanumeric or in the permitted_characters list. If
    invalid_replacement is not None, replace invalid characters with the specified replacement. Taken from from
    https://stackoverflow.com/a/295152.

    :param input_string:
    :type input_string:
    :param permitted_characters:
    :type permitted_characters:
    :param invalid_replacement:
    :type invalid_replacement:
    :return:
    :rtype:
    """
    return "".join(x if (x.isalnum() or x in permitted_characters) else (
        invalid_replacement if invalid_replacement is not None else "") for x in input_string)


def replace_signature(signature_donor):
    """
    It's far too complicated to merge signatures: https://github.com/Kwpolska/merge_args/blob/master/merge_args.py
    https://stackoverflow.com/a/60832711

    :param signature_donor:
    :type signature_donor:
    :return:
    :rtype:
    """
    def func_decorator(wrapped_func):
        @wraps(signature_donor)
        def wrapper(*args, **kwargs):
            return wrapped_func(*args, **kwargs)

        return wrapper

    return func_decorator


def append_docstring(signature_donor, doc_join_string="\nThis function extends the following functionality:\n"):
    """
    A simpler approach to just append some notice that this function works on the basis of some other function

    :param signature_donor:
    :type signature_donor:
    :param doc_join_string:
    :type doc_join_string:
    :return:
    :rtype:
    """
    def func_decorator(wrapped_func):
        def wrapper(*args, **kwargs):
            return wrapped_func(*args, **kwargs)

        wrapper.__doc__ = doc_join_string.join(
            [i for i in [wrapped_func.__doc__, signature_donor.__doc__] if i is not None]) or None
        return wrapper

    return func_decorator


def prepend_append_calls_to_function(input_func, run_before=tuple(), run_after=tuple()):
    @wraps(input_func)
    def wrapper(*args, **kwargs):
        for func_before in run_before:
            func_before()
        return_value = input_func(*args, **kwargs)
        for func_after in run_after:
            func_after()
        return return_value

    return wrapper


def prepend_append_calls_decorator(run_before=tuple(), run_after=tuple()):
    def func_decorator(wrapped_func):
        def wrapper(*args, **kwargs):
            for func_before in run_before:
                func_before()
            return_value = wrapped_func(*args, **kwargs)
            for func_after in run_after:
                func_after()
            return return_value

        return wrapper

    return func_decorator


def prepend_append_calls_to_object(input_object,
                                   methods_to_modify=None, methods_to_exclude=None,
                                   calls_before=tuple(), calls_after=tuple()):
    if methods_to_modify is None:
        methods_to_modify = [attribute for attribute in dir(input_object) if
                             callable(getattr(input_object, attribute)) and not attribute.startswith('_')]
    if methods_to_exclude is None:
        methods_to_exclude = []
    methods_to_modify, methods_to_exclude = set(methods_to_modify), set(methods_to_exclude)
    methods_to_modify -= methods_to_exclude
    for method in methods_to_modify:
        setattr(input_object, method,
                prepend_append_calls_to_function(getattr(input_object, method), calls_before, calls_after))


def get_public_attributes(input_object):
    """
    Returns a dict of all public attributes of an object, i.e. all attributes that are not private (start with "_") and
    are not callable.

    :param input_object: object to get public attributes from
    :type input_object: object
    :return: dict of public attributes
    :rtype: dict
    """
    public_attributes = {}
    for attr in dir(input_object):
        attr_value = getattr(input_object, attr)
        if not attr.startswith("_") and not callable(attr_value):
            public_attributes[attr] = attr_value
    return public_attributes


def get_size(obj, seen=None):
    """
    Recursively finds the size of an arbitrary Python object (e.g., list, tuple, set, dict, etc., anything really).
    The size is the sum of the size of all the objects in the object, including the object itself, and is the size the
    object takes up in memory. The size is returned in bytes.
    From https://stackoverflow.com/a/30316760
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def convert_bytes_to_string(size_bytes):
    """
    A function to convert bytes to a human-readable format. It automatically chooses the best unit to use.
    Adapted from https://stackoverflow.com/a/14822210.

    :param size_bytes: The size in bytes
    :type size_bytes: int
    :return: The size in a human-readable format
    :rtype: str
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def get_locals_and_globals():
    """
    Returns the local and global variables in the calling function. Useful for debugging and introspection.

    :return: A tuple containing the local and global variables
    :rtype: tuple
    """
    return inspect.currentframe().f_back.f_locals, inspect.currentframe().f_back.f_globals


def get_locals_size():
    """
    Returns the size of the local variables in the calling functions, ordered by size. Useful for debugging and
    introspection.

    :return: A dict containing the size of the local variables
    :rtype: dict
    """
    local_vars = inspect.currentframe().f_back.f_locals
    return {key: get_size(value) for key, value in local_vars.items()}


def cm2inch(*input_cm):
    """
    Converts any given number of centimeter values to inches.
    Taken from https://stackoverflow.com/a/22787457

    :param input_cm: A number of centimeter values
    :type input_cm: float
    :return: The same values in inches
    :rtype: tuple of float
    """
    inch = 2.54
    return tuple(i/inch for i in input_cm)


def inch2cm(*input_inch):
    """
    Converts any given number of inch values to centimeters.
    Taken from https://stackoverflow.com/a/22787457

    :param input_inch: A number of inch values
    :type input_inch: float
    :return: The same values in centimeters
    :rtype: tuple of float
    """
    inch = 2.54
    return tuple(i*inch for i in input_inch)


def output_formatter(output_values, output_flags, unpack_single=True):
    """
    Formats the output of a function according to the passed flags. The flags are a list of booleans, where each boolean
    indicates whether the corresponding value in the output_values should be returned. If the flag is False, the value
    is not returned. If the flag is True, the value is returned. If unpack_single is True, and only one value is
    returned, that value is unpacked from the list.
    The output_values and output_flags must have the same length.
    :param output_values: Values to return
    :type output_values: list or tuple
    :param output_flags: Flags indicating whether to return the corresponding value in output_values
    :type output_flags: list of bool or tuple of bool
    :param unpack_single: Whether to unpack the values if only one value is returned
    :type unpack_single: bool
    :return: The values in output_values that correspond to True flags
    :rtype: list
    """
    if len(output_values) != len(output_flags):
        raise ValueError("output_values and output_flags must have the same length")
    return_values = [output_values[i] for i, flag in enumerate(output_flags) if flag]
    if unpack_single and len(return_values) == 1:
        return_values = return_values[0]
    return return_values


def standardize_padding(padding):
    """
    Standardizes a given tuple of pad widths to adhere to the standard (top, bottom, left, right) format.
    Valid input formats are:
        (top, bottom, left, right)  -->  standard format
        (top & bottom, left & right)  -->  top = bottom, left = right
        (top & bottom & left & right)  -->  top = bottom = left = right
        top & bottom & left & right  -->  top = bottom = left = right (single value)
    :param padding: A tuple of pad widths or a single pad width.
    :type padding: tuple of (int or float) or int or float
    :return: A tuple of pad widths in the standard format.
    :rtype: tuple of (int or float)
    """

    assert isinstance(padding, (list, tuple, int, float)), "Padding must be a list, tuple, int or float"
    if isinstance(padding, (int, float)):
        padding = (padding, padding, padding, padding)
    if len(padding) == 4:
        pad_top, pad_bottom, pad_left, pad_right = padding
    elif len(padding) == 2:
        pad_top = pad_bottom = padding[0]
        pad_left = pad_right = padding[1]
    elif len(padding) == 1:
        pad_top = pad_bottom = pad_left = pad_right = padding[0]
    else:
        raise ValueError("Padding must be a list of length 1, 2, or 4.")
    return pad_top, pad_bottom, pad_left, pad_right


def round_to_nearest_multiple(input_value, reference_value):
    """
    Rounds the input value to the nearest multiple of the reference_value.
    Adapted from https://stackoverflow.com/a/7859208

    :param input_value: The value to round
    :type input_value: float
    :param reference_value: The reference value
    :type reference_value: float
    """
    return round(input_value / reference_value) * reference_value
