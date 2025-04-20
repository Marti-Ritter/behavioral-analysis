import inspect

from .dicts import ensure_keys_in_dict, invert_flat_dict
from .iterables import split_chunks


def get_func_kwargs(func):
    params = get_function_param_types(func)
    params = ensure_keys_in_dict(params, ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"),
                                 default_value=[])
    return (*params["POSITIONAL_OR_KEYWORD"], *params["KEYWORD_ONLY"])


def get_function_param_types(func):
    sig = inspect.signature(func)
    parameter_kind = {param_name: param_obj.kind.name for param_name, param_obj in sig.parameters.items()}
    return invert_flat_dict(parameter_kind, unpack_list_values=False)


def get_function_defaults(func):
    # https://stackoverflow.com/a/12627202
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def is_bound(method):
    """
    Checks whether a passed callable is bound to an instance of a class.
    Taken from https://stackoverflow.com/a/18955425.

    :param method: A callable.
    :type method: callable
    :return: A boolean indicating whether the passed callable is a bound method.
    :rtype: bool
    """
    return hasattr(method, '__self__')


def get_filtered_parameters(func, *args, filter_args=True, filter_kwargs=True, **kwargs):
    """
    Can perform a comparison between passed kwargs and the valid kwargs of a passed callable. Valid kwargs are detected
    through get_function_param_types, using the keys "POSITIONAL_OR_KEYWORD" and "KEYWORD_ONLY". If **kwargs are allowed
    in the passed callable, all passed kwargs are valid (as any "invalid" kwargs will be caught in **kwargs).
    This functionality can be enabled or disabled with filter_kwargs.
    Any passed args will be first split according to the "POSITIONAL_ONLY" and "POSITIONAL_OR_KEYWORD" args returned by
    get_function_param_types. If the passed "POSITIONAL_OR_KEYWORD" args cover an also passed kwarg, the kwarg will
    prevail. Finally the "POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD", and "VAR_POSITIONAL" (if *args are allowed) args
    will be returned as filtered args, and a dictionary of filtered kwargs not covered by the filtered args will be
    returned.
    Both filters can be enabled or disabled. If filter_kwargs is False, the passed kwargs will be returned and used
    during the filter of the args.
    Adapted from https://stackoverflow.com/a/40363565.

    :param func: A callable
    :type func: callable
    :param args: Positional parameters
    :type args: any
    :param filter_args: Whether to filter the passed args
    :type filter_args: bool
    :param filter_kwargs: Whether to filter the passed kwargs
    :type filter_kwargs: bool
    :param kwargs: Keyword parameters
    :type kwargs: any
    :return: filtered args in a tuple and filtered kwargs in a dict
    :rtype: (tuple, dict)
    """
    params = get_function_param_types(func)
    params = ensure_keys_in_dict(params, ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"), default_value=[])

    func_kwargs = (*params["POSITIONAL_OR_KEYWORD"], *params["KEYWORD_ONLY"])
    filtered_kwargs = {kw: arg for kw, arg in kwargs.items() if
                       kw in func_kwargs} if filter_kwargs and "VAR_KEYWORD" not in params else kwargs

    if filter_args:
        pos_only_args, pos_or_kw_args, var_pos_args = split_chunks(args, [len(params["POSITIONAL_ONLY"]),
                                                                          len(params["POSITIONAL_OR_KEYWORD"])])
        pos_or_kw_mapping = dict(zip(params["POSITIONAL_OR_KEYWORD"], pos_or_kw_args))
        pos_or_kw_args = [val_map if key not in filtered_kwargs else filtered_kwargs.pop(key) for key, val_map in
                          pos_or_kw_mapping.items()]
        filtered_args = [*pos_only_args, *pos_or_kw_args, *(var_pos_args if "VAR_POSITIONAL" in params else [])]
    else:
        filtered_args = args

    return filtered_args, filtered_kwargs


def apply_filtered_parameters(func, *args, filter_args=True, filter_kwargs=True, **kwargs):
    """
    Applies a filtered selection of passed positional and keyword parameters to a function, based on the filter
    implemented in get_filtered_parameters. Filters for positional and keyword parameters can each be enabled or
    disabled. See documentation for get_filtered_parameters.

    :param func: A callable
    :type func: callable
    :param args: Positional parameters
    :type args: any
    :param filter_args: Whether to filter the passed args
    :type filter_args: bool
    :param filter_kwargs: Whether to filter the passed kwargs
    :type filter_kwargs: bool
    :param kwargs: Keyword parameters
    :type kwargs: any
    :return: Return value of callable(*filtered_args, **filtered_kwargs)
    :rtype: any
    """
    filtered_args, filtered_kwargs = get_filtered_parameters(func, *args,
                                                             filter_args=filter_args, filter_kwargs=filter_kwargs,
                                                             **kwargs)
    return func(*filtered_args, **filtered_kwargs)
