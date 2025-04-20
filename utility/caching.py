"""
A module that provides caching functionality for functions returning (annotated) basic types, such as strings, lists,
dictionaries, and objects. The caching is done through the cache_result decorator, which caches the result of a
function to a file.

This module can also be expanded in other modules, such as adding more cache types, for example h5 files, and adding
their associated cache classes to the type_cache_map dictionary.
"""

import json
import os
import pickle
import warnings
from functools import wraps
from inspect import signature


class ObjectCache:
    """
    A class that caches an object to a file using the pickle module. The cache file is saved to the path specified in
    the cache_path attribute. The object can be loaded from the cache file using the load method, and the object can be
    stored to the cache file using the store method.
    This is the base class for the cache objects.
    """
    cache_extension = ".pkl"
    default_load_kwargs = {}
    default_store_kwargs = {}

    def __init__(self, cache_path, cache_load_kwargs=None, cache_store_kwargs=None, **kwargs):
        self.cache_path = self.adjust_cache_path_extension(cache_path)

        if cache_load_kwargs is None:
            cache_load_kwargs = {}
        self.load_kwargs = {**self.default_load_kwargs, **cache_load_kwargs}

        if cache_store_kwargs is None:
            cache_store_kwargs = {}
        self.store_kwargs = {**self.default_store_kwargs, **cache_store_kwargs}

    def adjust_cache_path_extension(self, cache_path):
        """
        Adjust the cache path extension to match the cache extension of the cache object. If the cache path extension
        does not match the cache extension of the cache object, a warning is raised.

        :param cache_path: The path to adjust the extension of
        :type cache_path: str
        :return: The adjusted cache path
        :rtype: str
        """
        cache_file, cache_ext = os.path.splitext(cache_path)
        if cache_ext != self.cache_extension:
            warnings.warn(f"Cache path extension does not match the cache extension of the cache object. "
                          f"Original extension: {cache_ext}, cache extension: {self.cache_extension}.")
            return cache_file + self.cache_extension
        return cache_path

    @property
    def exists(self):
        """
        Check if the cache file exists.

        :return: A boolean indicating whether the cache file exists
        :rtype: bool
        """
        return os.path.exists(self.cache_path)

    def remove(self):
        """
        Remove the cache file.

        :return: None
        """
        if self.exists:
            os.remove(self.cache_path)

    def load(self):
        """
        Load the object from the cache file.

        :return: The object loaded from the cache file
        :rtype: Any
        """
        with open(self.cache_path, "rb") as f:
            return pickle.load(f, **self.load_kwargs)

    def store(self, obj):
        """
        Store the object to the cache file.

        :param obj: The object to store to the cache file
        :type obj: Any
        :return: None
        """
        with open(self.cache_path, "wb") as f:
            pickle.dump(obj, f, **self.store_kwargs)


class StringCache(ObjectCache):
    cache_extension = ".txt"

    def load(self):
        with open(self.cache_path) as f:
            return f.read(**self.load_kwargs)

    def store(self, string):
        with open(self.cache_path, "w") as f:
            # This function takes no further arguments, so we can ignore the store_kwargs
            f.write(string)


class JsonCache(ObjectCache):
    """
    A class that caches a JSON serializable object to a file using the json module. The cache file is saved to the path
    specified in the cache_path attribute. The object can be loaded from the cache file using the load method, and the
    object can be stored to the cache file using the store method.
    Extends the ObjectCache class.
    """
    cache_extension = ".json"
    default_store_kwargs = {"indent": 4}

    def load(self):
        with open(self.cache_path) as f:
            return json.load(f, **self.load_kwargs)

    def store(self, obj):
        with open(self.cache_path, "w") as f:
            json.dump(obj, f, **self.store_kwargs)


default_type_cache_map = {
    str: StringCache,  # should always work
    dict: JsonCache,  # can be attempted to be stored as a JSON file, if it is JSON serializable
    list: JsonCache,  # same
    tuple: JsonCache,  # same again, if any of this fails it will fall back to the ObjectCache and be pickled
    object: ObjectCache,  # the default case, will always work
}


def cache_result_factory(*additional_type_cache_map):
    """
    A factory function that returns a cache_result decorator with the given additional type cache maps. The default type
    cache map is merged with the additional type cache maps.

    :param additional_type_cache_map: Additional type cache maps to merge with the default type cache map
    :type additional_type_cache_map: dict
    :return: A cache_result decorator with the merged type cache map
    :rtype: Callable
    """
    type_cache_map = {**default_type_cache_map}
    for additional_map in additional_type_cache_map:
        type_cache_map = {**type_cache_map, **additional_map}

    def _cache_result(func):
        """
        A generalized cache function that caches the result of a function to a file. The file type is determined by the
        return type of the function. The cache file is saved to the path specified in the cache_path keyword argument if
        it does not exist yet. If the cache_path is None, the result will not be cached, and it will be recomputed every
        time the function is called. If the store attempt fails, then the result will be cached as a pickle file.
        :param func: A function to cache the result of
        :type func: Callable
        :return: A function that caches the result of the passed function
        :rtype: Callable
        """

        @wraps(func)
        def call_func_check_result_cache(*args, cache_path=None, cache_init_kwargs=None,
                                         cache_load_kwargs=None, cache_store_kwargs=None,
                                         cache_overwrite=False, **kwargs):
            """
            A function that calls the passed function and caches the result to a file if a cache_path is given.
            :param args: Positional arguments to pass to the function
            :type args: Any
            :param cache_path: The path to cache the result to
            :type cache_path: str
            :param cache_init_kwargs: Additional keyword arguments to pass to the cache object during initialization
            :type cache_init_kwargs: dict
            :param cache_load_kwargs: Additional keyword arguments to pass to the load method of the cache object
            :type cache_load_kwargs: dict
            :param cache_store_kwargs: Additional keyword arguments to pass to the store method of the cache object
            :type cache_store_kwargs: dict
            :param cache_overwrite: Whether to overwrite the cache file if it exists
            :type cache_overwrite: bool
            :param kwargs: Keyword arguments to pass to the function
            :type kwargs: Any
            :return: The result of the function
            :rtype: Any
            """
            if cache_path is not None:
                # attempt to get the result type from the function signature
                sig = signature(func)
                result_type = sig.return_annotation
                if result_type == sig.empty:
                    result_type = object
                    warnings.warn(
                        "No return type annotation found for function {}. Defaulting to object.".format(func.__name__))

                # If a cache path is given, check if the cache file exists
                if cache_init_kwargs is None:
                    cache_init_kwargs = {}
                cache_obj = type_cache_map.get(result_type, ObjectCache)(cache_path=cache_path,
                                                                         cache_load_kwargs=cache_load_kwargs,
                                                                         cache_store_kwargs=cache_store_kwargs,
                                                                         **cache_init_kwargs)

                if not cache_overwrite and cache_obj.exists:
                    return cache_obj.load()

                # If the cache file does not exist, call the function and store the result
                result = func(*args, **kwargs)

                # Attempt to store the result to the cache file and fall back to pickling if it fails
                try:
                    cache_obj.store(result)
                except Exception as e:
                    # check if a file was created as the error occurred
                    if cache_obj.exists:
                        # remove the cache if it was created
                        cache_obj.remove()

                    cache_obj = ObjectCache(cache_path)
                    warnings.warn(f"An error occurred while storing the result to the cache file:\n{e}\n "
                                  f"Falling back to pickling the result at {cache_obj.cache_path}.")
                    cache_obj.store(result)

            else:
                # If no cache path is given, just return the result of the function
                result = func(*args, **kwargs)

            return result

        return call_func_check_result_cache

    return _cache_result


# The default cache_result decorator
cache_result = cache_result_factory()
