"""
A pandas oriented extension of the caching module from ..utility/caching.py.
This extension allows more detailed caching in h5 files, which can be useful for caching multiple pandas objects in a
single file.
"""

import pandas as pd

from ..utility.caching import cache_result_factory, ObjectCache


class H5Cache(ObjectCache):
    cache_extension = ".h5"

    def __init__(self, cache_key="pd_cache", **kwargs):
        """
        Initialize the H5 cache object. Takes an additional cache_key parameter, which is used as the key for the cache
        file. This ensures that load and store operations are performed on the same location in the cache file.

        :param cache_key: The key to use for the h5 file
        :type cache_key: str
        :param kwargs: Additional keyword arguments for the ObjectCache class
        :type kwargs: dict
        """
        super().__init__(**kwargs)
        self.store_kwargs["key"] = cache_key
        self.load_kwargs["key"] = cache_key

    @property
    def exists(self):
        file_exists = super().exists
        if not file_exists:
            return False

        else:
            cache_store = pd.HDFStore(self.cache_path, mode="r")
            exists = self.store_kwargs["key"] in cache_store
            cache_store.close()
            return exists

    def remove(self):
        if self.exists:
            cache_store = pd.HDFStore(self.cache_path, mode="a")
            if self.store_kwargs["key"] in cache_store:
                cache_store.remove(self.store_kwargs["key"])

                cache_key_count = len(cache_store.keys())
                cache_store.close()

                # this check is only run if the cache key actually existed in the file
                # This is not checked otherwise to avoid removing empty files that were created through other means
                if cache_key_count == 0:
                    # If the cache file is empty, remove it
                    super().remove()

    def load(self):
        return pd.read_hdf(self.cache_path, **self.load_kwargs)

    def store(self, obj):
        obj.to_hdf(self.cache_path, **self.store_kwargs)


pandas_type_cache_map = {
    pd.DataFrame: H5Cache,
    pd.Series: H5Cache,
}

# Get the cache_result function with the local type_cache_map added
pandas_cache_result = cache_result_factory(pandas_type_cache_map)
