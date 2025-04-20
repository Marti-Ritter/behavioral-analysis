import contextlib
import multiprocessing

import joblib
import pandas as pd


def return_with_label(label, func, value, *args):
    return label, func(value, *args)


def parallelize_on_all_dict_entries(input_dict, func, *args, processes=4, tqdm_kwargs_list=None):
    tqdm_kwargs_list = [None for _ in range(5)] if tqdm_kwargs_list is None else tqdm_kwargs_list
    pool_args = [(key, func, value, *args) for key, value in input_dict.items()]
    if tqdm_kwargs_list is not None:
        pool_args = [(*pool, tqdm_kwargs) for pool, tqdm_kwargs in zip(pool_args, tqdm_kwargs_list)]
    with multiprocessing.Pool(processes=processes) as p:
        results = p.starmap(return_with_label, pool_args)
    return {key: value for key, value in results}


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    Taken from https://stackoverflow.com/a/58936697
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def parallel_pd_apply(grouped_df, func, processes=4):
    with multiprocessing.Pool(processes) as p:
        ret_list = p.map(func, [group for name, group in grouped_df])
    return pd.concat(ret_list)
