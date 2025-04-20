import os

import h5py
import numpy as np
from ..utility.builtin_classes.objects import load_object, save_object


def apply_model_iterations(project_dir, model_name, test_data_path, test_data_format='sleap',
                           iteration_filter=None, save_results=True, overwrite=False):
    """
    Apply multiple saved iterations of a model to a test dataset. This function can be used to check whether the model
    stabilizes or improves over time, or to compare the performance of different iterations.

    :param project_dir: The root directory of the project.
    :type project_dir: str
    :param model_name: The name of the model to apply.
    :type model_name: str
    :param test_data_path: The path to the test dataset. See `keypoint_moseq.load_keypoints` for more information.
    :type test_data_path: str
    :param test_data_format: The format of the test dataset. See `keypoint_moseq.load_keypoints` for more information.
    :type test_data_format: str
    :param iteration_filter: A filter to select specific iterations to apply. If None, all saved iterations will be
    applied. Can be a function that takes an iteration number as input and returns a boolean, or a list or tuple of
    iteration numbers to apply.
    :type iteration_filter: Callable, list, or tuple
    :param save_results: Whether to save the iteration results to a file. If True, the results will be saved to
    `project_dir/model_name/IterationResults.pkl`.
    :type save_results: bool
    :param overwrite: Whether to overwrite the saved iteration results if they already exist.
    :type overwrite: bool
    :return: A dictionary containing the results of applying each iteration of the model to the test dataset.
    :rtype: dict
    """
    output_path = os.path.join(project_dir, model_name, "IterationResults.pkl")
    if os.path.exists(output_path) and not overwrite:
        print(f"Loading iteration results from {output_path}...")
        return load_object(output_path)

    checkpoint_path = os.path.join(project_dir, model_name, "checkpoint.h5")
    with h5py.File(checkpoint_path, "r") as f:
        saved_iterations = np.sort([int(i) for i in f["model_snapshots"]])

    if iteration_filter is not None:
        if callable(iteration_filter):
            filtered_iterations = [it for it in saved_iterations if iteration_filter(it)]
        elif isinstance(iteration_filter, (list, tuple)):
            filtered_iterations = [it for it in saved_iterations if it in iteration_filter]
        else:
            raise ValueError("iteration_filter must be a callable, list, or tuple.")
    else:
        filtered_iterations = saved_iterations

    import keypoint_moseq as kpms
    config = lambda: kpms.load_config(project_dir)

    coordinates, confidences, bodyparts = kpms.load_keypoints(test_data_path, test_data_format)
    data, metadata = kpms.format_data(coordinates, confidences, **config())

    iteration_results = {}
    for iteration in filtered_iterations:
        print(f"Applying model iteration {iteration}...")
        model = kpms.load_checkpoint(project_dir, model_name, iteration=iteration)[0]
        iteration_results[iteration] = kpms.apply_model(model, data, metadata, project_dir, model_name, **config(),
                                                        save_results=False)

    if save_results:
        save_object(iteration_results, output_path)
        print(f"Iteration results saved to {output_path}.")
    return iteration_results
