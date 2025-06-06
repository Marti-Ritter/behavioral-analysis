import cv2
from IPython.core.display import Image
from IPython.core.display_functions import display
from tqdm.auto import tqdm


def show_frame_in_notebook(cv2_frame, display_handle=None):
    """
    Displays a single frame in a Jupyter notebook. The frame is displayed as an IPython Image object in the notebook.

    :param cv2_frame: The frame to be displayed
    :type cv2_frame: np.ndarray
    :param display_handle: The handle to the display object. If None, a new display object will be created
    :type display_handle: IPython.display.display
    :rtype: None
    """

    if display_handle is None:
        display_handle = display(None, display_id=True)

    if cv2_frame is not None:
        _, frame = cv2.imencode(".png", cv2_frame)
        frame = Image(data=frame.tobytes())
    else:
        frame = None

    display_handle.update(frame)


def notebook_variable_reset(vars_to_keep=None, vars_to_delete=None, keep_reset_function=True):
    """
    From https://stackoverflow.com/a/49517289.
    Takes an optional list of variables to keep and delete. If vars_to_keep is None, all variables are kept. If
    vars_to_delete is None, no variables are deleted. If keep_reset_function is True, the notebook_variable_reset
    function is kept. Deletion takes precedence over keeping.
    Can only be used in a Jupyter notebook.

    :param vars_to_keep: A list of variables to keep
    :type vars_to_keep: list
    :param vars_to_delete: A list of variables to delete
    :type vars_to_delete: list
    :param keep_reset_function: Whether to keep the notebook_variable_reset function
    :type keep_reset_function: bool
    """
    from IPython import get_ipython

    globals_ = globals()
    if vars_to_keep is None:
        vars_to_keep = globals_.keys()
    if vars_to_delete is None:
        vars_to_delete = []
    vars_to_keep = set(vars_to_keep) - set(vars_to_delete)
    saved_globals = {var: globals_[var] for var in vars_to_keep}
    if keep_reset_function:
        saved_globals['notebook_variable_reset'] = notebook_variable_reset
    del globals_
    get_ipython().magic("reset")
    globals().update(saved_globals)


def show_frame_source_in_jupyter_notebook(frame_source, fps=30, show_progress=True):
    from IPython.display import display
    import time

    inter_frame_interval = 1 / fps

    d = display(None, display_id=True)
    frame_source = tqdm(frame_source) if show_progress else frame_source
    try:
        for _frame_index, frame in frame_source:
            start = time.time()
            show_frame_in_notebook(frame, display_handle=d)
            time.sleep(max(inter_frame_interval - (time.time() - start), 0))
    except KeyboardInterrupt:
        pass
