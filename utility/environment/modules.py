import importlib
import os
import pkgutil
from contextlib import contextmanager


def get_submodules(root_module=None, max_depth=1, print_errors=False, anchor_module=None):
    """
    Return a list of submodules to the specified root module up to a maximum depth.

    :param root_module: the name of the root module to retrieve submodules from. If None, use the current module.
    :type root_module: str or None
    :param max_depth: the maximum depth to search for submodules. Default is 1, which retrieves only immediate
    submodules.
    :type max_depth: int
    :param print_errors: whether to print any errors that occur while retrieving submodules. Default is False.
    :type print_errors: bool
    :param anchor_module: The name of the module to use as an anchor for the root_module.
    If None, only absolute root module names are permitted.
    :type anchor_module: str or None
    :return: a list of submodule names, as strings.
    :rtype: list
    """

    if root_module is None:
        root_module = __name__
    try:
        package_path = importlib.import_module(root_module, package=anchor_module).__path__
    except (AttributeError, ModuleNotFoundError) as e:
        if print_errors:
            print(f"Error while retrieving submodules for {root_module}: {str(e)}")
        return []

    submodules = []
    for loader, name, is_pkg in pkgutil.iter_modules(path=package_path):
        full_name = root_module + '.' + name
        if is_pkg:
            submodules.append(full_name)
            if max_depth > 1:
                submodules.extend(get_submodules(full_name, max_depth=max_depth - 1, print_errors=print_errors))

    return submodules


def get_package_path(package_name, resource_name=None, anchor_module=None):
    """
    Returns the path to either the package or a resource within the package.

    :param package_name: The name of the package.
    :type package_name: str
    :param resource_name: The name of the resource to retrieve.
    :type resource_name: str or None
    :param anchor_module: The name of the module to use as an anchor for the package.
    If None, only absolute package names are permitted.
    :type anchor_module: str or None
    :return: The path to either the package or the resource within the package.
    :rtype: str
    """

    package = importlib.import_module(package_name, package=anchor_module)
    package_path = os.path.dirname(package.__file__)

    if resource_name is None:
        return package_path
    else:
        resource_path = os.path.join(package_path, resource_name)
        return resource_path


def list_python_packages(directory_path, only_executable=False, max_depth=None):
    """
    Lists all available Python packages in a directory.

    :param directory_path: The directory path that will be searched for python packages
    :type directory_path: str
    :param only_executable: If True, only executable packages will be listed.
    :type only_executable: bool
    :param max_depth: The maximum depth of directories to search, defaults to None for unlimited depth
    :type max_depth: int or None
    :return: A list of packages in directory_path
    :rtype: list
    """

    packages = []

    for root, dirs, files in os.walk(directory_path):
        if max_depth is not None:
            rel_root = os.path.relpath(root, directory_path)
            depth = len(rel_root.split(os.sep))
            if depth > max_depth:
                del dirs[:]  # stop descending further

        relative_root = os.path.relpath(root, directory_path)
        relative_root = "" if relative_root == "." else relative_root
        for d in dirs:
            if os.path.exists(os.path.join(root, d, "__init__.py")):
                if only_executable:
                    if os.path.exists(os.path.join(root, d, "__main__.py")):
                        packages.append(os.path.join(relative_root, d))
                else:
                    packages.append(os.path.join(relative_root, d))

    return packages


@contextmanager
def optional_dependencies(error: str = "ignore"):
    """
    Context manager to handle optional dependencies. From https://stackoverflow.com/a/73838937.
    Usage:

    with optional_dependencies("warn"):
        import optional_dependency
        optional_dependency.some_function()

    :param error: The error handling method. Can be "raise", "warn", or "ignore".
    :type error: str
    :return: None
    """
    assert error in {"raise", "warn", "ignore"}
    try:
        yield None
    except ImportError as e:
        if error == "raise":
            raise e
        if error == "warn":
            msg = f'Missing optional dependency "{e.name}". Use pip or conda to install.'
            print(f'Warning: {msg}')
