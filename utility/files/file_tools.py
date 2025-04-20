import json
import mimetypes
import os

from ..general import convert_bytes_to_string


def os_cp(origin_path, target_path):
    """

    :param origin_path:
    :type origin_path: str
    :param target_path:
    :type target_path: str
    """
    os.system(f"copy \"{origin_path}\" \"{target_path}\"")


def list_files(directory_path):
    """

    :param directory_path:
    :type directory_path: str
    :return:
    :rtype: list
    """

    file_list = []
    for base_path, directories, files in os.walk(directory_path):
        if directory_path == base_path:
            file_list.extend(files)
        else:
            file_list.extend([os.path.join(os.path.relpath(base_path, directory_path), file) for file in files])
    return file_list


def read_json(file_path):
    """

    :param file_path:
    :type file_path: str
    :return:
    :rtype: dict
    """
    with open(file_path) as json_file:
        return json.load(json_file)


def write_json(file_path, json_dict):
    """

    :param file_path:
    :type file_path: str
    :param json_dict:
    :type json_dict: dict
    """
    with open(file_path, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)


def read_txt(file_path):
    """

    :param file_path:
    :type file_path: str
    :return:
    :rtype: str
    """
    with open(file_path) as txt_file:
        return txt_file.read()


def write_txt(file_path, txt_string):
    """

    :param file_path:
    :type file_path: str
    :param txt_string:
    :type txt_string: str
    """
    with open(file_path, 'w') as txt_file:
        txt_file.write(txt_string)


def ensure_directory(directory_path):
    """

    :param directory_path:
    :type directory_path: str
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def find_files(root_directory, extension="", case_sensitive=False, max_depth=None, **kwargs):
    """
    Find files in a directory with a given extension. If no extension is given, all files are returned. If max_depth is
    given, the search will not go deeper than that. If case_sensitive is False, the extension is matched case
    insensitively.

    :param root_directory: The root directory to search in.
    :type root_directory: str
    :param extension: The extension to search for.
    :type extension: str
    :param case_sensitive: Whether the extension should be matched case sensitively.
    :type case_sensitive: bool
    :param max_depth: The maximum depth to search in. If None, the search will not be limited.
    :type max_depth: int
    :param kwargs: Additional keyword arguments to pass to os.walk.
    :type kwargs: dict
    :return: A list of found files.
    :rtype: list[str]
    """

    found_files = []
    for root, dirs, files in os.walk(root_directory, **kwargs):
        if not case_sensitive:
            found_files.extend([os.path.join(root, f) for f in files if f.lower().endswith(extension.lower())])
        else:
            found_files.extend([os.path.join(root, f) for f in files if f.endswith(extension)])
        # https://stackoverflow.com/a/42720948
        if (max_depth is not None) and (root.count(os.sep) - root_directory.count(os.sep) == max_depth - 1):
            del dirs[:]
    return found_files


def remove_empty_directories(root_dir):
    # https://stackoverflow.com/a/47093793
    directories = list(os.walk(root_dir))[1:]
    for directory in directories:
        # folder example: ('root_dir/3', [], ['file'])
        if not directory[2]:
            os.rmdir(directory[0])


def filter_files_for_mimetype(input_files, mime_type_start):
    relevant_files = []
    for f in input_files:
        mimetype = mimetypes.guess_type(f)[0]
        if (mimetype is not None) and (mimetype.startswith(mime_type_start)):
            relevant_files.append(f)
    return relevant_files


def get_mimetype_extensions(requested_mime_type):
    """
    Return a list of file extensions that match the given MIME type.

    :param requested_mime_type: The MIME type to search for.
    :type requested_mime_type: str
    :return: A list of file extensions that match the given MIME type.
    :rtype: list[str]
    """
    if not mimetypes.inited:
        mimetypes.init()

    extensions = []
    for extension, mime_type in mimetypes.types_map.items():
        if not "/" in requested_mime_type:
            mime_type = mime_type.split('/')[0]
        if mime_type == requested_mime_type and extension not in extensions:
            extensions.append(extension)

    return extensions


def file_size(file_path):
    """
    this function will return the file size
    Adapted from https://stackoverflow.com/a/39988702
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes_to_string(file_info.st_size)


def try_to_remove_empty_dir(empty_dir_path):
    """
    Try to remove a directory. This will fail if the directory is not empty or does not exist.
    Copied from: https://stackoverflow.com/a/23488980

    :param path: The path to the directory to remove.
    :type path: str
    """
    try:
        os.rmdir(empty_dir_path)
    except OSError:
        pass


def remove_empty_sub_dirs(root_dir):
    """
    Recursively remove empty subdirectories in a given root directory.
    Copied from: https://stackoverflow.com/a/23488980

    :param path: The path to the root directory.
    :type path: str
    """
    for root, dir_names, filenames in os.walk(root_dir, topdown=False):
        for dir_name in dir_names:
            try_to_remove_empty_dir(os.path.realpath(os.path.join(root, dir_name)))


def scandir_walk(root_dir_path):
    """
    A generator that recursively scans a directory and yields all files and directories in all subdirectories using
    os.scandir(). Mimics the behavior of os.walk().

    :param root_dir_path: The path to the root directory to scan.
    :type root_dir_path: str
    :return: A generator that yields all files and directories in all subdirectories.
    :rtype: Generator[os.DirEntry]
    """

    for entry in os.scandir(root_dir_path):
        if entry.is_dir():
            yield from scandir_walk(entry.path)
        else:
            yield entry