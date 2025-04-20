import datetime
import json
import os
from pathlib import Path

from ..files.json_tools import find_valid_json

from .powershell import (get_powershell_set_working_directory_string, add_transcript_to_powershell_command,
                         run_command_in_powershell)

anaconda_config = {"conda_hook_path": r"C:\ProgramData\Anaconda3\shell\condabin\conda-hook.ps1"}


def get_powershell_anaconda_init_command(env_name=None):
    """
    Gets the PowerShell command to initialize an Anaconda environment.

    :param env_name: The name of the Anaconda environment to initialize. If None, then the base environment is used.
    :type env_name: str or None
    :return: The PowerShell command to initialize an Anaconda environment.
    :rtype: str
    """
    if env_name is None:
        env_name = "base"
    return f"{anaconda_config['conda_hook_path']}; conda activate {env_name}"


def get_powershell_python_run_command(path_or_command_string, flags=None, arg_string=None):
    """
    Gets the PowerShell command to run a Python script or module.

    :param path_or_command_string: The path to the Python script or module to run, or a command string.
    :type path_or_command_string: str
    :param flags: A string containing the flags to pass to the Python command (e.g., '-u', '-m').
    :type flags: str or None
    :param arg_string: A string containing the command line arguments for the script or module.
    :type arg_string: str or None
    :return: The PowerShell command to run a Python script or module.
    :rtype: str
    """
    if flags is None:
        flags = ""

    python_command = f"python {flags} '{path_or_command_string}'"

    if arg_string is not None:
        python_command += f" {arg_string}"

    return python_command


def run_python_in_anaconda_env(env_name, script_path, flags=None, arg_string=None, cwd=None, keep_open=True,
                               transcript=True, transcript_name=None, blocking=True, return_result=True, show=True):
    """
    Runs a Python script or module in a specific Anaconda environment using PowerShell.

    :param env_name: The name of the Anaconda environment to use.
    :type env_name: str
    :param script_path: The path to the Python script or module to run.
    :type script_path: str
    :param flags: A string containing the flags to pass to the Python command (e.g., '-u', '-m').
    :type flags: str or None
    :param arg_string: A string containing the configuration for the script or module.
    :type arg_string: str or None
    :param cwd: The current working directory to use for running the script or module.
    :type cwd: str or None
    :param keep_open: Whether to keep the PowerShell window open the script or module has run.
    :type keep_open: bool
    :param transcript: Whether to save the output of the script or module to a transcript file.
    :type transcript: bool
    :param transcript_name: The name to use for the transcript file.
    If None, then the name of the script or module is used.
    :type transcript_name: str or None
    :param blocking: Whether to wait for the script or module to finish before returning.
    :type blocking: bool
    :param return_result: Whether to return the result of the script or module.
    :type return_result: bool
    :param show: Whether to show the powershell window. If hidden, then the process has to close after finishing.
    :type show: bool
    :return: None
    """

    anaconda_init_command = get_powershell_anaconda_init_command(env_name=env_name)
    powershell_command = f"{anaconda_init_command};"

    if cwd is not None:
        powershell_command += f" {get_powershell_set_working_directory_string(cwd)};"

    python_run_command = get_powershell_python_run_command(script_path, flags=flags, arg_string=arg_string)
    powershell_command += f" {python_run_command};"

    if transcript:
        if transcript_name is None:
            transcript_name = Path(script_path).stem
        transcript_path = rf".\{transcript_name}_{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        powershell_command = add_transcript_to_powershell_command(powershell_command, transcript_path)

    run_command_in_powershell(powershell_command, show=show, keep_open=keep_open, blocking=blocking,
                              return_result=return_result)


def get_existing_anaconda_envs():
    """
    Returns a list of existing Anaconda environments by launching a PowerShell process to run
    a command that gets a JSON-formatted list of all the environments. The function then
    extracts the environment names from the JSON output and returns them as a list.

    :return: A list of existing Anaconda environment names.
    :rtype: list of str
    """
    # Construct the PowerShell command to get all existing environments
    ps_command = f"conda env list --json"

    # Launch a PowerShell process and run the command
    ps_output = run_command_in_powershell(ps_command, show=False, keep_open=False, return_result=True)

    # Parse the JSON output and extract the list of environment names
    return [os.path.basename(env) for env in json.loads(find_valid_json(ps_output, raise_error=True)[0])["envs"]]


def is_valid_anaconda_env(env_name):
    """
    Checks if the specified string is a valid and existing Anaconda environment name.

    :param env_name: The environment name to check.
    :type env_name: str
    :return: True if the environment exists, False otherwise.
    """
    # Get all existing anaconda environments.
    env_list = get_existing_anaconda_envs()

    # Check if the specified environment name is in the list of environment names
    return env_name in env_list


def create_anaconda_env(env_file_path, env_name=None, keep_open=False, blocking=False):
    """
    Creates an Anaconda environment from the specified environment file.

    :param env_file_path: The path to the environment file.
    :type env_file_path: str
    :param env_name: The name to give the environment. If None, the name will be taken from the environment file.
    :type env_name: str or None
    :param keep_open: Whether to keep the PowerShell window opened after the installation is finished.
    :type keep_open: bool
    :param blocking: Whether to block the current thread until the environment is created.
    :type blocking: bool
    :return: None
    """
    # Set the default environment name to None
    env_name_arg = "" if env_name is None else f"--name {env_name}"

    # Construct the PowerShell command to create the environment
    ps_command = f"conda env create {env_name_arg} --file '{env_file_path}'"

    run_command_in_powershell(ps_command, keep_open=keep_open, blocking=blocking)


def ensure_anaconda_env(env_name, env_file_path, update=False, prune=True, keep_open=True, blocking=False):
    """
    Ensures that an Anaconda environment exists and is up-to-date with the packages specified in the environment file.

    :param env_name: The name of the environment to ensure.
    :type env_name: str
    :param env_file_path: The path to the environment file.
    :type env_file_path: str
    :param update: Whether to update the environment if it already exists.
    :type update: bool
    :param prune: Whether to remove packages from the environment that are not specified in the environment file.
    :type prune: bool
    :param keep_open: Whether to keep the PowerShell window opened after the installation is finished.
    :type keep_open: bool
    :param blocking: Whether to block the current thread until the environment is created.
    :type blocking: bool
    :return: None
    """
    # Check if the environment already exists
    if is_valid_anaconda_env(env_name):
        # Update the environment if requested
        if update:
            update_anaconda_env(env_file_path, env_name, prune=prune, keep_open=keep_open, blocking=blocking)
    else:
        # Create the environment if it doesn't exist
        create_anaconda_env(env_file_path, env_name, keep_open=keep_open, blocking=blocking)


def update_anaconda_env(env_file_path, env_name=None, prune=True, keep_open=True, blocking=False):
    """
    Updates an Anaconda environment with the packages specified in the environment file.

    :param env_file_path: The path to the environment file.
    :type env_file_path: str
    :param env_name: The name of the environment to update. If None, the default environment will be updated.
    :type env_name: str or None
    :param prune: Whether to remove packages that are not specified in the environment file.
    :type prune: bool
    :param keep_open: Whether to keep the PowerShell window opened after the installation is finished.
    :type keep_open: bool
    :param blocking: Whether to block the current thread until the environment is updated.
    :type blocking: bool
    :return: None
    """
    # Set the default environment name to base
    env_name_arg = "" if env_name is None else f"--name {env_name}"

    # Construct the PowerShell command to update the environment
    ps_command = f"conda env update {env_name_arg} --file '{env_file_path}'"

    if prune:
        ps_command += " --prune"

    run_command_in_powershell(ps_command, keep_open=keep_open, blocking=blocking)


def delete_anaconda_env(env_name, keep_open=True, blocking=False):
    """
    Deletes the specified Anaconda environment.

    :param env_name: The name of the environment to delete.
    :type env_name: str
    :param keep_open: Whether to keep the PowerShell window opened after the installation is finished.
    :type keep_open: bool
    :param blocking: Whether to block the current thread until the environment is deleted.
    :type blocking: bool
    :return: None
    """
    # Construct the PowerShell command to delete the environment
    ps_command = f"conda env remove --name {env_name}"

    run_command_in_powershell(ps_command, keep_open=keep_open, blocking=blocking)
