import os
import subprocess


def run_command_in_powershell(command, show=True, keep_open=True, blocking=False, return_result=False):
    """
    Runs the specified command in a PowerShell process.

    :param command: The command to run.
    :type command: str
    :param show: Whether to show the PowerShell window.
    :type show: bool
    :param keep_open: Whether to keep the PowerShell window open after the command is finished.
    :type keep_open: bool
    :param blocking: Whether to block the current thread until the command is finished.
    :type blocking: bool
    :param return_result: Whether to return the result of the command. If true then the command is blocking.
    :type return_result: bool
    :return: If return_result is true then the result of the command is returned. Otherwise, None is returned.
    :rtype: Any or None
    """

    arg_string = "-NoExit" if keep_open else ""
    if not show:
        # Hide the PowerShell window, overriding the keep_open argument if necessary
        # (to avoid that keep_open is True, then the window would be hidden but not closed)
        arg_string = "-WindowStyle Hidden"

    powershell_command = f"{arg_string} -Command \"{command}\""

    if return_result:
        return os.popen(f"powershell.exe {powershell_command}").read()
    elif blocking:
        process = subprocess.Popen(f"start powershell {powershell_command}", shell=True, stdout=subprocess.PIPE,)
        _exit_code = process.communicate()
    else:
        os.system(f"start powershell {powershell_command}")


def add_transcript_to_powershell_command(command, transcript_path):
    """
    Adds a transcript to the specified PowerShell command.

    :param command: The command to add the transcript to.
    :type command: str
    :param transcript_path: The path to the transcript file.
    :type transcript_path: str
    :return: The command with the tran^script added.
    :rtype: str
    """

    return f"Start-Transcript -Path '{transcript_path}'; {command}; Stop-Transcript"


def get_powershell_datetime_command_string(date_format="yyyy-MM-dd_HH-mm-ss"):
    """
    Gets the PowerShell command string to get the current date and time.

    :param date_format: The format to use for the date and time.
    :type date_format: str
    :return: The PowerShell command string to get the current date and time.
    :rtype: str
    """
    return rf"$(Get-Date -Format \"{date_format}\")"


def get_powershell_set_working_directory_string(cwd):
    """
    Gets the PowerShell command string to set the working directory.

    :param cwd: The working directory to set.
    :type cwd: str
    :return: The PowerShell command string to set the working directory.
    :rtype: str
    """
    # Convert the cwd argument to an absolute path
    cwd = os.path.abspath(cwd)

    return f"Set-Location -Path '{cwd}'"
