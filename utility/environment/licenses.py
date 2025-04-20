import importlib.metadata
import re


common_license_regex_dict = dict(
    MIT=r"MIT",
    BSD=r"BSD",
    Apache=r"Apache",
    GPL=r"GPL",
    LGPL=r"LGPL",
    NoLicense=r"No license"
)


def clean_license_string(license_str):
    """
    Cleans the license string by removing all but the first line. If the license string is None, returns "No license".

    :param license_str: The license string
    :type license_str: str or None
    :return: The cleaned license string
    :rtype: str
    """
    return license_str.split("\n")[0] if license_str is not None else "No license"


def match_license_string_to_common(license_str):
    """
    Matches a license string to a common license name. If no match is found, returns "Other". First cleans the license
    string with clean_license_string.

    :param license_str: The license string
    :type license_str: str or None
    :return: The common license name
    :rtype: str
    """
    license_str = clean_license_string(license_str)
    for license_name, license_regex in common_license_regex_dict.items():
        if re.search(license_regex, license_str):
            return license_name
    return "Other"


def get_distribution_licenses(return_matched=True):
    """
    Returns the names and licenses of all installed distributions.

    :return: A list of tuples containing the distribution name and license
    :rtype: list of (str, str)
    """
    installed_distributions = list(importlib.metadata.distributions())
    dist_names = [dist.metadata["name"] for dist in installed_distributions]
    dist_licenses = [dist.metadata["license"] for dist in installed_distributions]
    if return_matched:
        dist_licenses = [match_license_string_to_common(_license) for _license in dist_licenses]
    return list(zip(dist_names, dist_licenses))
