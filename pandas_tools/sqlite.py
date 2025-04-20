import sqlite3
from typing import Dict

from pandas import read_sql_query
from tqdm.auto import tqdm


def list_sqlite_tables(db_file_path):
    """
    Lists all tables in a SQLite database file.

    :param db_file_path: The path to the SQLite database file
    :type db_file_path: str
    :return: A list of table names
    :rtype: list of str
    """
    with sqlite3.connect(db_file_path) as dbcon:
        tables = list(read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])
    return tables


def sqlite_between_query(column, start, end):
    """
    Creates a BETWEEN query for a SQLite database.

    :param column: The column to query
    :type column: str
    :param start: The start of the range
    :type start: int or float
    :param end: The end of the range
    :type end: int or float
    :return: A BETWEEN query for a SQLite database
    :rtype: str
    """

    return f"WHERE {column} BETWEEN {start} AND {end}"


def read_sqlite_table(db_file_path, table, additional_query=""):
    """
    Reads a table from a SQLite database file into a pandas.DataFrame object.

    :param db_file_path: The path to the SQLite database file
    :type db_file_path:  str
    :param table: The name of the table to read
    :type table: str
    :param additional_query: Additional query to add to the end of the SQL query
    :type additional_query: str
    :return: A pandas.DataFrame object
    :rtype: pandas.DataFrame
    """
    with sqlite3.connect(db_file_path) as dbcon:
        out = read_sql_query(f"SELECT * from {table} {additional_query}", dbcon)
    return out


def read_sqlite(db_file_path):
    """
    Reads all tables from a SQLite database file into a dictionary of pandas.DataFrame objects.
    Adapted from https://stackoverflow.com/a/67938218

    :param db_file_path: The path to the SQLite database file
    :type db_file_path: str
    :return: A dictionary of pandas.DataFrame objects
    :rtype: Dict[str, pandas.DataFrame]
    """
    tables = list_sqlite_tables(db_file_path)
    out = {tbl: read_sqlite_table(db_file_path, tbl) for tbl in tqdm(tables)}
    return out


def write_sqlite_table(db_file_path, table, df, if_exists="replace"):
    """
    Writes a pandas.DataFrame object to a table in a SQLite database file.

    :param db_file_path: The path to the SQLite database file
    :type db_file_path: str
    :param table: The name of the table to write to
    :type table: str
    :param df: The pandas.DataFrame object to write
    :type df: pandas.DataFrame
    :param if_exists: What to do if the table already exists. "replace" will replace the table, "append" will append to
    the table, and "fail" will raise an error.
    :type if_exists: str
    """
    with sqlite3.connect(db_file_path) as dbcon:
        df.to_sql(table, dbcon, if_exists=if_exists)


def write_sqlite(db_file_path, df_dict, if_exists="replace"):
    """
    Writes a dictionary of pandas.DataFrame objects to a SQLite database file.

    :param db_file_path: The path to the SQLite database file
    :type db_file_path: str
    :param df_dict: A dictionary of pandas.DataFrame objects
    :type df_dict: Dict[str, pandas.DataFrame]
    :param if_exists: What to do if the table already exists. "replace" will replace the table, "append" will append to
    the table, and "fail" will raise an error.
    :type if_exists: str
    """
    for table, df in tqdm(df_dict.items()):
        write_sqlite_table(db_file_path, table, df, if_exists=if_exists)
