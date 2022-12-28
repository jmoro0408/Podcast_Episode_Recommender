"""
Small module that holds useful functions.
"""

import pandas as pd
from sqlalchemy import create_engine
import toml
from typing import Union
from pathlib import Path

def read_toml(toml_file: Union[str, Path]) -> dict:
    """reads info from a toml file

    Args:
        config_file (Union[str,Path]): filepath of toml file

    Returns:
        dict: dictionary containing toml data
    """
    return toml.load(toml_file)

def read_from_db(query:str, config_dict:dict) -> pd.DataFrame:
    """Reads a SQL query into a pandas dataframe.

    Args:
        query (str): Query to pass to db
        config_dict (dict): config dict outlining the db parameters.
        Should include host, database, user, and password.

    Returns:
        pd.DataFrame: _description_
    """
    host = config_dict["host"]
    database = config_dict["database"]
    user = config_dict["user"]
    password = config_dict["password"]
    engine = create_engine(f"postgresql://{user}:{password}@{host}:5432/{database}")
    return pd.read_sql_query(query, engine)


if __name__ == "__main__":

    db_info_dict = read_toml(r"db_info.toml")["database"]
    df = read_from_db("""SELECT * from episodes""", db_info_dict)
    print(df.head())