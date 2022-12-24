"""
Module reads in the SYSK transcript parquet files and subsequenty writes them to a
postgresql database.
Parquest files are lazily evaluated so no more than one file is left in memory before
being written.
"""

from pathlib import Path
from typing import Union

import pandas as pd
import toml
from sqlalchemy import create_engine

DATA_DIR = Path(r"Data/transcript_parquets")


def get_parquet_files(data_dir: Union[str, Path]) -> list[Path]:
    """Returns a list of all parquest files in a directory.
    Does not look in sub folders recursively.

    Args:
        data_dir (Union[str,Path]): Folder to look for.

    Returns:
        list[Path]: List of parquet filepaths.
    """

    return list(Path(data_dir).glob("*.parquet"))


def read_parquet_to_df(filepath: Union[str, Path]) -> pd.DataFrame:
    """returns a pandas dataframe from parquet file

    Args:
        filepath (Union[str,Path]): parquet file path

    Returns:
        pd.DataFrame: pandas df
    """
    return pd.read_parquet(filepath)


def read_df_text(episode_df: pd.DataFrame) -> str:
    """returns the transcript text from an individual episode dataframe

    Args:
        episode_df (pd.DataFrame): podcast episode dataframe

    Returns:
        str: episode transcript
    """
    return episode_df["transcript"].values[0]


def read_toml(toml_file: Union[str, Path]) -> dict:
    """reads info from a toml file

    Args:
        config_file (Union[str,Path]): filepath of toml file

    Returns:
        dict: dictionary containing toml data
    """
    return toml.load(toml_file)


def write_to_db(df: pd.DataFrame, table: str, config_dict: dict) -> None:
    """writes pandas dataframe to a postgres table.

    Args:
        df (pd.DataFrame): df to be written
        table (str): name of table to write to
        config_dict (dict): config dict outlining the db parameters.
        Should include host, database, user, and password.

    Returns:
        None: None
    """
    host = config_dict["host"]
    database = config_dict["database"]
    user = config_dict["user"]
    password = config_dict["password"]
    engine = create_engine(f"postgresql://{user}:{password}@{host}:5432/{database}")
    df.to_sql(table, engine, if_exists="append", index=False)
    print("Dataframe successfully written.")
    return None


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataframe prior to db writing.
    Creates a copy of the episode ID value as provided and drops the ID columns

    Args:
        df (pd.DataFrame): Original df

    Returns:
        pd.DataFrame: cleaned df
    """
    df["episode_id"] = df["id"].copy()
    df = df.drop("id", axis=1)
    return df

def write_all_parquet(filelist:list, *args, **kwargs):
    pass

if __name__ == "__main__":
    parquet_files = get_parquet_files(DATA_DIR)
    test_df = read_parquet_to_df(parquet_files[0])
    test_df = clean_df(test_df)
    db_info_dict = read_toml("db_info.toml")["database"]
    write_to_db(df=test_df, table="episodes", config_dict=db_info_dict)
