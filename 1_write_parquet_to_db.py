"""
Module reads in the SYSK transcript parquet files and subsequenty writes them to a
postgresql database.
Parquet files are lazily evaluated so no more than one file is left in memory before
being written.
"""

from pathlib import Path
from typing import Union

import pandas as pd
import toml
from sqlalchemy import create_engine
from utils import read_toml

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
    return None

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataframe prior to db writing.
    Creates a copy of the episode ID value as provided and drops the ID columns.
    Also removes any columns that don't appear in the db schema.
    This helps avoid errors when writing to the db later.

    Args:
        df (pd.DataFrame): Original df

    Returns:
        pd.DataFrame: cleaned df
    """
    df["episode_id"] = df["id"].copy()
    df = df.drop("id", axis=1)
    cols = set(df.columns)
    db_cols = set(
        [
            "title",
            "link",
            "desc",
            "summary",
            "pubDate",
            "pubFormatted",
            "enc_len",
            "enc_type",
            "audio_url",
            "transcript",
            "categories",
            "chapters",
            "episode_id",
        ]
    )
    cols_to_drop = cols.difference(db_cols)
    df = df.drop(cols_to_drop, axis=1)
    return df


def read_clean_write(filepath: Union[str, Path], config_dict: dict) -> None:
    """Reads in a parquet file to a pandas dataframe, cleans it with the clean_df function
    and subsequently write it to the database.
    The function allows dataframes to be lazily evaluated so no more than one df is
    held in memory at any time.

    Args:
        filepath (Union[str, Path]): Parquet file to read.
        config_dict (dict): Config dict of the database. See write_to_db docstring.

    Yields:
        _type_: None
    """
    df = read_parquet_to_df(filepath)
    df = clean_df(df)
    yield write_to_db(df=df, table="episodes", config_dict=config_dict)


def main(data_dir: Union[str, Path], config_dict_filepath: Union[str, Path]) -> None:
    """Main function to read in, clean, and write all parquet files to database.

    Args:
        data_dir (Union[str, Path]): Directory holding all parquet files
        config_dict_filepath (Union[str, Path]): Filepath of config TOML. See write_to_db docstring.
    """
    parquet_files = get_parquet_files(data_dir)
    db_info_dict = read_toml(config_dict_filepath)["database"]
    for idx, file in enumerate(parquet_files):
        next(read_clean_write(file, db_info_dict))
        if idx % 50 == 0:
            print(f"Dataframe written. Current count: {idx}")


if __name__ == "__main__":
    main(data_dir=DATA_DIR, config_dict_filepath="db_info.toml")
