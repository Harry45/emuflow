"""
Author: Arrykrishna
Date: May 2024
Email: arrykrish@gmail.com
Project: Normalising flows for joint analysis.
"""

import os
import dill
import pickle
from typing import Any
import logging
import sys
from datetime import datetime
from hydra.core.hydra_config import HydraConfig

NOW = datetime.now()
FORMATTER = logging.Formatter("[%(levelname)s] - %(asctime)s - %(name)s : %(message)s")
CONSOLE_FORMATTER = logging.Formatter("[%(levelname)s]: %(message)s")
DATETIME = NOW.strftime("%d-%m-%Y-%H-%M")


def pickle_save(file: list, folder: str, fname: str) -> None:
    """Stores a list in a folder.
    Args:
        list_to_store (list): The list to store.
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    """

    # create the folder if it does not exist
    os.makedirs(folder, exist_ok=True)

    # use compressed format to store data
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "wb") as dummy:
        pickle.dump(file, dummy, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(folder: str, fname: str):
    """Reads a list from a folder.
    Args:
        folder_name (str): The name of the folder.
        file_name (str): The name of the file.
    Returns:
        Any: the stored file
    """
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "rb") as dummy:
        file = pickle.load(dummy)
    return file


def dill_save(file: Any, fname: str) -> None:
    """Stores a file, for example, MCMC samples.
    Args:
        file (Any): the file we want to store.
        fname (str): the name of the file.
    """

    # get the folder where the files are stored
    folder = HydraConfig.get()["runtime"]["output_dir"]

    # use compressed format to store data
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "wb") as dummy:
        dill.dump(file, dummy)


def dill_load(folder: str, fname: str) -> Any:
    """Reads a file from a folder.
    Args:
        folder(str): the name of the folder.
        file (str): the name of the file.
    Returns:
        Any: the stored file
    """
    path = os.path.join(folder, fname)
    with open(path + ".pkl", "rb") as dummy:
        file = dill.load(dummy)
    return file


def get_logger() -> logging.Logger:
    """Generates a logging file for storing all information.

    Returns:
        logging.Logger: the logging module
    """
    folder = HydraConfig.get()["runtime"]["output_dir"]
    fname = os.path.join(folder, "main.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fhandler = logging.FileHandler(filename=fname)
    fhandler.setLevel(logging.DEBUG)
    fhandler.setFormatter(FORMATTER)

    logger.addHandler(fhandler)

    return logger
