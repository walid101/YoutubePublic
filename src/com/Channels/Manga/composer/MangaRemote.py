""" """

from pathlib import Path
from typing import Union
from OZKDB.src.com.db.dbsource import DBSource
from OZKDB.src.com.db.google.googledb import GoogleDriveDB
from ozkit.src.com.Utils import Utils
from dataclasses import dataclass
from manga.src.models import MangaDataModel


class MangaStatus:
    def __init__(self, manga: MangaDataModel, progress: int, compute: str):
        self.manga = manga
        self.progress = progress
        self.compute = compute

    def to_dict():
        pass

    def from_dict():
        pass


class MangaRemote:
    def init(self, root: Union[str, Path], db: DBSource):
        pass

    # TODO:
    # 1. Have a way to maintain a json file (object) to check what titles are in what stage of progress. - List of Manga Status (Make sure this is thread safe)
    # 2. Given a compute method callback (either function or something else, etc) which will conduct the compute steps. Accepts remote db classes.
    def construct():
        # TODO: Check if remote db has the json file containing manga status's if not create it.
        pass

    def check():

        pass

    def cleanup():
        pass
