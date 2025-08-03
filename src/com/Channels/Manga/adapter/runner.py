from typing import List
from ozkapi.src.com.api.MangaApi import MangaApi
from manga.src.models import (
    MangaDataModel,
    ChapterDataModel,
    PageDataModel,
    PanelDataModel,
)

from src.com.channels.manga.adapter.MangaMagiSegmenterAdapter import (
    MangaMagiSegmenterAdapter,
)


def get_api():
    db = r"H:\Youtube"
    api = MangaApi(local_db=db)
    return api


def get_magi(api: MangaApi):
    weights_path = r"D:\Models\magiv2\pytorch_model.bin"
    return MangaMagiSegmenterAdapter(api=api, weights=weights_path)


def hydrate_manga(api: MangaApi, manga: MangaDataModel) -> MangaDataModel:
    api.hydrate_manga(manga=manga, from_db=True)
    for chpt in manga.chapters:
        api.hydrate_chapter(chpt, from_db=True)
        for page in chpt.pages:
            api.hydrate_page(page=page)

    return manga


def main():
    pass


if __name__ == "__main__":
    main()
