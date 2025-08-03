import time
from ozkapi.src.com.api.MangaApi import MangaApi
from manga.src.manga import Manga


def run_manga_api():
    api = MangaApi()
    core = Manga()
    manga = core.get_manga(
        title="Survival Story Of A Sword King In A Fantasy World",
        chpt_range=(1, 10),
    )
    print("Manga id: ", manga.uuid)
    print(f"Number of chapters: {len(manga.chapters)}")
    for chpt in manga.chapters:
        # Hydrate Chapters for the first time
        try:
            api.hydrate_chapter(chapter=chpt, from_db=False)
        except Exception as e:
            break
    start = time.time()
    api.save_manga(manga, depth=4, force=True, threads={"io": 8, "cpu": 8})
    print(f"TOTAL TIME TAKEN: {time.time() - start:.2f} seconds")


def test_find():
    api = MangaApi()
    db = api.db_api
    if db.mega.find_path_descriptor("Manga/Naruto/2"):
        print("Found Folder")
    else:
        print("DID NOT FIND FOLDER")
    pass


# test_find()
run_manga_api()
