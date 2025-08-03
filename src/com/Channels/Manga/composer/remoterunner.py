import os
from typing import Dict, List
from manga.src.manga import MangaCon
from ozkapi.src.com.api.MangaApi import MangaApi
import torch
from src.com.channels.manga.adapter.MangaMagiSegmenterAdapter import (
    MangaMagiSegmenterAdapter,
)

from src.com.channels.manga.audio.MangaAudio import MangaAudio
from src.com.channels.manga.auxilary.MangaAuxilary import MangaAuxilary
from src.com.channels.manga.composer.MangaComposer import (
    MangaComposer,
    MangaComposerConfig,
)
from .MangaNarrator import MangaNarratorGoogle
from manga.src.models import (
    PageDataModel,
    PanelDataModel,
    CharacterDataModel,
    ImageModel,
)

from ozkai_tts.KKROTTSModel import KKROTTSModel
from ozkai_tts.ELVNTTSRemoteModel import ELVNTTSRemoteModel

# from ozkai_tts.WHISPERSRTModel import WHISPERSRTModel
from ozkai_tts.WHISPERXSRTModel import WHISPERXSRTModel
from ozkit.src.com.network.Network import Network
from ozkit.src.com.Utils import Utils


def get_api2():
    return MangaCon(proxy=False)


def get_api():
    db = r"/content/drive/MyDrive/MangaYoutube/Youtube"
    api = MangaApi(local_db=db)
    return api


def get_magi(api: MangaApi):
    weights_path = r"/content/drive/MyDrive/magiv2/pytorch_model.bin"
    return MangaMagiSegmenterAdapter(api=api, weights=weights_path)


def get_elvn():
    model_id = "eleven_flash_v2_5"
    voice_parameters = {
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "stability": 0.7,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
    }
    return ELVNTTSRemoteModel(model_id=model_id, voice_params=voice_parameters)


def get_kkro(name: str = "am_eric"):
    model_path = r"/content/drive/MyDrive/Kokoro-82M/kokoro-v1_0.pth"
    config_path = r"/content/drive/MyDrive/Kokoro-82M/config.json"
    voice_path = os.path.join(
        r"/content/drive/MyDrive/Kokoro-82M/voices", name + ".pt"
    )  # Eric is the anime voice

    return KKROTTSModel(model=model_path, config=config_path, voice=voice_path)


def get_whisperx():
    whisper_path = "large-v3"
    whisperx = WHISPERXSRTModel(whisper_path=whisper_path)
    return whisperx


def run_narrator():
    manga_title = "Survival Story Of A Sword King In A Fantasy World"
    api = get_api()
    narrator = MangaNarratorGoogle(api=api, manga=manga_title)
    manga = api.get_manga(manga_title=manga_title)
    api.hydrate_manga(manga=manga, from_db=True)
    panels: List[PanelDataModel] = []
    for chpt in manga.chapters:
        api.hydrate_chapter(chpt, from_db=True)
        for page in chpt.pages:
            api.hydrate_page(page=page)
            page_panels = page.panels
            if page_panels:
                panels.extend(page_panels)
        break

    print("Length of panels: ", len(panels), "\n")
    panels.sort(key=lambda panel: (panel.page, panel.ord))
    narration: List[Dict] = narrator.summarize_panels(
        panels=panels[0:30], batch_size=15
    )
    for entry in narration:
        # print all summaries
        print(f"page: {entry['page']}")
        print(f"panel: {entry['panel']}")
        print(f"narration: {entry['text']}")


def test_search_character():
    manga_title = "Survival Story Of A Sword King In A Fantasy World"
    api = get_api()
    narrator = MangaNarratorGoogle(api=api, manga=manga_title)
    char_list: List[CharacterDataModel] = []
    char = "Atisse"
    search = narrator.search_character(character=char, init=char_list)
    if search:
        print("SEARCH: ", search.name)
    else:
        print("Char not found: ", char)
    for ch in char_list:
        print("Char name: ", ch.name)
    ImageModel.fetch_image_static(url=search.image_url).show()


def run_test():
    api = get_api()
    manga = api.manga_api.get_manga(
        title="The Regressed Mercenary Has a Plan", chpt_range=(1, 3), params={}
    )
    for chpt in manga.chapters:
        api.manga_api.get_pages(chapter=chpt)

    api.save_manga(manga=manga, force=True)


def run_na_panel_test():
    manga_title = "Survival Story Of A Sword King In A Fantasy World"
    api = get_api()
    manga = api.get_manga(manga_title=manga_title)
    api.hydrate_manga(manga=manga)
    for chpt in manga.chapters[0:1]:
        api.hydrate_chapter(chpt, from_db=True)
        for page in chpt.pages:
            api.hydrate_page(page=page)
            # Check for na panel summaries
            for panel in page.panels:
                narration = panel.meta.get("summary")
                if narration and narration == "N/A":
                    print(
                        f"Panel {panel.ord} in page {page.ord} in chapter {chpt.ord} is N/A, narration: {narration}"
                    )
                else:
                    print(narration, "\n\n")


# @Network.proxy
def run_magi_segmenter():
    title = "An Exiled Blacksmith Uses His Cheat Skills to Forge Legends"
    api = get_api()
    magi = get_magi(api=api)
    chapter_range = (1, 10)
    manga = api.hydrate_manga(
        api.get_manga(manga_title=title, chapter_range=chapter_range), from_db=False
    )
    for chpt in manga.chapters:
        print(f"Hydrating Chapter: {chpt.ord}...\n")
        if not api.hydrate_chapter(chapter=chpt).pages:
            chapter = api.hydrate_chapter(chapter=chpt, from_db=False)
            api.save_chapter(chapter=chapter, depth=2)
    chapter_range = (6, 10)
    batch_size = 3
    for chapter in manga.chapters:
        if (
            int(chapter.ord) >= chapter_range[0]
            and int(chapter.ord) <= chapter_range[-1]
        ):
            # Process pages in batches
            print(f"Segmenting chapter: {chapter.ord}")
            for i in range(0, len(chapter.pages), batch_size):
                batch_pages = chapter.pages[i : i + batch_size]
                magi.segment_pages(pages=batch_pages, characters=None)  # Saves Pages


# def run_thumbnail_selector():
#     api = get_api()
#     manga_title = "Tsuiho Sareta Ore Ga Hazure Gift"
#     chapter_range = (2, 8)
#     chapter_filter = (
#         lambda chapter_num, **kwargs: chapter_range[0]
#         <= float(chapter_num)
#         <= chapter_range[1]
#     )
#     manga = api.get_manga(
#         manga_title=manga_title,
#         chapter_range=chapter_range,
#         chapter_filter=chapter_filter,
#     )
#     panels = []
#     if not manga.chapters:
#         manga = api.hydrate_manga(manga=manga)
#     for chapter in manga.chapters:
#         api.hydrate_chapter(chapter=chapter)
#         for page in chapter.pages:
#             api.hydrate_page(page=page)
#             panels.extend(page.panels)
#     aux = MangaAuxilary(api=api, manga=manga_title)
#     thumbnail = aux.construct_thumbnail(panels=panels)
#     print("THUMBNAIL: ", thumbnail.to_dict())


# def check_cudnn():
#     import torch

#     print("PyTorch version:", torch.__version__)
#     print("CUDA version (PyTorch compiled with):", torch.version.cuda)
#     print("cuDNN version:", torch.backends.cudnn.version())
#     print("cuDNN enabled:", torch.backends.cudnn.enabled)


# def check_device():
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
#     else:
#         device = torch.device("cpu")
#         print("CUDA not available. Using CPU.")

#     return device


# def run_title_generation():
#     api = get_api()
#     manga_title = Utils.sanitize(
#         "Apocalypse Bringer Mynoghra%3A World Conquest Starts with the Civilization of Ruin"
#     )
#     chapter_range = (1, 4)
#     chapter_filter = (
#         lambda chapter_num, **kwargs: chapter_range[0]
#         <= float(chapter_num)
#         <= chapter_range[1]
#     )
#     manga = api.get_manga(
#         manga_title=manga_title,
#         chapter_range=chapter_range,
#         chapter_filter=chapter_filter,
#     )
#     chapters = []
#     references = [
#         (
#             "Hero Got NTR-ED By Demons So He Reincarnates As Dark Lord To Take Revenge",
#             441580,
#         ),
#         ("When The Prodigy With a GOD BODY Enters The Dungeon Academy", 218365),
#         (
#             "ISEKAID For A Slow-Life But Got A Unique Water Magic That Is Too OP To Stay Still",
#             272000,
#         ),
#         ("Lonely Boy Saves His Teacher From NTR And She Couldn't Hold Back", 197950),
#         ("His Secret Skill Lets Him Steal Any Monster’s Ability.", 95000),
#         # ("He Was HUMILIATED and Left to Die, But Reincarnated as a GOD!", 293142),
#     ]
#     if not manga.chapters:
#         api.hydrate_manga(manga=manga)
#     for chapter in manga.chapters:
#         if not chapter_range[0] <= float(chapter.ord) <= chapter_range[-1]:
#             continue  # Skip.
#         api.hydrate_chapter(chapter=chapter)
#         chapters.append(chapter)
#     aux = MangaAuxilary(api=api, manga=manga_title)
#     title = aux.construct_title(chapters=chapters, references=references)  # Create Ref
#     print("Generated Title: ", title)


# def run_upsample():
#     whisperx = get_whisperx()
#     kkro = get_kkro(name="am_eric")  # Eric > Adam >= Onyx
#     filepath = r"H:\Youtube\Videos\video\echo_voiceover\final\Almark_8_8_video.mp4"
#     output_path = (
#         r"H:\Youtube\Videos\video\echo_voiceover\final\Almark_8_8_video_upsampled.mp4"
#     )
#     aud = MangaAudio(tts=kkro, whisper=whisperx)
#     aud.resample(
#         filepath=filepath, start_sample=24000, end_sample=44100, output_path=output_path
#     )


# @Network.proxy
def run_composer():
    db = r"/content/drive/MyDrive/MangaYoutube/Youtube"
    narration_ex_path = r"H:\Youtube\narration_ex.txt"
    manga = "An Exiled Blacksmith Uses His Cheat Skills to Forge Legends"  # "Mikoto-chan wa Kirawaretakunai!"  # "Apocalypse Bringer Mynoghra: World Conquest Starts with the Civilization of Ruin"
    manga_title_main = manga
    manga_id = "9bfa6f66-174b-4ed1-bb75-2e31a77ef0b6"  # "4e338970-63e8-489c-b937-5e7e3005e1e3"  # None  # "a967d6e2-e269-4ab8-9306-04ffc4c2cb52"
    chapter_range = (1, 10)  # DO it in batches of 10 at a time. First 20
    chapter_filter = (
        lambda chapter_num, **kwargs: chapter_range[0]
        <= float(chapter_num)
        <= chapter_range[1]
    )
    api = MangaApi(
        local_db=db, proxy=False
    )  # Proxy off because MangaDEX CDN will still rate limit datacenter based ip ranges.
    output_dir = rf"/content/drive/MyDrive/MangaYoutube/Youtube/Videos/Manga/production/{Utils.sanitize(manga_title_main)}"
    bg_filepath = r"/content/drive/MyDrive/MangaYoutube/Youtube/Manga/bgm_alt.wav"  # r"H:\Youtube\Manga\bgm.wav"
    bg_volume = 0.032
    na_volume = 1.15
    whisperx = get_whisperx()
    elvn = get_elvn()
    kkro = get_kkro(
        name="bm_daniel"  # "bm_daniel"
    )  # Eric > Adam = Echo >= Onyx, # Britain:  Daniel = Lewis >= George
    tts = kkro
    segmenter = get_magi(api=api)
    config = MangaComposerConfig(
        force_panel_coords=False,
        force_panel_segments=False,
        tts_speed=1.10,
        fps=30,
        ref_chapter_summaries=True,
        bg_filepath=bg_filepath,
        bg_volume=bg_volume,
        volume=na_volume,
        temperature=0.64,  # Dont go below this, LLM gets confused with JSON structure.
    )
    aux = MangaAuxilary(api=api, manga=manga)
    composer = MangaComposer(
        manga=manga,
        api=api,
        whisper=whisperx,  # whisper,
        tts=tts,
        segmenter=segmenter,
        aux=aux,
        config=config,
        chapter_range=chapter_range,
        manga_id=manga_id,
        output_dir=output_dir,
        chapter_filter=chapter_filter,
    )
    composer.construct(
        narration_ex=narration_ex_path,
        batch_size=150,
    )


def main():
    # run_narrator()
    # test_search_character()
    # run_test()
    # run_eleven_labs_test()
    # run_magi_segmenter()
    # run_chapter_summarizer()
    # run_thumbnail_selector()
    # run_title_generation()
    # run_hydrate_chapter()
    # run_mangacon()
    run_composer()
    # run_magi_segmenter()

    # check_device()
    # check_cudnn()


if __name__ == "__main__":
    main()
