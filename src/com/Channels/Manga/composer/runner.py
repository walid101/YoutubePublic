import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
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
from ozkai_tts.WHISPERSRTModel import WHISPERSRTModel
from ozkai_tts.WHISPERXSRTModel import WHISPERXSRTModel
from ozkit.src.com.network.Network import Network
from ozkit.src.com.Utils import Utils


def get_mangacon_api():
    return MangaCon(proxy=False)


def get_manga_api(db: str = r"H:\Youtube"):
    api = MangaApi(local_db=db)
    return api


def get_magi(api: MangaApi, weights_path: str = r"D:\Models\magiv2\pytorch_model.bin"):
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


def get_kkro(name: str = "am_eric", kkro_path: str = r"D:\Models\TTS\Kokoro-82M"):
    # Checks
    if not os.path.exists(kkro_path):
        raise FileNotFoundError(f"KKRO base folder not found at {kkro_path}")

    model_path = os.path.join(kkro_path, r"kokoro-v1_0.pth")
    config_path = os.path.join(kkro_path, r"config.json")
    voice_path = os.path.join(kkro_path, r"voices", rf"{name}.pt")
    return KKROTTSModel(model=model_path, config=config_path, voice=voice_path)


def get_whisperx(whisper_path="large-v3"):
    whisperx = WHISPERXSRTModel(whisper_path=whisper_path)
    return whisperx


def check_cudnn():
    import torch

    print("PyTorch version:", torch.__version__)
    print("CUDA version (PyTorch compiled with):", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("cuDNN enabled:", torch.backends.cudnn.enabled)


def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    return device


# @Network.proxy
def run_magi_segmenter():
    title = "The Exiled Alchemist Unwittingly Becomes a Legend Living Happily on the Frontier with his Yandere Sister"
    api = get_manga_api()
    magi = get_magi(api=api)
    chapter_range = (1, 16)
    manga = api.hydrate_manga(
        api.get_manga(manga_title=title, chapter_range=chapter_range), from_db=True
    )
    for chpt in manga.chapters:
        print(f"Hydrating Chapter: {chpt.ord}...\n")
        if not api.hydrate_chapter(chapter=chpt).pages:
            chapter = api.hydrate_chapter(chapter=chpt, from_db=False)
            api.save_chapter(chapter=chapter, depth=2)
    # chapter_range = (6, 10)
    batch_size = 3
    for chapter in manga.chapters:
        if (
            Utils.ifloat(chapter.ord) >= chapter_range[0]
            and Utils.ifloat(chapter.ord) <= chapter_range[-1]
        ):
            # Process pages in batches
            print(f"Segmenting chapter: {chapter.ord}")
            for i in range(0, len(chapter.pages), batch_size):
                batch_pages = chapter.pages[i : i + batch_size]
                magi.segment_pages(pages=batch_pages, characters=None)  # Saves Pages


def run_chapter_summarizer():
    api = get_manga_api()
    manga_title = "Almark"
    manga = api.get_manga(manga_title=manga_title, chapter_range=(4, 4))
    narrator = MangaNarratorGoogle(api=api, manga=manga_title)
    narrator.summarize_chapters(chapters=manga.chapters)
    # Get first chapter


def run_thumbnail_selector():
    api = get_manga_api()
    manga_title = "An Exiled Blacksmith Uses His Cheat Skills to Forge Legends"
    chapter_range = (1, 2)
    chapter_filter = (
        lambda chapter_num, **kwargs: chapter_range[0]
        <= Utils.ifloat(chapter_num)
        <= chapter_range[1]
    )
    manga = api.get_manga(
        manga_title=manga_title,
        chapter_range=chapter_range,
        chapter_filter=chapter_filter,
    )
    panels = []
    if not manga.chapters:
        manga = api.hydrate_manga(manga=manga)
    for chapter in manga.chapters:
        api.hydrate_chapter(chapter=chapter)
        for page in chapter.pages:
            api.hydrate_page(page=page)
            panels.extend(page.panels)
    aux = MangaAuxilary(api=api, manga=manga_title)
    thumbnail = aux.construct_thumbnail(panels=panels)
    print("THUMBNAIL: ", thumbnail.to_dict())


def run_title_generation():
    api = get_manga_api()
    manga_title = Utils.sanitize(
        "Apocalypse Bringer Mynoghra%3A World Conquest Starts with the Civilization of Ruin"
    )
    chapter_range = (1, 4)
    chapter_filter = (
        lambda chapter_num, **kwargs: chapter_range[0]
        <= float(chapter_num)
        <= chapter_range[1]
    )
    manga = api.get_manga(
        manga_title=manga_title,
        chapter_range=chapter_range,
        chapter_filter=chapter_filter,
    )
    chapters = []
    references = [
        (
            "Hero Got NTR-ED By Demons So He Reincarnates As Dark Lord To Take Revenge",
            441580,
        ),
        ("When The Prodigy With a GOD BODY Enters The Dungeon Academy", 218365),
        (
            "ISEKAID For A Slow-Life But Got A Unique Water Magic That Is Too OP To Stay Still",
            272000,
        ),
        ("Lonely Boy Saves His Teacher From NTR And She Couldn't Hold Back", 197950),
        ("His Secret Skill Lets Him Steal Any Monster’s Ability.", 95000),
        # ("He Was HUMILIATED and Left to Die, But Reincarnated as a GOD!", 293142),
    ]
    if not manga.chapters:
        api.hydrate_manga(manga=manga)
    for chapter in manga.chapters:
        if not chapter_range[0] <= float(chapter.ord) <= chapter_range[-1]:
            continue  # Skip.
        api.hydrate_chapter(chapter=chapter)
        chapters.append(chapter)
    aux = MangaAuxilary(api=api, manga=manga_title)
    title = aux.construct_title(chapters=chapters, references=references)  # Create Ref
    print("Generated Title: ", title)


def run_upsample():
    whisperx = get_whisperx()
    kkro = get_kkro(name="am_eric")  # Eric > Adam >= Onyx
    filepath = r"H:\Youtube\Videos\video\echo_voiceover\final\Almark_8_8_video.mp4"
    output_path = (
        r"H:\Youtube\Videos\video\echo_voiceover\final\Almark_8_8_video_upsampled.mp4"
    )
    aud = MangaAudio(tts=kkro, whisper=whisperx)
    aud.resample(
        filepath=filepath, start_sample=24000, end_sample=44100, output_path=output_path
    )


# @Network.proxy
def run_composer():
    db = r"H:\Youtube"
    narration_ex_path = r"H:\Youtube\narration_ex.txt"
    manga = "Monster Stein"
    manga_title_main = manga
    manga_id = "2c22152b-9a6f-48f3-adb6-27b2f4cc2758"  # "4e338970-63e8-489c-b937-5e7e3005e1e3"  # None  # "a967d6e2-e269-4ab8-9306-04ffc4c2cb52"
    chapter_range = (1, 16)  # DO it in batches of 10 at a time. First 20
    chapter_filter = (
        lambda chapter_num, **kwargs: chapter_range[0]
        <= float(chapter_num)
        <= chapter_range[1]
    )
    api = MangaApi(
        local_db=db, proxy=False
    )  # Proxy off because MangaDEX CDN will still rate limit datacenter based ip ranges.
    output_dir = (
        rf"H:\Youtube\Videos\Manga\production\{Utils.sanitize(manga_title_main)}"
    )
    bg_filepath = r"H:\Youtube\Manga\bgm.wav"  # r"H:\Youtube\Manga\bgm.wav"
    bg_volume = 0.025
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


def run_mangacon():
    title = "Monster Stein"  # "sensou kyoushitsu"
    id = None  # "7342/sensou-kyoushitsu"
    chapter_range = (1, 11)
    provider = "mangahere"
    chapter_filter = (
        lambda chapter_num, **kwargs: chapter_range[0]
        <= float(chapter_num)
        <= chapter_range[1]
    )
    mangacon_api = get_mangacon_api()
    db_api = get_manga_api()
    manga = mangacon_api.get_manga(
        title=title,
        id=id,
        chapter_filter=chapter_filter,
        provider=provider,
    )
    print("MANGA: ", manga.to_dict())
    for chpt in manga.chapters:
        print("Chapter: ", chpt.ord, " ", chpt.to_dict())
        mangacon_api.get_pages(chapter=chpt, provider=provider)
    db_api.save_manga(manga=manga, force=True)


def parse_references(ref_str: str) -> List[Tuple[str, int]]:
    """Parse references from JSON string or file path"""
    try:
        if ref_str.endswith(".json"):
            with open(ref_str, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(ref_str)

        # Convert to list of tuples
        if isinstance(data, list):
            return [(item["title"], item["views"]) for item in data]
        elif isinstance(data, dict):
            return [(title, views) for title, views in data.items()]
        else:
            raise ValueError("Invalid references format")
    except Exception as e:
        raise ValueError(f"Failed to parse references: {e}")


def parse_chapter_range(range_str: str) -> Tuple[float, float]:
    """Parse chapter range from string format like '1-16' or '1,16' or '1-19.2'"""
    if "-" in range_str:
        start, end = range_str.split("-", 1)
    elif "," in range_str:
        start, end = range_str.split(",", 1)
    else:
        raise ValueError(
            f"Invalid chapter range format: {range_str}. Use 'start-end' or 'start,end'"
        )

    return (float(start.strip()), float(end.strip()))


# @Network.proxy
def run_composer_with_params(
    manga_title: str,
    chapter_range: Tuple[float, float],
    manga_id: Optional[str] = None,
    db_path: str = r"H:\Youtube",
    narration_ex_path: str = r"H:\Youtube\narration_ex.txt",
    output_dir: Optional[str] = None,
    bg_filepath: str = r"H:\Youtube\Manga\bgm.wav",
    weights_path: str = r"D:\Models\magiv2\pytorch_model.bin",
    whisper_path: str = "large-v3",
    kkro_path: str = r"D:\Models\TTS\Kokoro-82M",
    voice_name: str = "bm_daniel",
    tts_speed: float = 1.10,
    fps: int = 30,
    bg_volume: float = 0.025,
    na_volume: float = 1.15,
    temperature: float = 0.64,
    batch_size: int = 150,
):
    """Run composer with parameters"""
    if output_dir is None:
        output_dir = (
            rf"H:\Youtube\Videos\Manga\production\{Utils.sanitize(manga_title)}"
        )

    chapter_filter = (
        lambda chapter_num, **kwargs: chapter_range[0]
        <= float(chapter_num)
        <= chapter_range[1]
    )

    api = MangaApi(local_db=db_path, proxy=False)
    whisperx = get_whisperx(whisper_path=whisper_path)
    kkro = get_kkro(name=voice_name, kkro_path=kkro_path)
    segmenter = get_magi(api=api, weights_path=weights_path)

    config = MangaComposerConfig(
        force_panel_coords=False,
        force_panel_segments=False,
        tts_speed=tts_speed,
        fps=fps,
        ref_chapter_summaries=True,
        bg_filepath=bg_filepath,
        bg_volume=bg_volume,
        volume=na_volume,
        temperature=temperature,
    )

    aux = MangaAuxilary(api=api, manga=manga_title)
    composer = MangaComposer(
        manga=manga_title,
        api=api,
        whisper=whisperx,
        tts=kkro,
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
        batch_size=batch_size,
    )


def run_thumbnail_selector_with_params(
    manga_title: str, chapter_range: Tuple[float, float], db_path: str = r"H:\Youtube"
):
    """Run thumbnail selector with parameters"""
    api = get_manga_api(db=db_path)
    chapter_filter = (
        lambda chapter_num, **kwargs: chapter_range[0]
        <= Utils.ifloat(chapter_num)
        <= chapter_range[1]
    )
    manga = api.get_manga(
        manga_title=manga_title,
        chapter_range=chapter_range,
        chapter_filter=chapter_filter,
    )
    panels = []
    if not manga.chapters:
        manga = api.hydrate_manga(manga=manga)
    for chapter in manga.chapters:
        api.hydrate_chapter(chapter=chapter)
        for page in chapter.pages:
            api.hydrate_page(page=page)
            panels.extend(page.panels)
    aux = MangaAuxilary(api=api, manga=manga_title)
    thumbnail = aux.construct_thumbnail(panels=panels)
    print("THUMBNAIL: ", thumbnail.to_dict())


def run_title_generation_with_params(
    manga_title: str,
    chapter_range: Tuple[float, float],
    references: Optional[List[Tuple[str, int]]] = None,
    db_path: str = r"H:\Youtube",
):
    """Run title generation with parameters"""
    api = get_manga_api(db=db_path)
    sanitized_title = Utils.sanitize(manga_title)
    chapter_filter = (
        lambda chapter_num, **kwargs: chapter_range[0]
        <= float(chapter_num)
        <= chapter_range[1]
    )
    manga = api.get_manga(
        manga_title=sanitized_title,
        chapter_range=chapter_range,
        chapter_filter=chapter_filter,
    )
    chapters = []
    if not manga.chapters:
        api.hydrate_manga(manga=manga)
    for chapter in manga.chapters:
        if not chapter_range[0] <= float(chapter.ord) <= chapter_range[-1]:
            continue
        api.hydrate_chapter(chapter=chapter)
        chapters.append(chapter)
    aux = MangaAuxilary(api=api, manga=sanitized_title)
    title = aux.construct_title(chapters=chapters, references=references)
    print("Generated Title: ", title)


def run_mangacon_with_params(
    title: str,
    chapter_range: Tuple[float, float],
    provider: str = "mangahere",
    manga_id: Optional[str] = None,
    db_path: str = r"H:\Youtube",
):
    """Run mangacon with parameters"""
    chapter_filter = (
        lambda chapter_num, **kwargs: chapter_range[0]
        <= float(chapter_num)
        <= chapter_range[1]
    )
    mangacon_api = get_mangacon_api()
    db_api = get_manga_api(db=db_path)
    manga = mangacon_api.get_manga(
        title=title,
        id=manga_id,
        chapter_filter=chapter_filter,
        provider=provider,
    )
    print("MANGA: ", manga.to_dict())
    for chpt in manga.chapters:
        print("Chapter: ", chpt.ord, " ", chpt.to_dict())
        mangacon_api.get_pages(chapter=chpt, provider=provider)
    db_api.save_manga(manga=manga, force=True)


def run_magi_segmenter_with_params(
    manga_title: str,
    chapter_range: Tuple[float, float],
    manga_id: Optional[str] = None,
    batch_size: int = 3,
    db_path: str = r"H:\Youtube",
    weights_path: str = r"D:\Models\magiv2\pytorch_model.bin",
):
    """Run magi segmenter with parameters"""
    api = get_manga_api(db=db_path)
    magi = get_magi(api=api, weights_path=weights_path)

    # Use manga_id if provided, otherwise use manga_title
    if manga_id:
        manga = api.hydrate_manga(
            api.get_manga(manga_id=manga_id, chapter_range=chapter_range), from_db=True
        )
    else:
        manga = api.hydrate_manga(
            api.get_manga(manga_title=manga_title, chapter_range=chapter_range),
            from_db=True,
        )

    for chpt in manga.chapters:
        print(f"Hydrating Chapter: {chpt.ord}...\n")
        if not api.hydrate_chapter(chapter=chpt).pages:
            chapter = api.hydrate_chapter(chapter=chpt, from_db=False)
            api.save_chapter(chapter=chapter, depth=2)

    for chapter in manga.chapters:
        if (
            Utils.ifloat(chapter.ord) >= chapter_range[0]
            and Utils.ifloat(chapter.ord) <= chapter_range[-1]
        ):
            print(f"Segmenting chapter: {chapter.ord}")
            for i in range(0, len(chapter.pages), batch_size):
                batch_pages = chapter.pages[i : i + batch_size]
                magi.segment_pages(pages=batch_pages, characters=None)


def orchestrator():
    """Main orchestrator function that handles command-line arguments and routes to operations"""
    parser = argparse.ArgumentParser(
        description="Manga Processing Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s thumbnail --manga-title "Example Manga" --chapter-range "1-5"
  %(prog)s title --manga-title "Example Manga" --chapter-range "1-5" --references references.json
  %(prog)s mangacon --title "Example Manga" --chapter-range "1-10" --provider mangahere --manga-id "12345"
  %(prog)s composer --manga-title "Example Manga" --chapter-range "1-5" --voice-name "bm_daniel" --manga-id "67890"
  %(prog)s segment --manga-title "Example Manga" --chapter-range "1-5" --batch-size 5 --manga-id "67890"
        """,
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="operation", help="Available operations")

    # Common arguments
    def add_common_args(subparser):
        subparser.add_argument(
            "--db-path",
            default=r"H:\Youtube",
            help="Database path (default: H:\\Youtube)",
        )
        subparser.add_argument(
            "--chapter-range",
            required=True,
            type=str,
            help='Chapter range in format "start-end" or "start,end"',
        )

    # Thumbnail selector
    thumb_parser = subparsers.add_parser("thumbnail", help="Run thumbnail selector")
    thumb_parser.add_argument("--manga-title", required=True, help="Manga title")
    add_common_args(thumb_parser)

    # Title generation
    title_parser = subparsers.add_parser("title", help="Run title generation")
    title_parser.add_argument("--manga-title", required=True, help="Manga title")
    title_parser.add_argument(
        "--references", help="Optional references as JSON string or path to JSON file"
    )
    add_common_args(title_parser)

    # Mangacon
    mangacon_parser = subparsers.add_parser("mangacon", help="Run mangacon")
    mangacon_parser.add_argument("--title", required=True, help="Manga title")
    mangacon_parser.add_argument("--manga-id", help="Optional manga ID")
    mangacon_parser.add_argument(
        "--provider", default="mangahere", help="Manga provider"
    )
    add_common_args(mangacon_parser)

    # Composer
    composer_parser = subparsers.add_parser("composer", help="Run composer")
    composer_parser.add_argument("--manga-title", required=True, help="Manga title")
    composer_parser.add_argument("--manga-id", help="Optional manga ID")
    composer_parser.add_argument(
        "--narration-ex-path",
        default=r"H:\Youtube\narration_ex.txt",
        help="Narration example file path",
    )
    composer_parser.add_argument(
        "--output-dir", help="Output directory (auto-generated if not provided)"
    )
    composer_parser.add_argument(
        "--bg-filepath",
        default=r"H:\Youtube\Manga\bgm.wav",
        help="Background music file path",
    )
    composer_parser.add_argument(
        "--weights-path",
        default=r"D:\Models\magiv2\pytorch_model.bin",
        help="Model weights path",
    )
    composer_parser.add_argument(
        "--whisper-path", default="large-v3", help="Whisper model path"
    )
    composer_parser.add_argument(
        "--kkro-path", default=r"D:\Models\TTS\Kokoro-82M", help="Kokoro TTS model path"
    )
    composer_parser.add_argument("--voice-name", default="bm_daniel", help="Voice name")
    composer_parser.add_argument(
        "--tts-speed", type=float, default=1.10, help="TTS speed"
    )
    composer_parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second"
    )
    composer_parser.add_argument(
        "--bg-volume", type=float, default=0.025, help="Background volume"
    )
    composer_parser.add_argument(
        "--na-volume", type=float, default=1.15, help="Narration volume"
    )
    composer_parser.add_argument(
        "--temperature", type=float, default=0.64, help="LLM temperature"
    )
    composer_parser.add_argument(
        "--batch-size", type=int, default=150, help="Processing batch size"
    )
    add_common_args(composer_parser)

    # Magi segmenter
    segment_parser = subparsers.add_parser("segment", help="Run magi segmenter")
    segment_parser.add_argument("--manga-title", required=True, help="Manga title")
    segment_parser.add_argument("--manga-id", help="Optional manga ID")
    segment_parser.add_argument(
        "--batch-size", type=int, default=3, help="Processing batch size"
    )
    segment_parser.add_argument(
        "--weights-path",
        default=r"D:\Models\magiv2\pytorch_model.bin",
        help="Model weights path",
    )
    add_common_args(segment_parser)

    # Parse arguments
    args = parser.parse_args()

    if not args.operation:
        parser.print_help()
        sys.exit(1)

    try:
        # Parse chapter range
        chapter_range = parse_chapter_range(args.chapter_range)

        # Route to appropriate function
        if args.operation == "thumbnail":
            run_thumbnail_selector_with_params(
                manga_title=args.manga_title,
                chapter_range=chapter_range,
                db_path=args.db_path,
            )

        elif args.operation == "title":
            references = None
            if args.references:
                references = parse_references(args.references)
            references = references or [
                (
                    "Hero Got NTR-ED By Demons So He Reincarnates As Dark Lord To Take Revenge",
                    441580,
                ),
                ("When The Prodigy With a GOD BODY Enters The Dungeon Academy", 218365),
                (
                    "ISEKAID For A Slow-Life But Got A Unique Water Magic That Is Too OP To Stay Still",
                    272000,
                ),
                (
                    "Lonely Boy Saves His Teacher From NTR And She Couldn't Hold Back",
                    197950,
                ),
                ("His Secret Skill Lets Him Steal Any Monster’s Ability.", 95000),
                # ("He Was HUMILIATED and Left to Die, But Reincarnated as a GOD!", 293142),
            ]
            run_title_generation_with_params(
                manga_title=args.manga_title,
                chapter_range=chapter_range,
                references=references,
                db_path=args.db_path,
            )

        elif args.operation == "mangacon":
            run_mangacon_with_params(
                title=args.title,
                chapter_range=chapter_range,
                provider=args.provider,
                manga_id=args.manga_id,
                db_path=args.db_path,
            )

        elif args.operation == "composer":
            run_composer_with_params(
                manga_title=args.manga_title,
                chapter_range=chapter_range,
                manga_id=args.manga_id,
                db_path=args.db_path,
                narration_ex_path=args.narration_ex_path,
                output_dir=args.output_dir,
                bg_filepath=args.bg_filepath,
                weights_path=args.weights_path,
                whisper_path=args.whisper_path,
                kkro_path=args.kkro_path,
                voice_name=args.voice_name,
                tts_speed=args.tts_speed,
                fps=args.fps,
                bg_volume=args.bg_volume,
                na_volume=args.na_volume,
                temperature=args.temperature,
                batch_size=args.batch_size,
            )

        elif args.operation == "segment":
            run_magi_segmenter_with_params(
                manga_title=args.manga_title,
                chapter_range=chapter_range,
                manga_id=args.manga_id,
                batch_size=args.batch_size,
                db_path=args.db_path,
                weights_path=args.weights_path,
            )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# Update main function to use orchestrator by default
def main():
    """Main entry point - can be used for manual testing or orchestrated execution"""
    import sys

    if len(sys.argv) > 1:
        # Command line arguments provided, use orchestrator
        orchestrator()
    else:
        # No arguments, run default behavior (manual testing)
        print("No command line arguments provided. Running default operations...")
        print("Use --help to see available commands.")

        # Checks
        # check_device()
        # check_cudnn()

        # Manual operations for testing
        # run_thumbnail_selector()
        # run_title_generation()
        # run_mangacon()
        run_composer()
        # run_magi_segmenter()


if __name__ == "__main__":
    main()
