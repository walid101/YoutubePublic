# Imports
import os
import srt
import logging
import collections
from PIL import Image
from pathlib import Path
from srt import Subtitle
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple, Union

# Models
from ..datamodels.models import Timestamp
from ..audio.MangaAudio import MangaAudio
from manga.src.models import (
    MangaDataModel,
    ChapterDataModel,
    PageDataModel,
    PanelDataModel,
)
from src.com.models.AudioSource import AudioSource
from src.com.models.VideoSource import VideoSource
from ozkai_tts.WHISPERSRTModel import WHISPERSRTModel
from ozkai_tts.AbstractTTSModel import AbstractTTSModel
from ozkai_tts.AbstractRemoteTTSModel import AbstractRemoteTTSModel

# Custom
from ..video.MangaVideo import MangaPageVideo
from ozkit.src.com.Utils import Utils
from ozkapi.src.com.api.MangaApi import MangaApi
from ..composer.MangaNarrator import MangaNarratorGoogle
from ..auxilary.MangaAuxilary import MangaAuxilary
from src.com.channels.manga.adapter.MangaBaseSegmenter import MangaBaseSegmenter


@dataclass
class MangaComposerConfig:
    """
    Manga Composer Configurations. \n

    @params: \n
    force_panel_coords: re-calculate panel coords on a page. \n
    force_panel_segments: re-segment panels based on page coords, even if panels exist. \n
    """

    # LLM specific
    temperature: float = 0.72

    # Video specific
    view_window: Tuple[int, int] = (1920, 1080)
    force_panel_coords: bool = False
    force_panel_segments: bool = False
    tts_speed: float = (
        0.90  # If its too fast, the subtitle generator has trouble, we can just speed it up at the end.
    )
    time_between: float = 0.35
    volume: float = 0.8
    bg_volume: float = 0.02
    fps: int = 30

    # Pre-processing specific
    ref_chapter_summaries: bool = True

    # Post-processing specific
    bg_filepath: Union[Path, str] = None


class MangaComposer:
    def __init__(
        self,
        manga: str,
        api: MangaApi,
        whisper: WHISPERSRTModel,
        tts: Union[AbstractTTSModel, AbstractRemoteTTSModel],
        segmenter: MangaBaseSegmenter,
        aux: MangaAuxilary,
        config: MangaComposerConfig,
        chapter_range: Tuple[float, float],
        chapter_filter: Optional[Callable] = None,
        output_dir: Union[Path, str] = r"./",
        **kwargs,
    ):
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        # Checks
        self.api = api
        self.tts = tts
        self.whisper = whisper
        self.audio = MangaAudio(tts=tts, whisper=whisper)
        self.manga_title = Utils.sanitize(manga)
        self.narrator = MangaNarratorGoogle(
            api=api, manga=manga, temperature=config.temperature
        )
        self.segmenter = segmenter
        self.aux = aux
        self.config = config
        self.chapter_range = chapter_range
        self.chapter_filter = chapter_filter
        self.output_dir = output_dir
        self.manga_id = kwargs.get("manga_id")
        self.manga_videofile = f"{self.manga_title}_{self.chapter_range[0]}_{self.chapter_range[-1]}_video.mp4"
        self.manga_audiofile = f"{self.manga_title}_{self.chapter_range[0]}_{self.chapter_range[-1]}_audio.mp3"
        self.manga_srtfile = f"{self.manga_title}_{self.chapter_range[0]}_{self.chapter_range[-1]}_srt.srt"
        self.manga_aligned_srtfile = f"{self.manga_title}_{self.chapter_range[0]}_{self.chapter_range[-1]}_aligned_srt.srt"

    def construct(
        self,
        narration_ex: Union[str, Path] = None,
        batch_size: int = 10,
    ) -> str:
        """
        Takes all required raw input, outputs video, Chapters processed batch-wise (panels of 30).
        Notes
        -----
        Panel narrations are saved in-place
        """
        # Narration
        manga, pages = self.construct_narration(
            chapter_range=self.chapter_range,
            chapter_filter=self.chapter_filter,
            narration_ex=narration_ex,
            batch_size=batch_size,
        )  # populated page narrations.
        pages = [
            page
            for page in pages
            if (
                page.panels
                and page.meta.get("summary")
                and page.meta.get("summary") not in ("N/A", None)
            )
        ]
        # Panel Mapping
        panel_mapping = []  # To track which page/panel each dialogue belongs to
        panels_narrations = []
        for page in pages:
            page_num = page.ord
            for panel in page.panels:
                dialogue = panel.meta.get("summary")
                if dialogue not in ("N/A", None):
                    panels_narrations.append(dialogue)
                    panel_mapping.append(f"{page.path[-1]}_{page_num}_{panel.ord}")

        # Srt
        audio, srt = self.construct_audio(dialogue=panels_narrations)

        # Timestamps
        self.logger.info("Constructing Timestamps...")
        timestamps: Dict[str, Timestamp] = {}
        raw_timestamps = self.audio.parse_srt(srt_path=srt)
        for i, key in enumerate(panel_mapping):
            if i < len(raw_timestamps):
                timestamps[key] = raw_timestamps[i]

        Utils.pause(
            prompt="\nFinished constructing timestamps. \nDo you want to continue to constructing video?",
            options=["y", "e"],
        )

        video_filepath = os.path.join(self.output_dir, self.manga_videofile)
        if not os.path.exists(path=video_filepath):
            video_filepath = self.construct_video(pages=pages, timestamps=timestamps)
        else:
            self.logger.info("Video already constructed. Continuing...")

        response = Utils.pause(
            prompt="\nDo you want to add narration to video?", options=["y", "n", "e"]
        )

        video_filepath_narrated = os.path.join(
            self.output_dir, "narrated", self.manga_videofile
        )
        video_filepath_final = os.path.join(
            self.output_dir, "final", self.manga_videofile
        )
        if response == "y":
            video_filepath = self.merge(
                video=video_filepath,
                audio=audio,
                volume=self.config.volume,
                output_path=video_filepath_narrated,
            )

        # response = Utils.pause(
        #     prompt=f"\nDo you want to add background music?\n{self.config.bg_filepath}",
        #     options=["y", "n", "e"],
        # )

        # if response == "y":
        #     if not os.path.exists(video_filepath_final):
        #         video_filepath_final = self.merge(
        #             video=video_filepath_narrated,  # narrated path only.
        #             audio=self.config.bg_filepath,
        #             volume=self.config.bg_volume,
        #             output_path=video_filepath_final,
        #         )
        return video_filepath

    def construct_video(
        self,
        pages: List[PageDataModel],
        timestamps: Dict[str, Timestamp],
    ) -> str:
        """
        Given a list of PIL panel images (fully processed), subtitles (Aligned), construct video.
        """
        # Filter pages
        video_filepath = os.path.join(self.output_dir, self.manga_videofile)
        if not os.path.exists(video_filepath):
            MangaPageVideo(
                pages=pages,
                timestamps=timestamps,
                fps=self.config.fps,
                max_workers=8,
                view_window=self.config.view_window,
                cache_limit_multiplier=0.5,
                use_fp16=True,
            ).construct(
                output_path=os.path.join(self.output_dir, self.manga_videofile),
            )  # Saves video, returns filepath
        return video_filepath

    def construct_audio(self, dialogue: List[str]) -> Tuple[str, str]:
        # Construct TTS
        subtitles: List[srt.Subtitle] = None
        audio_filepath = os.path.join(self.output_dir, self.manga_audiofile)
        srt_filepath = os.path.join(self.output_dir, self.manga_srtfile)
        srt_aligned_filepath = os.path.join(self.output_dir, self.manga_aligned_srtfile)
        if not os.path.exists(audio_filepath):
            # Clean dialogue.
            subtitles = self.audio.construct(
                dialogue=dialogue,
                filepath=audio_filepath,
                speed=self.config.tts_speed,
                time_between=self.config.time_between,
            )

        if subtitles:
            Utils.pause(prompt="\nConstructed TTS and SRT.\n", options=("y", "e"))
            # Save subtitles at aligned filepath and normal filepath
            self.audio.convert_subtitles_to_srt(
                subtitles=subtitles,
                save_path=srt_aligned_filepath,
            )
            self.audio.convert_subtitles_to_srt(
                subtitles=subtitles,
                save_path=srt_filepath,
            )
            return audio_filepath, srt_aligned_filepath

        Utils.pause(
            prompt=(
                "\nConstructed TTS.\n",
                "Generating SRT, would you like to continue? (y/n/e): \n",
            )
        )
        # Construct SRT
        if not os.path.exists(srt_filepath):
            self.audio.get_subtitles_by_word(
                audio_path=audio_filepath,
                output_path=srt_filepath,
                ref_text=" ".join(dialogue),
            )
        Utils.pause(
            prompt=(
                "\nAligning SRT.\n",
                "would you like to continue? (y/n/e): \n",
            )
        )
        if not os.path.exists(srt_aligned_filepath):
            # Align subtitles
            subtitles: List[Subtitle] = self.audio.align_srt_by_reference(
                subtitle=srt_filepath, reference=dialogue
            )
            self.audio.convert_subtitles_to_srt(
                subtitles=subtitles,
                save_path=srt_aligned_filepath,
            )  # List of subtitles

        Utils.pause(
            prompt=(
                "\nConstructed aligned SRT.\n",
                "Generating video, would you like to continue? (y/n/e): \n",
            )
        )
        return audio_filepath, srt_aligned_filepath

    def construct_narration(
        self,
        chapter_range: Tuple[float, float],
        chapter_filter: Optional[Callable] = None,
        narration_ex: Union[str, Path] = None,
        batch_size: int = 10,
    ) -> Tuple[MangaDataModel, List[PageDataModel]]:
        """
        Generates narration for manga pages, processing in batches and handling context.
        """
        Utils.pause(
            prompt=(
                "\nPreprocessing and Hydrating Pages \n",
                "Would you like to continue? (y/n/e): \n",
            )
        )
        all_pages, manga = self.preprocess_and_hydrate_pages(
            chapter_range=chapter_range, chapter_filter=chapter_filter
        )

        all_pages = sorted(all_pages, key=lambda page: (float(page.path[-1]), page.ord))

        if not all_pages:
            self.logger.warning(
                "No pages found or hydrated for the given chapter range."
            )
            return [], []

        # Display information and ask for confirmation
        self.logger.info(f"Manga: {manga.title}")
        self.logger.info(f"Chapters: {chapter_range[0]}-{chapter_range[1]}")
        self.logger.info(f"Pages loaded: {len(all_pages)}")
        self.logger.info(
            f"Total panels detected: {sum(len(page.panels) for page in all_pages)}"
        )

        Utils.pause(
            prompt=(
                "\nFinished loading chapter, pages, and panels (segmented). \n",
                "Starting narrations, would you like to continue? (y/n/e): \n",
            )
        )
        self.logger.info(f"Starting narration generation for {len(all_pages)} pages...")

        chapters_map = {float(chapter.ord): chapter for chapter in manga.chapters}
        all_narrations: List[str] = []
        processed_chapters: set[ChapterDataModel] = set()
        recent_summary_context: Deque[str] = collections.deque(maxlen=batch_size)

        pages_by_chapter = {}
        for page in all_pages:
            chapter_num = float(page.path[-1])
            if chapter_num not in pages_by_chapter:
                pages_by_chapter[chapter_num] = []
            pages_by_chapter[chapter_num].append(page)

        all_batches = []
        for chapter_num in sorted(pages_by_chapter.keys()):
            chapter_pages = pages_by_chapter[chapter_num]

            # If the entire chapter can fit in a single batch, process it as one unit
            if len(chapter_pages) <= batch_size:
                all_batches.append(chapter_pages)
            else:
                for i in range(0, len(chapter_pages), batch_size):
                    all_batches.append(chapter_pages[i : i + batch_size])

        # Process each batch
        for current_batch_pages in all_batches:
            current_chapter_ord = float(current_batch_pages[0].path[-1])
            batch_requires_summarization = False
            for page in current_batch_pages:
                if not page.panels:
                    self.api.hydrate_page(page=page)

                if not page.meta.get("summary"):
                    batch_requires_summarization = True
                    self.logger.info(f"PAGE DOESNT HAVE SUMMARY: Page {page.ord}")
                    break

                for panel in page.panels:
                    summary = panel.meta.get("summary")
                    if not summary:
                        self.logger.info(
                            f"PANEL IN PAGE DOESNT HAVE SUMMARY: Page {page.ord}, Panel {panel.ord}"
                        )
                        batch_requires_summarization = True
                        break
                if batch_requires_summarization:
                    break

            # Prepare Context from Previous Batches
            page_context_str = None
            if recent_summary_context:
                context_prompt = (
                    "Here are the narrations of the PREVIOUS pages, for context."
                )
                context_content = " ".join(recent_summary_context)
                if context_content.strip():  # Ensure context isn't just whitespace
                    page_context_str = f"{context_prompt}\n\n{context_content}"

            chapter_context_str = self.construct_chapter_summaries(
                chapters=list(processed_chapters), max_ord=current_chapter_ord
            )
            if batch_requires_summarization:
                context = []
                if page_context_str:
                    context.append(page_context_str)
                batch_index = all_batches.index(current_batch_pages)
                self.logger.info(f"Batch {batch_index} requires summarization.")
                if page_context_str:
                    self.logger.debug(
                        f"Providing context from previous {len(recent_summary_context)} valid summaries."
                    )

                try:
                    if chapters_map.get(current_chapter_ord):
                        processed_chapters.add(chapters_map.get(current_chapter_ord))
                    self.narrator.summarize_chapters(chapters=list(processed_chapters))
                    self.logger.info(
                        f"Summarization for chapters call successful for batch {batch_index}."
                    )

                    # Pass the *entire contiguous batch to maintain sequence for the summarizer
                    self.narrator.summarize_pages(
                        pages=current_batch_pages,
                        characters=manga.characters,
                        context=context,
                        batch_size=batch_size,
                    )
                    self.logger.info(
                        f"Summarization for pages call successful for batch {batch_index}."
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error during summarization call for batch {batch_index}: {e}",
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Failed to summarize batch {batch_index}: {e}"
                    ) from e
            else:
                batch_index = all_batches.index(current_batch_pages)
                self.logger.info(
                    f"Skipping summarization for batch {batch_index}, all pages/panels have valid summaries."
                )

            # Collect Results & Update Context for Next Iteration
            current_batch_valid_summaries = []
            for page in current_batch_pages:
                if self.__valid_narration(p=page):
                    current_batch_valid_summaries.append(page.meta.get("summary"))

            all_narrations.extend(current_batch_valid_summaries)

            # Update the context deque: Use summaries from *this* batch for the *next* batch's context
            recent_summary_context.clear()
            recent_summary_context.extend(current_batch_valid_summaries)

            batch_index = all_batches.index(current_batch_pages)
            self.logger.debug(
                f"Processed batch {batch_index}. "
                f"Collected {len(current_batch_valid_summaries)} valid summaries this batch. "
                f"Total narrations: {len(all_narrations)}."
            )

        Utils.pause(
            prompt=(
                "\nFinished building narrations. \n",
                f"Generating TTS ({type(self.tts)}), would you like to continue? (y/n/e): \n",
            )
        )
        return manga, all_pages

    def preprocess_narration(
        self,
        chapter_range: Tuple[float, float],
        chapter_filter: Optional[Callable] = None,
    ) -> MangaDataModel:
        """
        Hydrates manga chapters, pages, and characters.

        @params:
        chapter_range: Tuple[float, float]: Chapter range to hydrate.
        """
        manga = self.api.get_manga(
            manga_title=self.manga_title,
            manga_id=self.manga_id,
            chapter_range=chapter_range,
            chapter_filter=chapter_filter,
        )
        if not manga.chapters:
            Utils.pause("Could not find chapters, hydrating...")
            self.api.hydrate_manga(manga=manga, chapter_range=chapter_range)

        present_chapters = {float(ch.ord) for ch in manga.chapters if ch.ord} or [-1]
        if max(present_chapters) < chapter_range[-1]:
            max_present = max(present_chapters)
            max_present = max_present if max_present > 0 else chapter_range[0] - 1
            missing_range = f"{max_present}-{chapter_range[-1]}"
            Utils.pause(f"Missing chapters {missing_range}, hydrating remotely...")
            self.api.hydrate_manga(
                manga=manga, from_db=False, chapter_range=chapter_range
            )

        Utils.pause(f"Hydrated {len(manga.chapters)} chapters. Continue?")
        if not manga.characters:
            self.logger.info("Hydrating Manga Characters...")
            if not self.api.hydrate_characters(manga=manga).characters:
                self.logger.info(
                    "Could not find characters in DB, hydrating from api..."
                )
                self.api.hydrate_characters(manga=manga, from_db=False)
            if not manga.characters:
                Utils.pause(
                    f"Unable to hydrate characters for Manga\n{manga.title}, continue?"
                )
            else:
                Utils.pause(f"Found characters for Manga\n{manga.title}, continue?")

        for chapter in manga.chapters:
            self.logger.info("Hydrating Manga Chapters...")
            hydrated_chapter = self.api.hydrate_chapter(chapter=chapter)
            if not hydrated_chapter.pages:
                self.api.hydrate_chapter(chapter=chapter, from_db=False)

        # Save manga - let the save function handle checking if things exist
        self.logger.info("Saving manga to database")
        self.api.save_manga(manga=manga, force=True)
        return manga

    def construct_context(self, context: Union[Path, str]) -> str:
        output = None
        if context:
            narration_content = None
            if isinstance(context, (str, Path)):
                try:
                    with open(context, "r") as f:
                        narration_content = f.read()
                except (FileNotFoundError, IOError, OSError):
                    narration_content = context

            if narration_content:
                output = (
                    "I will be giving you a speech sample I made for a manga completely different from the one I gave you. "
                    "I want you to deeply analyze how I talk in this speech sample and do your best to imitate it with your own narration. "
                    "Notice how I keep the flow between panels, making it as seamless as possible. "
                    "Here is the speech sample: \n\n" + narration_content + "\n\n"
                )
        return output

    def construct_panels(
        self,
        pages: List[PageDataModel],
        characters: Optional[Dict[str, List]],
        batch: int = 10,
    ) -> List[PanelDataModel]:
        """
        Given a list of PageDataModel, segments and saves a list of ordered PanelDataModels.
        @params
        pages: List of PageDataModel models.
        batch: Size of batches to process pages for segmentation (default: 10)
        """
        for page in pages:
            self.api.hydrate_page(page=page)
        pages = sorted(pages, key=lambda page: (float(page.path[-1]), page.ord))
        constructed_panels = []
        filtered_pages: List[PageDataModel] = [
            page
            for page in pages
            if (
                not (page.panels and page.meta.get("magi"))  # If neither of these exist
                or self.config.force_panel_coords  # Or if we want to force panels
            )
        ]

        if not filtered_pages:
            return []

        if self.config.force_panel_segments:
            pass

        all_processed_pages = []
        for i in range(0, len(filtered_pages), batch):
            batch_pages = filtered_pages[i : i + batch]
            print(
                f"Processing batch {i//batch + 1}/{(len(filtered_pages) + batch - 1)//batch}: {len(batch_pages)} pages"
            )

            # Process and segment current batch of pages
            processed_batch: List[List[Image.Image]] = self.segmenter.segment_pages(
                pages=batch_pages,
                characters=characters,
                force=self.config.force_panel_coords,
            )
            all_processed_pages.extend(processed_batch)

        # Create panels from all processed pages
        for page, page_panel_images in zip(filtered_pages, all_processed_pages):
            panels = []
            panel_dir = self.get_panel_path(page=page)
            for idx, panel_image in enumerate(page_panel_images):
                panel_image_path = os.path.join(
                    panel_dir, f"panel_{page.ord}_{idx}.jpg"
                )
                panel_image.save(panel_image_path)

                panel = PanelDataModel(
                    page=page.ord,
                    ord=idx,
                    image_urls=[panel_image_path],
                    path=page.path,
                )
                panels.append(panel)
                self.api.save_panel(panel=panel)
            page.panels = panels
            constructed_panels.extend(panels)  # Ordered.
        return constructed_panels

    """Helper Methods"""

    def construct_chapter_summaries(
        self, chapters: List[ChapterDataModel], max_ord: int
    ):
        """
        Concatenate all chapter summaries from [1-max_ord] included.
        """
        init_prompt = "\nHere are the PREVIOUS CHAPTER summaries: \n"
        chapters_filtered = sorted(
            [chapter for chapter in chapters if 1 <= float(chapter.ord) <= max_ord],
            key=lambda chapter: chapter.ord,
        )
        chapter_summaries = []
        for chapter in chapters_filtered:
            summary = chapter.meta.get("summary")
            if summary:
                chapter_summaries.append(summary)
        return init_prompt + "\n".join(chapter_summaries)

    @staticmethod
    def construct_timestamps(srt: Union[Path, str], ref: List[str]) -> List[Timestamp]:
        """
        Given a subtitle file "srt" and a list of strings "ref", align the subtitle timestamps so that timestamps are partitioned on
        whole sentences per ref entry.

        Args:
            srt: Path to subtitle file.
            ref: List of strings (sentences) that will align subtitle timestamps.

        Returns:
            List[Timestamp]: List of timestamp objects aligned with reference sentences.
        """
        merged = MangaAudio.align_srt_by_reference(
            subtitle=srt,
            reference=ref,
        )

        timestamps = []
        for subtitle in merged:
            timestamp = Timestamp(
                start=subtitle.start.total_seconds(),
                end=subtitle.end.total_seconds(),
                duration=subtitle.end.total_seconds() - subtitle.start.total_seconds(),
            )
            timestamps.append(timestamp)

        return timestamps

    def concat_panel_summaries(self, panels: List[PanelDataModel]) -> str:
        return " ".join(
            panel.meta.get("summary", "")
            for panel in panels
            if self.__valid_narration(panel=panel)
        )

    def concat_page_summaries(self, pages: List[PageDataModel]) -> str:
        return " ".join(
            page.meta.get("summary", "")
            for page in pages
            if self.__valid_narration(page=page)
        )

    def __valid_narration(self, p: Union[PanelDataModel, PageDataModel]):
        return p.meta and p.meta.get("summary") not in [None, "N/A"]

    def get_panel_path(self, page: PageDataModel):
        if self.api.local_db:
            page_image_filepath = page.get_image_filepath(page.image_urls)
            if page_image_filepath:
                return os.path.dirname(page_image_filepath)
            else:
                self.logger.error(
                    f"Could not find a page filepath in given page: {page.to_dict()}"
                )
                return None
        else:
            # TODO - Implement remote features.
            return None

    def _is_page_summarized(self, page: PageDataModel) -> bool:
        """
        Checks if both the page itself and all its panels have summaries.
        Considers a page with no panels fully summarized if the page summary exists.
        """
        page_summary_exists = bool(page.meta.get("summary"))
        if not page.panels:
            # If no panels, only the page summary matters
            return page_summary_exists

        panels_summarized = all(panel.meta.get("summary") for panel in page.panels)
        return page_summary_exists and panels_summarized

    def preprocess_and_hydrate_pages(
        self,
        chapter_range: Tuple[float, float],
        chapter_filter: Optional[Callable] = None,
    ) -> Tuple[List[PageDataModel], MangaDataModel]:
        manga = self.preprocess_narration(
            chapter_range=chapter_range, chapter_filter=chapter_filter
        )
        all_pages_ordered: List[PageDataModel] = []
        start_chapter, end_chapter = chapter_range

        chapters_to_process: List[ChapterDataModel] = []
        for chapter in manga.chapters:
            try:
                self.logger.info(f"THE CHAPTER ORD: {chapter.ord}")
                if start_chapter <= float(chapter.ord) <= end_chapter:
                    chapters_to_process.append(chapter)
            except (ValueError, TypeError, AttributeError):
                self.logger.warning(
                    f"Skipping chapter with invalid ord for range check: {getattr(chapter, 'ord', 'N/A')}"
                )
                pass  # Skip chapters with ordinals that cannot be converted to int

        if not chapters_to_process:
            self.logger.warning(
                f"No valid chapters found within range {chapter_range}."
            )
            return (
                [],
                manga,
            )  # Return empty pages, but potentially populated manga object

        # Sort the chapters confirmed to be in range by their integer order
        sorted_chapters = sorted(chapters_to_process, key=lambda ch: float(ch.ord))
        self.logger.info(
            f"Processing {len(sorted_chapters)} chapters in range {chapter_range}."
        )
        for chapter in sorted_chapters:
            if chapter.pages:
                self.construct_panels(pages=chapter.pages, characters=None)
                try:
                    # Sort pages within the chapter by integer order
                    sorted_pages = sorted(chapter.pages, key=lambda p: float(p.ord))
                    all_pages_ordered.extend(sorted_pages)
                except (ValueError, TypeError, AttributeError):
                    self.logger.error(
                        f"Failed to sort pages for Chapter {chapter.ord}, adding unsorted.",
                        exc_info=True,
                    )
                    all_pages_ordered.extend(chapter.pages)  # Fallback
            else:
                self.logger.warning(
                    f"Chapter {chapter.ord} (in range) has no pages after hydration."
                )
        self.segmenter.offload_model()  # Offload Segmenter
        return all_pages_ordered, manga

    def merge(
        self,
        video: Union[Path, str],
        audio: Union[Path, str],
        volume: float = 0.8,
        start_time: float = 0,
        end_time: float = -1,
        output_path: Union[Path, str] = None,
    ) -> str:
        """
        Merges video, audio data into one.
        """
        if output_path is not None:
            output_path_str = (
                str(output_path) if isinstance(output_path, Path) else output_path
            )

            output_dir = os.path.dirname(output_path_str)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

        return AudioSource.overlay(
            media_source=video,
            audio_source=audio,
            volume=volume,
            output_path=output_path,
            start_time=start_time,
            end_time=max(end_time, 99999.0),
        )
