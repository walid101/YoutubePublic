import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Union
from manga.src.models import ChapterDataModel, PageDataModel
from ozkapi.src.com.api.MangaApi import MangaApi
from transformers import PreTrainedModel, AutoModel
from ozkai_segment.MangaSegmenterMagi import MangaSegmenterMagi
from src.com.channels.manga.adapter.MangaBaseSegmenter import MangaBaseSegmenter
from manga.src.models import PageDataModel, CharacterDataModel


class MangaMagiSegmenterAdapter(MangaBaseSegmenter, MangaSegmenterMagi):
    def __init__(
        self,
        api: MangaApi,
        weights: Union[Path, str],
        device: str = None,
    ):
        # Call both parent class initializers correctly
        MangaBaseSegmenter.__init__(self, api=api)
        MangaSegmenterMagi.__init__(self, weights=weights, device=device)

    def segment_pages(
        self,
        pages: List[PageDataModel],
        characters: Union[List[ChapterDataModel], Dict[str, List]] = None,
        force=False,
    ) -> List[List[Image.Image]]:
        unprocessed_pages = [
            cv2.cvtColor(
                self.pad_image(page.fetch_page_image(), x_pad=20, y_pad=20),
                cv2.COLOR_BGR2RGB,
            )
            for page in pages
            if "magi" not in page.meta.keys() or force
        ]

        if isinstance(characters, List):
            characters = self.construct_character_dict(characters=characters)

        if unprocessed_pages:
            batch_size = 50
            all_results = []

            if len(unprocessed_pages) > batch_size:
                for i in range(0, len(unprocessed_pages), batch_size):
                    current_batch = unprocessed_pages[i : i + batch_size]
                    batch_results = self.predict(
                        pages=current_batch, characters=characters, do_ocr=True
                    )
                    all_results.extend(batch_results)
            else:
                all_results = self.predict(
                    pages=unprocessed_pages, characters=characters, do_ocr=True
                )

            # Only update pages that were actually processed
            processed_indices = [
                i for i, page in enumerate(pages) if "magi" not in page.meta or force
            ]
            for result, idx in zip(all_results, processed_indices):
                page = pages[idx]
                page.meta["magi"] = result
                self.api.save_page(page=page, force=True)

        panels: List[Image.Image] = []
        for page in pages:
            panels.append(self.segment_page(page=page))

        unprocessed_pages.clear()

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info(
                    "Successfully cleared cache after segmenting page batch."
                )
        except ImportError:
            self.logger.info(
                "Page segmenting finished but could not clear frames from GPU."
            )
            pass

        return panels

    def segment_page(self, page: PageDataModel) -> List[Image.Image]:
        """
        Assuming the given page has magi saved to meta. Returns segmented panel images.

        @params:
        page: PageDataModel to segment.
        """
        # Checks
        if not (page.meta.get("magi", {}).get("panels")):
            self.logger.error("Magi must be provided in page metadata.")
            return []

        panels = []
        nested_panel_coords = page.meta.get("magi").get("panels")
        for coords in nested_panel_coords:
            raw_image = self.segment_panel(image=page.fetch_page_image(), coords=coords)
            panels.append(self.convert_cv_to_pil(raw_image))

        return panels

    def visualize_page(self, page: PageDataModel):
        # Checks
        page_magi = page.meta.get("magi")
        if not page_magi:
            self.logger.error(
                f"Cannot visualize page if magi metadata not present, for page: {page.to_dict()}"
            )

        # Visualize
        processed_page = self.model.visualise_single_image_prediction(
            image_as_np_array=page.fetch_page_image(), predictions=page_magi
        )

        return processed_page

    """Helpers"""

    def construct_character_dict(self, characters: List[CharacterDataModel]):
        """
        Constructs a dictionary containing lists of character image URLs and names.

        Args:
            characters: List of CharacterDataModel objects

        Returns:
            Dict with two keys: 'images' containing image URLs and 'names' containing character names
        """
        character_bank = {"images": [], "names": []}

        for character in characters:
            # Try local path first
            local_image_path = os.path.join(
                self.api.local_db,
                self.api.root_path,
                *character.path,
                f"{character.filename}.jpg",
            )
            # Add the character images
            if os.path.exists(local_image_path):
                character_bank["images"].append(local_image_path)
            else:
                character_bank["images"].append(character.image_url)
            # Add the character name to names list
            character_bank["names"].append(character.name)
        return character_bank
