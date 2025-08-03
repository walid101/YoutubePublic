from dataclasses import dataclass
import os
import re
import json
import re
import traceback
from rapidfuzz import fuzz
from typing import Dict, List, Optional
from PIL import Image
from google.genai import types
from typing import List
from ozkai_nlp.llm.Google import GoogleLLM
from manga.src.models import (
    ChapterDataModel,
    PageDataModel,
    PanelDataModel,
    CharacterDataModel,
    ImageModel,
)
from ozkapi.src.com.api.MangaApi import MangaApi
from ozkit.src.com.Utils import Utils
from ozkit.src.com.llm.json.LLMJsonParser import LLMJsonParser


# Constants
@dataclass(frozen=True)
class ChapterNarratorConstants:
    ORD: str = "chapter_number"
    SUMMARY_LABEL: str = "chapter_summary"


@dataclass(frozen=True)
class PageNarratorConstants:
    NARRATION_LABEL: str = "page_narration"
    ORD: str = "page_number"
    PANELS: str = "panel_narrations"


@dataclass(frozen=True)
class PanelNarratorConstants:
    NARRATION_LABEL: str = "panel_narration"
    ORD: str = "panel_number"


class MangaNarratorGoogle(GoogleLLM):
    def __init__(
        self,
        api: MangaApi,
        manga: str,
        models=[
            "models/gemini-2.5-pro-preview-06-05",
            "models/gemini-2.5-pro-preview-05-06",
            "models/gemini-2.5-pro-preview-03-25",
        ],
        temperature=0.72,
    ):
        self.api = api
        self.manga = manga
        self.google_models = models
        self.temperature = temperature
        self.json_parser = LLMJsonParser()
        super().__init__(models)

    """ Character """

    def search_character(
        self,
        character: str,
        init: List[CharacterDataModel],
        confidence=80,
        max_length=129,
    ) -> Optional[CharacterDataModel]:
        """Search for character name, expanding and preserving search space if not found."""
        self.logger.info(f"Searching for character: {character}")
        limit = max(len(init), 15)

        # Convert character to lowercase for case-insensitive comparison
        char_lower = character.lower()

        while limit <= max_length:
            for item in init:
                item_lower = item.name.lower()
                if char_lower in item_lower:
                    return item

                # Try partial_ratio which is good for substring fuzzy matching
                if fuzz.partial_ratio(char_lower, item_lower) >= confidence:
                    return item

            limit *= 2
            new_chars = self.api.manga_api.get_characters(
                manga_title=self.manga, limit=limit
            )
            if not new_chars or len(new_chars) <= len(init):
                break
            init[:] = new_chars

        return None

    def extract_characters_hydrated(
        self, characters: List[str] = None
    ) -> Dict[str, Image.Image]:
        """
        Given a list of unique character names, map them to their correct images
        """
        pass

    def extract_characters_unhydrated(
        self, panels: List[PanelDataModel], characters: List[str] = None
    ) -> List[str]:
        """
        Extract character names from panels using LLM identification without hydrating with images.
        Modifies the input characters list in place and returns it.

        @params
        panels: List of panels to find characters from.
        characters: Initial list of characters to add to. Modified in place. Default is empty list.

        @returns
        List[str]: The modified list of unique character names found in the panels.
        """
        if characters is None:
            characters = []

        panels = sorted(panels, key=lambda p: p.ord)
        character_names = set()  # Temporary set to gather new names from panels
        unprocessed_panels: List[PanelDataModel] = []

        for panel in panels:
            if chars := panel.meta.get("characters", []):
                character_names.update(chars)
            else:
                unprocessed_panels.append(panel)

        if unprocessed_panels:
            contents = [
                "I will be giving you a list of panel images. "
                "I want you to give me a list of unique character names that are relevant/required in these panels. "
                "The output should be a comma separated list of names. "
            ]
            contents.extend(panel.fetch_panel_image() for panel in unprocessed_panels)
            self.logger.info(f"model name: {self.google_model}")
            response = self.generate_content_with_models(
                models=self.google_models,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction="You are the leading expert in identifying important and relevant character names from a list of manga panel images.",
                    temperature=self.temperature,
                ),
            )

            if response.text:
                character_names.update(
                    name.strip() for name in response.text.split(",") if name.strip()
                )

        if not character_names:
            self.logger.info(
                f"No characters found in panels: {panels[0].ord} - {panels[-1].ord}"
            )
            return []
        result = characters.copy()

        for name in character_names:
            # Skip if similar character name already exists in our result
            if not any(
                fuzz.ratio(name.lower(), existing.lower()) >= 80 for existing in result
            ):
                result.append(name)
            else:
                self.logger.info(f"Skipping duplicate character name: {name}")

        return result

    def extract_characters(
        self, panels: List[PanelDataModel], characters: Dict[str, Image.Image] = {}
    ) -> Optional[Dict[str, Image.Image]]:
        """
        Extract and match character images from panels using LLM identification.
        @params
        panels: List of panels to find characters from.
        characters: Initial list of characters to add to. Modified in place - Typically comes from MangaDataModel object.
        """
        panels = sorted(panels, key=lambda p: p.ord)
        character_names = set()
        unprocessed_panels: List[PanelDataModel] = []
        for panel in panels:
            if chars := panel.meta.get("characters", []):
                character_names.update(chars)
            else:
                unprocessed_panels.append(panel)

        if unprocessed_panels:
            contents = [
                "I will be giving you a list of panel images.\n"
                "I want you to give me a list of unique character names that are relevant/required in these panels.\n"
                "The output should be a comma separated list of names."
            ]
            contents.extend(panel.fetch_panel_image() for panel in unprocessed_panels)
            self.logger.info(f"model name: {self.google_model}")
            response = self.generate_content_with_models(
                models=self.google_models,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction="You are the leading expert in identifying important and relevant character names from a list of manga panel images.",
                    temperature=0.65,
                ),
            )

            if response.text:
                character_names.update(
                    name.strip() for name in response.text.split(",") if name.strip()
                )

        if not character_names:
            self.logger.info(
                f"No characters found in panels: {panels[0].ord} - {panels[-1].ord}"
            )
            return None

        character_pool = []

        for name in character_names:
            if identity := self.search_character(
                character=name, init=character_pool, confidence=80
            ):
                # Skip if similar character name already exists
                if any(
                    fuzz.ratio(identity.name.lower(), existing.lower()) >= 80
                    for existing in characters
                ):
                    self.logger.info(f"Skipping duplicate character: {name}")
                    continue

                characters[name] = ImageModel.fetch_image_static(identity.image_url)
            else:
                self.logger.info(f"Character not found: {name}")

        return characters

    def __construct_char_prompt(self, characters: List[CharacterDataModel]) -> List:
        """
        Construct character prompt given a list of Character models.
        @params
        characters: List of character data models.

        @return
        string prompt for batch summarization.
        """
        if not characters:
            return []

        char_intro = (
            "I will be giving you a list of names, their role, and an image of each character. \n"
            "The order will be name, role, sex/gender and then an image \n."
            "I want you to use this information when narrating. \n"
            "NOTE: you will NOT be given every character in the story, just a sparse set of what is available. \n"
            "Before I give you the characters list, here is some DO's and DO NOT's: \n"
            "DO's: \n"
            "Map characters based on CONTEXT first before relying on the given images. \n"
            "Only map character names to characters in the panel if you are confident (87%+) that you are identifying the correct character. \n"
            "Use specific facial features, body features, to map as precisely as possible when identifying characters. \n"
            "DO NOT's: \n"
            "DO NOT force a matches, try your best to identify characters with some flexibility but it is worse if you get the character wrong. \n"
            "DO NOT get character mappings WRONG ever. It is better if you dont directly use names rather than reffering to characters with the WRONG name. \n"
            "Some characters WILL look alike, so you need to use context clues to make sure you do NOT use the wrong names. \n"
            "If the given image does match, and is reliable make sure to get the gender correct, as given here. \n"
        )

        char_prompt: List = [char_intro]

        for char in characters:
            curr_char_prompt = f"({char.name}, {char.role}, {char.sex or 'Unknown'})"
            char_prompt.append(curr_char_prompt)
            # Ideally this should be done in the model itself.
            char_path = os.path.join(
                self.api.local_db,
                self.api.root_path,
                *char.path,
                f"{char.filename}.jpg",
            )
            char_prompt.append(ImageModel.fetch_image_local_static(filepath=char_path))

        return char_prompt

    """ Page """

    def __construct_pages_prompt(self, pages: List[PageDataModel]) -> List:
        result = []
        pages = sorted(pages, key=lambda page: page.ord)
        for page in pages:
            result.append(f"Page {page.ord}")
            result.append(page.fetch_page_image())
            for panel in page.panels:
                result.append(f"Panel {panel.ord}")
                result.append(panel.fetch_panel_image())
        return result

    def summarize_pages_batch(
        self,
        manga_title: str,
        pages: List[PageDataModel],
        characters: List[CharacterDataModel],
        context: List[str] = [],
    ) -> List:
        """
        Creates a page-by-page narration of manga panels with manga context. DOES NOT use pre-page context yet.

        @params:\n
        manga_title: Title of the manga we are narrating.\n
        pages: List of page data models to narrate.\n
        characters: List of character data models this manga references.\n
        context: List of additional context to pass to LLM.\n

        @returns
        JSON string containing narrative segments for each panel.\n
        format:\n
        [{'page': int, 'text': str, 'view_window', 'view_window_start'}, {...}, ...]\n
        """
        self.logger.info(f"Narrating Pages... Pages to process: {len(pages)}")
        sys_prompt = "You are THE leading expert in narrating manga pages fluidly."
        initial_prompt = (
            f"{'This manga is called ' + manga_title + '. ' if manga_title else ''}"
        )
        char_prompt = self.__construct_char_prompt(characters=characters)
        base_prompt = f"{initial_prompt}"

        contents = []
        contents.append(base_prompt)
        contents.extend(char_prompt)
        narrative_instructions = (
            "---------------------------------------- THIS IS THE MAIN PROMPT ---------------------------------------- \n"
            "I will be giving you a set of manga pages and for each page I will also give you a set of panel images. \n"
            "Each page will come in the following format: Page number, followed immediately by page image, and then the panel images for that page given as panel # and panel image pairs. \n"
            "Your goal is to create a fluid well-written narration of the manga story given these pages. \n"
            "Then for each section of your narration, choose the best fit panel for it. \n"
            "The best fit panel is the panel that best captures the segment of the narration you picked for it. \n"
            "[CRITICAL]: Review all pages first to understand the context. Use this to properly map characters to text boxes and their names, and understanding the story. \n"
            "Here is a detailed template of how you will get the input, this example is for one page: \n"
            "Page number, Page Image, Panel Image 1, Panel Image 2, Panel Image 3 ... \n"
            "Here is a checklist list, ranked by importance (after ensuring JSON validity as described above): \n"
            "1. You need to embody the role of being a narrator. Don't overexplain scenes, but rather encapsulate the story. [CRITICAL] \n"
            "2. You need to understand the story so that it makes logical sense, esepcially when referring to characters doing actions, etc. \n"
            "3. When describing scenes and character interactions take into consideration all the panels and pages near it, so as to keep the continuity between scenes. \n"
            "4. Make sure the overall narrations per page flow fluidly between each other, I should not be able to tell that we are flipping pages. \n"
            "5. [SUPER-CRITICAL]: ALWAYS use the RIGHT name for each character in every panel. Mixing up names ruins the story instantly. Review each of your narration lines with the panel image AND the surrounding pages to get the best accuracy in identifying characters. \n"
            "6. [SUPER-CRITICAL]: Always attribute speech bubbles to the correct characters by carefully analyzing surrounding pages for contextual clues. Look beyond just speech bubble placement to ensure logical character interactions. As incorrect dialogue attribution severely confuses readers. \n"
            "7. [CRITICAL]: Keep your narration per page concise, do not repeat narrations. DO NOT directly read off of the text boxes, instead subtly incorporate them in your narration fluidly. \n"
            "8. When you break up each page narration for each panel you select, the total page narration itself MUST be as fluid as possible. \n"
            "9. IMPORTANT: Be cohesive when narrating a page, it must make sense within the context of surrounding pages. \n"
            "10 [CRITICAL]: Be concise with the panels you choose, especially if multiple panels refer to the same thing, you dont have to drone through every panel, try to be concise. \n"
            "11. IMPORTANT: Speech bubbles may appear in different panels than the speaking character—use context clues from the conversation flow and narrative to determine who is speaking, in addition to visual positioning. \n"
            "12. [CRITICAL]: Your narrations should NOT , under any circumstances, sound fragmented and unnatural, they should flow naturally and make sense throughout the entire narration. \n"
            "13. [CRITICAL]: Make sure your narration description accurately reflects what a viewer would see and understand from that page IN THE CONTEXT OF THE ENTIRE CHAPTER. Ensure your narration matches the reader's visual experience. \n"
            "14. IMPORTANT: Do NOT repeat plot points multiple times. \n"
            "15. You need to effectively narrate so that the pace momentum does NOT get stale. In specific dont repeat similar narrations. \n"
            "16. The narration segment you choose for a panel MUST ALWAYS be in complete sentences. \n"
            "17. Panels CANNOT have the same segment. If you cant find a good segment for a panel, then just label its narration as 'N/A'. \n"
            "18. When selecting panels, you need to select the most optimal ones such that the narration pace is not stale. \n"
            "19. IMPORTANT: Speak in the third person, and use names whenever possible but accurately, so that the reader knows who you are referring to. \n"
            "20. You do not have to use the full name of a character when reffering to them. \n"
            "21. IMPORTANT: If the page isn't a story page, but rather a cover page, or ending page, etc, return 'N/A' for its 'page_narration' entry. \n"
            "22. IMPORTANT: If a panel is not as relevant for a page's narration or too insignificant, return 'N/A' for it's narration, your goal is to give me the best panels within a page, not to use ALL of them. \n"
            "23. Use clear, standard language for easy subtitle conversion. Avoid special characters (ex: tildas ~ are not allowed), symbols, or unusual punctuation. Write out complete words rather than abbreviations. (Note: Standard double quotation marks within narrations are allowed ONLY IF properly escaped for JSON as per CRITICAL JSON OUTPUT REQUIREMENTS point C. Prefer indirect narration as per AVOID list rule #1). \n"
            "Here is a list of thing you should AVOID doing, ranked by importance: \n"
            "1. Do NOT directly quote text or speech bubbles by using quotation marks (\") in your narrative text unless absolutely unavoidable for a very short, key phrase. Instead, rephrase ALL dialogue and text from speech bubbles into indirect narration. For example, instead of narrating 'He said, \"Let's go!\"', narrate as 'He suggested they leave.' or 'He exclaimed that they should go.' This is an ABSOLUTE rule and helps ensure valid JSON. If you find it absolutely necessary to use quotation marks, they MUST be correctly escaped for JSON (see CRITICAL JSON OUTPUT REQUIREMENTS, point C). \n"
            "2. Do NOT narrate every panel - focus ONLY on the ones that drive the story forward. Any and all 'fluff' panels are strictly to be avoided. Identify what's important to the narrative and integrate ONLY these key moments naturally, skipping over less significant details. \n"
            "3. Do NOT be repetitive in your word choices and composition. \n"
            "4. DO NOT skip any page or panels. If you determine they are unimportant, label their narrations as 'N/A'. All pages and panels MUST either have a narration OR 'N/A'; you must guarantee this. \n"
            "5. [CRITICAL]: Do NOT make it obvious that we are reading a page. Narrate the story. don't tell me things like 'the text says' or 'the next panel' or 'going down the page', 'scene', 'the narrator says', etc. \n"
            "6. Do NOT overly refer to character names; this is highly unnatural. Make it concise and to the point. \n"
            "7. [CRITICAL]: DO NOT repeat text you see in panels. Only do so if it is ABSOLUTELY needed and adds to the narration. Rather work the panels into your narration ELEGANTLY. \n"
            "8. Do NOT pick panels that have a significantly small segment of narration, even if it is a complete sentence. \n"
            "9. Do NOT include any panel that doesn't have content in it (e.g., a cover panel, or a header, etc.). Label the narration as 'N/A'. \n"
            "10. Do NOT treat each batch as if it were the first batch given, unless explicitly told that it is the first batch. That means don't start with things like 'The chapter begins with...', etc. This will often break the flow of the narrative and seem jarring. \n"
            "11. Do NOT be overly wordy in your narration, too much complexity in word compisition is a detriment. \n"
            "12. Do NOT misgender characters, use context of each character mentioned within pages to accuratly gender all characters. \n"  # 05/27/25
            "Your response MUST be a valid JSON array. Each element in the array should be a JSON object representing a page. \n"
            "ULTRA CRITICAL JSON OUTPUT REQUIREMENTS: \n"
            "A. [ULTRA-CRITICAL]: COMMA SEPARATION: ALL key-value pairs within a JSON object MUST be separated by a comma. Similarly, ALL elements in a JSON array MUST be separated by a comma. This is IMPERATIVE. REVISE your response before sending to ABSOLUTELY make sure it is correct and all commas are present, especially if a PANEL or PAGE narration is 'N/A'. You MUST check this properly and revise if needed. \n"
            "B. Your entire response MUST be a single, valid JSON array. No other text, explanations, or markdown formatting should surround this JSON array. \n"
            "C. All string values within the JSON (such as those for 'page_narration' and 'panel_narration' keys) MUST themselves be valid JSON strings. \n"
            'D. This means any double quotation mark (") that is part of the narration text itself (e.g., for dialogue or emphasis) MUST be escaped with a backslash. For example, if a narration is \'He thought, "This is it!"\', it MUST appear in the JSON string as "He thought, \\"This is it!\\"". \n'
            "E. Incorrectly formatted JSON, especially due to unescaped quotes within narration strings, will make the output unusable. You MUST guarantee valid JSON. This is the MOST IMPORTANT requirement, overriding all other stylistic considerations if there is a conflict. \n"
            "Each page object must contain the following keys: \n"
            f"- '{PageNarratorConstants.ORD}': an integer representing the page number provided. \n"
            f"- '{PageNarratorConstants.NARRATION_LABEL}': a string containing the overall narration for the page, or the string \"N/A\" for non-story pages. \n"
            f"- '{PageNarratorConstants.PANELS}': a JSON array of panel objects. \n"
            "Each panel object within the '{PageNarratorConstants.PANELS}' array must contain the following keys: \n"
            f"- '{PanelNarratorConstants.ORD}': an integer representing the panel number. \n"
            f"- '{PanelNarratorConstants.NARRATION_LABEL}': a string containing a segment of the page_narration that best fits this panel, or the string \"N/A\". \n"
            "Ensure the entire response is ONLY the JSON array, with no other surrounding text or explanations. \n"
            "[ULTRA-CRITICAL] You MUST get the page and panel numbers correct in your output, this is absolutely IMPERATIVE. Even if the narration is 'N/A' \n"
            "Finally, Here are the pages and panels interleaved: \n"
        )
        pages_instructions = self.__construct_pages_prompt(pages=pages)
        context_instructions = f"{context}" if context else None
        try:
            contents.append(narrative_instructions)
            contents.extend(pages_instructions)
            contents.append(context_instructions)

            # Filter out None values before sending to API
            contents = [item for item in contents if item is not None]

            # Check tokens
            # tokens = self.token_count(model=self.google_models[0], contents=contents)
            # self.logger.info(f"Tokens: {tokens}")

            # # List models
            # models = self.list_models()
            # self.logger.info(f"Models: {[model.get('name') for model in models]}")
            # exit(0)

            response = self.generate_content_with_models(
                models=self.google_models,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    temperature=self.temperature,  # 0.7 previous (increasing this make the model more random/creative)
                    response_mime_type="application/json",
                    # thinking_config=types.ThinkingConfig(thinking_budget=24576),
                ),
            )

            try:
                formatted_narration = self.json_parser.parse(response.text)
                page_map: Dict[any, PageDataModel] = {
                    page.ord: page for page in pages
                }  # Assumes ORD is unique
                for page_entry in formatted_narration:
                    page_narration = page_entry.get(
                        PageNarratorConstants.NARRATION_LABEL
                    )
                    if page_narration:
                        page = page_map.get(page_entry.get(PageNarratorConstants.ORD))
                        self.logger.info(
                            f"Narrator saving page narration at path: {page.path}"
                        )
                        page.meta["summary"] = page_narration
                        panel_entries = page_entry.get(PageNarratorConstants.PANELS)
                        if panel_entries:
                            if not page.panels:
                                self.api.hydrate_page(page=page)

                            # Populate panel meta.
                            panel_map = {panel.ord: panel for panel in page.panels}
                            for panel_entry in panel_entries:
                                ord = int(panel_entry.get(PanelNarratorConstants.ORD))
                                panel = panel_map.get(ord)
                                if not panel:
                                    self.logger.error(
                                        f"Could not find panel with ord: {ord} for page: {page.ord}"
                                    )
                                    Utils.pause("Continue? (y/n/e)")
                                else:
                                    summary = panel_entry.get(
                                        PanelNarratorConstants.NARRATION_LABEL
                                    )
                                    if not panel.meta:
                                        panel.meta = {"summary": summary}
                                    else:
                                        panel.meta["summary"] = summary
                                    self.api.save_panel(panel=panel, force=True)
                        self.api.save_page(page=page, depth=2, force=True)

                # Assume no summary means N/A
                for page in pages:
                    if not page.meta.get("summary"):
                        while True:
                            user_response = input(
                                f"Page {page.id} is missing a summary. Would you like to fill it with 'N/A'? (y/n/e): "
                            )
                            if user_response.lower() in ["y", "n"]:
                                break
                            print("Invalid response. Please enter 'y' or 'n'.")

                        if user_response.lower() == "y":
                            page.meta["summary"] = "N/A"

                        panel_updates = False
                        for panel in page.panels:
                            if not panel.meta.get("summary"):
                                while True:
                                    panel_response = input(
                                        f"Panel {panel.id} in page {page.id} is missing a summary. Would you like to fill it with 'N/A'? (y/n/e): "
                                    )
                                    if panel_response.lower() in ["y", "n"]:
                                        break
                                    print("Invalid response. Please enter 'y' or 'n'.")

                                if panel_response.lower() == "y":
                                    panel.meta["summary"] = "N/A"
                                    panel_updates = True

                        # Only save if any changes were made
                        if user_response.lower() == "y" or panel_updates:
                            self.api.save_page(page=page, depth=2, force=True)

                return formatted_narration
            except Exception as e:
                self.logger.error(
                    f"Error processing response: {str(e)}\n{traceback.format_exc()}"
                )
                Utils.pause("Continue? (y/n/e)")
                return None
        except Exception as e:
            self.logger.info(f"Encountered Error while summarizing pages in batch: {e}")
            return None

    def summarize_pages(
        self,
        pages: List[PageDataModel],
        characters: List[CharacterDataModel] = [],
        context: List[str] = [],
        batch_size: int = 15,
    ) -> List[Dict]:
        """
        Creates a page-by-page narration of manga pages with manga context.
        Returns a JSON string containing narrative segments for each page.
        """
        self.logger.info("Constructing character map...")
        batch_idx = 0
        batch_narrations: List[List[Dict]] = []
        for i in range(0, len(pages), batch_size):
            batch = pages[i : i + batch_size]
            batch_prefix = (
                " ".join(
                    narration["text"]
                    for narration in batch_narrations[batch_idx - 1]
                    if narration.get("text") not in ("N/A", None)
                )
                if batch_idx > 0
                else ""
            )
            if batch_prefix:
                context.append(
                    (
                        "Here are the PREVIOUS page narrations for additional context. ",
                        "Most importantly, I need you to continue the narration fluidly given these previous page narrations. ",
                        batch_prefix,
                    )
                )

            # Process pages
            narration: List[Dict] = self.summarize_pages_batch(
                manga_title=self.manga,
                pages=batch,
                characters=characters,
                context=context,
            )

            # Append narration
            if narration:
                batch_narrations.append(narration)
                batch_idx += 1
            else:
                self.logger.warning(f"Failed to narrate batch starting at index {i}")

        return [item for sublist in batch_narrations for item in sublist]  # Flattened

    """ Chapter """

    def __constructs_chapters_prompt(self, chapters: List[ChapterDataModel]):
        content = []
        for chapter in chapters:
            chapter_narrations = []
            skip_chapter = False

            # Hydrate chapter if needed
            if not chapter.pages:
                self.api.hydrate_chapter(chapter=chapter)
            for page in chapter.pages:
                page_narration = page.meta.get("summary")
                if page_narration is None:  # Only skip if it's None
                    skip_chapter = True
                    self.logger.info(
                        f"Skipping chapter {chapter.ord} summarization due to a missing page narration: {page.ord}"
                    )
                    break
            if skip_chapter:
                continue
            for page in chapter.pages:
                page_narration = page.meta.get("summary")
                if (
                    page_narration != "N/A"
                ):  # Skip "N/A" values but include everything else
                    chapter_narrations.append(page_narration)

            if chapter_narrations:
                content.append(f"Chapter {chapter.ord}\n")
                content.append("\n".join(chapter_narrations))
        return "\n".join(content)

    def summarize_chapters(self, chapters: List[ChapterDataModel]):
        """Given a list of chapters, extract a narration for each given populated page narrations."""
        # Checks
        chapters = [chapter for chapter in chapters if not chapter.meta.get("summary")]

        if len(chapters) == 0:
            self.logger.info("All chapters have summaries, not summarizing.")
            return  # Dont continue for empty chapters.
        sys_prompt = "You are the leading expert in summarizing manga chapters based on its page narrations."
        chapter_instructions = self.__constructs_chapters_prompt(chapters=chapters)
        if not chapter_instructions:
            return None  # No chapter instruction
        narrative_instructions = (
            "Your job is to give me a chapter summary of about 2-4 sentences for each chapter, given a consolidated string of each chapter's page narrations. \n"
            "You must properly summarize everything given, making sure to choose the most relevant texts of information such that as a whole the chapter can be understood deeply. \n"
            "Your input will be the following: the chapter number, followed by a large string of text that is the aggregated narrations of each page in that chapter, then the next chapter number followed by its page narrations, etc... \n"
            "Here are your TODO's sorted top down by importance: \n"
            "1. Chapter summaries must always be between 2-4 sentences. \n"
            "2. Chapter summaries MUST at all costs be able to encapsulate the entirety of the chapter deeply. \n"
            "3. Be descriptive but not overly wasteful when constructing the summary. \n"
            "Give me the output in the following format: \n"
            "\n\n[\n"
            "  {\n"
            f"    {ChapterNarratorConstants.ORD}: integer, // The Chapter number provided\n"
            f"    {ChapterNarratorConstants.SUMMARY_LABEL}: string, // Summary for the chapter\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
            "Finally, Here are the chapters and pages interleaved: \n"
        )
        contents = []
        contents.append(narrative_instructions)
        contents.append(chapter_instructions)

        contents = [item for item in contents if item is not None]
        chapter_map = {
            chapter.ord: chapter for chapter in chapters
        }  # Assumes ORD is unique
        response = self.generate_content_with_models(
            models=["models/gemini-2.5-flash-preview-05-20"],
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                temperature=self.temperature,  # 0.7 previous
            ),
        )
        try:
            formatted_summaries = self.json_parser.parse(response.text)
            for chapter_entry in formatted_summaries:
                chapter_summary = chapter_entry.get(
                    ChapterNarratorConstants.SUMMARY_LABEL
                )
                if chapter_summary:
                    ord = str(chapter_entry.get(ChapterNarratorConstants.ORD))
                    chapter = chapter_map.get(ord)
                    self.logger.info(f"PULLED CHAPTER: {chapter}")
                    if chapter:
                        self.logger.info(
                            f"Narrator saving chapter summary for chapter: {chapter.ord}"
                        )
                        chapter.meta["summary"] = chapter_summary
                        self.api.save_chapter(chapter=chapter, force=True, depth=1)
        except Exception as e:
            self.logger.error(f"Error processing chapter summaries: {str(e)}")
            return None

    """ Helpers """

    def _strip_markdown_json_fences(self, text: str) -> str:
        """
        Strips Markdown code fences (```json ... ``` or ``` ... ```)
        from the beginning and end of a string.
        """
        match = re.search(
            r"^\s*```(?:[a-zA-Z0-9]*)?\s*\n?(.*?)\n?\s*```\s*$",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            extracted_content = match.group(1)
            self.logger.debug(
                f"Stripped Markdown fences. Extracted content (first 100 chars): {extracted_content[:100]}..."
            )
            return extracted_content

        # Fallback for a simpler case if the above regex is too strict or misses an edge case
        # This is less robust but might catch some simple ```...``` wraps not caught above.
        stripped_text = text.strip()
        if stripped_text.startswith("```") and stripped_text.endswith("```"):
            lines = stripped_text.splitlines()
            if len(lines) > 1:  # Needs at least one line for content + fences
                start_index = 0
                if lines[0].lower().strip().startswith("```"):  # e.g. ```json or ```
                    start_index = 1

                end_index_offset = 0
                if lines[-1].strip() == "```":
                    end_index_offset = 1

                if start_index == 1 and end_index_offset == 1 and len(lines) > 2:
                    cleaned_text = "\n".join(lines[start_index:-end_index_offset])
                    self.logger.debug(
                        f"Fallback Markdown fence stripping applied. Result (first 100 chars): {cleaned_text[:100]}..."
                    )
                    return cleaned_text
                elif (
                    start_index == 0
                    and end_index_offset == 0
                    and lines[0].startswith("```")
                    and lines[0].endswith("```")
                    and len(lines[0]) > 6
                ):
                    cleaned_text = lines[0][3:-3]
                    self.logger.debug(
                        f"Fallback single-line Markdown fence stripping. Result: {cleaned_text[:100]}..."
                    )
                    return cleaned_text

        self.logger.debug(
            "No Markdown fences detected or stripped by _strip_markdown_json_fences."
        )
        return text

    def __parse_response_json(self, response):
        raw_narration_from_llm = response.text
        self.logger.debug(
            f"Raw response from LLM (first 500 chars):\n{raw_narration_from_llm[:500]}"
        )

        # --- OPTIMIZED FIRST ATTEMPT: Direct parsing of raw LLM response ---
        try:
            formatted_narration = json.loads(raw_narration_from_llm)
            self.logger.info(
                "Direct JSON parsing of raw LLM response succeeded (mime_type likely honored)."
            )
            return formatted_narration
        except json.JSONDecodeError as e:
            self.logger.warning(
                f"Direct JSON parsing of raw LLM response failed: {e}. Proceeding with cleaning attempts."
            )

        text_after_fence_stripping = self._strip_markdown_json_fences(
            raw_narration_from_llm
        )
        if text_after_fence_stripping != raw_narration_from_llm:
            self.logger.debug(
                f"Text after fence stripping (first 500 chars):\n{text_after_fence_stripping[:500]}"
            )
            try:
                formatted_narration = json.loads(text_after_fence_stripping)
                self.logger.info(
                    "JSON parsing succeeded after stripping Markdown fences."
                )
                return formatted_narration
            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"JSON parsing failed even after stripping Markdown fences: {e}. Attempting further fixes."
                )
                text_to_process = text_after_fence_stripping
        else:
            self.logger.debug(
                "No changes made by fence stripping, using original raw_narration for further cleaning."
            )
            text_to_process = raw_narration_from_llm

        original_response_text_for_error = raw_narration_from_llm

        # --- Second attempt: Try various cumulative cleaning approaches ---
        def _escape_quotes_in_string_content(match_obj):
            string_content = match_obj.group(2)
            escaped_content = re.sub(r'(?<!\\)"', r'\\"', string_content)
            return f"{match_obj.group(1)}{escaped_content}{match_obj.group(3)}"

        cleaning_steps = [
            ("strip_whitespace", lambda x: x.strip()),
            (
                "remove_bom_invisible_chars",
                lambda x: x.replace("\ufeff", "").replace("\u200b", ""),
            ),
            (
                "normalize_quotes",
                lambda x: x.replace("'", '"')
                .replace("`", '"')
                .replace("“", '"')
                .replace("”", '"'),
            ),
            ("escape_literal_newlines", lambda x: re.sub(r"(?<!\\)\n", r"\\n", x)),
            (
                "escape_internal_double_quotes_in_strings",
                lambda x: re.sub(
                    r'(")([^"\\]*(?:\\.[^"\\]*)*)(")',
                    _escape_quotes_in_string_content,
                    x,
                ),
            ),
            (
                "fix_js_style_trailing_commas",
                lambda x: re.sub(r",\s*([}\]])", r"\1", x),
            ),
        ]

        self.logger.debug(
            f"Starting cumulative cleaning attempts on text (first 200): {text_to_process[:200]}"
        )
        for name, clean_func in cleaning_steps:
            try:
                cleaned_text_candidate = clean_func(text_to_process)
                formatted_narration = json.loads(cleaned_text_candidate)
                self.logger.info(
                    f"JSON parsing succeeded after cumulative cleaning step: '{name}'"
                )
                return formatted_narration
            except json.JSONDecodeError:
                if cleaned_text_candidate != text_to_process:
                    self.logger.debug(
                        f"Cleaning step '{name}' applied. Text changed but still not parsable."
                    )
                    text_to_process = cleaned_text_candidate
                else:
                    self.logger.debug(
                        f"Cleaning step '{name}' made no change or did not lead to parsable JSON."
                    )
            except Exception as e_clean:
                self.logger.warning(
                    f"Error during '{name}' cleaning: {e_clean}. Using text from before this failed step."
                )

        self.logger.debug("Cumulative cleaning finished. Proceeding to extraction.")

        # --- Third attempt: Try to extract JSON structures ---
        extraction_patterns = [
            (r"^\s*\[(.*)\]\s*$", lambda m: f"[{m.group(1)}]"),
            (r"^\s*\{(.*)\}\s*$", lambda m: f"{{{m.group(1)}}}"),
            (r"\[(.*?)\]", lambda m: f"[{m.group(1)}]"),
            (r"\{(.*?)\}", lambda m: f"{{{m.group(1)}}}"),
        ]

        for pattern_regex, formatter in extraction_patterns:
            json_match = re.search(pattern_regex, text_to_process, re.DOTALL)
            if json_match:
                extracted_text = formatter(json_match)
                self.logger.debug(
                    f"Attempting to parse extracted text (using pattern '{pattern_regex}', first 200 chars): {extracted_text[:200]}"
                )
                current_extracted_text_to_fix = extracted_text
                text_after_quote_fix = current_extracted_text_to_fix
                try:
                    formatted_narration = json.loads(current_extracted_text_to_fix)
                    self.logger.info(
                        "JSON parsing succeeded after extraction (direct parse)."
                    )
                    return formatted_narration
                except json.JSONDecodeError:
                    pass
                try:
                    text_after_quote_fix = current_extracted_text_to_fix.replace(
                        "'", '"'
                    )
                    if text_after_quote_fix != current_extracted_text_to_fix:
                        formatted_narration = json.loads(text_after_quote_fix)
                        self.logger.info(
                            "JSON parsing succeeded after extraction and quote fixing."
                        )
                        return formatted_narration
                except json.JSONDecodeError:
                    pass
                try:
                    base_for_comma_fix = text_after_quote_fix
                    text_after_comma_fix = re.sub(
                        r",\s*([}\]])", r"\1", base_for_comma_fix
                    )
                    if text_after_comma_fix != base_for_comma_fix:
                        formatted_narration = json.loads(text_after_comma_fix)
                        self.logger.info(
                            "JSON parsing succeeded after extraction, quote, and comma fixing."
                        )
                        return formatted_narration
                except json.JSONDecodeError:
                    pass
            else:
                self.logger.debug(
                    f"Extraction pattern '{pattern_regex}' did not match."
                )

        self.logger.debug(
            "All extraction and sub-fixing attempts failed. Proceeding to line-by-line."
        )

        # --- Fourth attempt: Use a line-by-line approach ---
        self.logger.debug(
            f"Attempting line-by-line fixing on text (first 200): {text_to_process[:200]}"
        )
        lines = text_to_process.split("\n")
        fixed_lines = []
        for i, line_content in enumerate(lines):
            if not line_content.strip():
                if fixed_lines or i < len(lines) - 1:
                    fixed_lines.append(line_content)
                continue
            processed_line = line_content
            processed_line = re.sub(r"^\s*([\w.-]+)\s*:", r'"\1":', processed_line)
            processed_line = re.sub(
                r"([{,]\s*)([\w.-]+)\s*:", r'\1"\2":', processed_line
            )
            processed_line = re.sub(r",\s*$", "", processed_line.rstrip("\r")).strip()
            if processed_line != line_content:
                self.logger.debug(
                    f"Line {i+1} changed: '{line_content}' -> '{processed_line}'"
                )
            fixed_lines.append(processed_line)

        final_attempted_text = "\n".join(fixed_lines)
        self.logger.debug(
            f"Attempting to parse line-by-line fixed JSON (first 500 chars):\n{final_attempted_text[:500]}"
        )
        try:
            formatted_narration = json.loads(final_attempted_text)
            self.logger.info("JSON parsing succeeded after line-by-line fixing.")
            return formatted_narration
        except json.JSONDecodeError as e:
            final_error_message = f"All JSON parsing attempts failed. Last error on line-by-line fixed text: {e.msg}. Error at pos {e.pos} (line {e.lineno}, col {e.colno}). Original LLM response (first 100 chars): '{original_response_text_for_error[:100]}...'"
            self.logger.error(final_error_message)
            error_context_lines = final_attempted_text.split("\n")
            start_idx = max(0, e.lineno - 3)
            end_idx = min(len(error_context_lines), e.lineno + 2)
            context = "\n".join(
                [
                    f"{idx+1:03d}: {error_context_lines[idx]}"
                    for idx in range(start_idx, end_idx)
                ]
            )
            self.logger.error(
                f"Problematic context in final attempted parse (lines {e.lineno-2}-{e.lineno+1}):\n{context}"
            )
            print("\n=== RAW RESPONSE (JSON PARSING FAILED) ===\n")
            print(original_response_text_for_error)
            print("\n=== FINAL ATTEMPTED TEXT (JSON PARSING FAILED) ===\n")
            print(final_attempted_text)
            print("\n==============================================\n")
            raise json.JSONDecodeError(
                msg=final_error_message, doc=final_attempted_text, pos=e.pos
            ) from e
