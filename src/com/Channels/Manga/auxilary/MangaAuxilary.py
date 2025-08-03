# Imports
import re
from typing import List, Optional, Tuple
from google.genai import types

# Custom
from ozkit.src.com.Utils import Utils
from ozkai_nlp.llm.Google import GoogleLLM
from ozkapi.src.com.api.MangaApi import MangaApi
from manga.src.models import (
    ChapterDataModel,
    PageDataModel,
    PanelDataModel,
    CharacterDataModel,
    ImageModel,
)


class MangaAuxilary(GoogleLLM):
    def __init__(
        self,
        api: MangaApi,
        manga: str,
        models=[
            "models/gemini-2.5-pro-preview-06-05",
            "models/gemini-2.5-flash-preview-05-20",
            "models/gemini-2.5-pro-preview-03-25",
            "models/gemini-2.5-flash-preview-04-17",
        ],
    ):
        self.api = api
        self.manga = manga
        self.google_models = models
        super().__init__(models)

    """Title"""

    def __construct_title_prompt(self, chapters: List["ChapterDataModel"]) -> List[str]:
        summaries = [chapter.meta.get("summary") for chapter in chapters]

        if None in summaries:
            summaries = []
            for chapter in chapters:
                hydrated_chapter = self.api.hydrate_chapter(chapter=chapter)
                if not hydrated_chapter.pages:
                    self.logger.warning(
                        "Continuous summary chain not found in either chapter or pages."
                    )
                    Utils.pause("Continue to contruct title? (y/n/e)")
                    continue
                else:
                    summaries.extend(
                        [
                            page.meta.get("summary", "")
                            for page in hydrated_chapter.pages
                        ]
                    )
        return [str(s) for s in summaries if s is not None]

    def __construct_reference_prompt(
        self, references: List[Tuple[str, int]] = []
    ) -> List[str]:
        refs = []
        for reference in references:
            title = reference[0]
            views = reference[-1]
            refs.append(f"{title} ---> {views}")
        return refs

    def construct_title(
        self,
        chapters: List["ChapterDataModel"],
        references: List[Tuple[str, int]],
        context: Optional[List[any]] = None,
    ) -> Optional[str]:
        generated_title = None

        try:
            summaries_list = self.__construct_title_prompt(chapters=chapters)
            summaries = "\n".join(summaries_list)

            refs_list = self.__construct_reference_prompt(references=references)
            refs = "\n".join(refs_list)

            context = context if context else []

            if not summaries and not refs:
                self.logger.error(
                    "Cannot generate title without summaries or references."
                )
                return None

            sys_prompt = "You are the leading expert in creating one liners for Youtube manga recap video titles."
            title_prompt_lines = (
                "I will be giving you a brief list of summaries for a manga. \n",
                "Your goal in life is to create the very best one-liner title that would get the most clicks based on the summaries. \n",
                "I will also be giving you a series of one-liners that have done really well that others have used. \n",
                "The one liners will be given like this: <reference_title> ---> <number_of_views> \n",
                "Analyze the one liners deeply, and combined with the summaries I gave you create the very best possible title to get views. \n",
                "Your output should just be the one sentence title. \n",
                "Your aim is to guarantee clicks. \n",
            )
            title_prompt = "".join(title_prompt_lines)

            contents = [title_prompt, summaries, refs, *context]
            self.logger.info(f"Requesting title creation from LLM.")
            response = self.generate_content_with_models(
                models=self.google_models,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    temperature=0.85,
                ),
            )

            if response and hasattr(response, "text") and response.text:
                response_text = response.text.strip()
                if response_text:
                    generated_title = response_text
                    self.logger.info(
                        f"Successfully generated title: '{generated_title}'"
                    )
                else:
                    self.logger.error(
                        "LLM returned an empty string response for the title."
                    )
                    Utils.pause("Continue? (y/n/e)")
            else:
                self.logger.error(
                    "LLM response was empty or invalid for title generation."
                )
                Utils.pause("Continue? (y/n/e)")

        except Exception as e:
            self.logger.error(
                f"Error during title generation API call or processing: {e}",
                exc_info=True,
            )
            Utils.pause("Continue? (y/n/e)")

        return generated_title

    """Thumbnail"""

    def __get_panel_id(self, panel: PanelDataModel) -> str:
        return abs(panel.__hash__())

    def __get_panel_id_map(self, panels: List[PanelDataModel]):
        panel_id_map = {}
        for panel in panels:
            panel_id = self.__get_panel_id(panel=panel)
            if panel_id in panel_id_map:
                self.logger.warning(
                    f"Collision detected for panel ID {panel_id}. Panel ord={panel.ord} might overwrite a previous one in the map. Using stable IDs is recommended."
                )
                Utils.pause("Continue? (y/n/e)")
            panel_id_map[panel_id] = panel
        return panel_id_map

    def find_common(self, nums: List[int]):
        d = {}
        print("SIZE: ", len(nums))
        for num in nums:
            if d.get(num) is not None:
                print("Found repetition: ", num)
                return
            else:
                d[num] = num
        print("ALL NUMS ARE UNIQUE!")

    def __construct_panels_prompt(self, panels: List[PanelDataModel]) -> List:
        result = []
        panels = sorted(panels, key=lambda panel: panel.ord)
        for panel in panels:
            result.append(f"Panel {self.__get_panel_id(panel=panel)}")
            result.append(panel.fetch_panel_image())
        return result

    def construct_thumbnail(
        self, panels: List["PanelDataModel"]
    ) -> Optional["PanelDataModel"]:
        if not panels:
            self.logger.warning("No panels provided for thumbnail selection.")
            return None

        contents = []
        sys_prompt = (
            "You are a world class editor who knows precisely how the youtube algorithm works "
            "and are the leading expert in selecting the very best thumbnail for a manga recap channel to optimize views."
        )
        thumbnail_instruction = (
            "I will be giving you a series of panels with their panel id and image. \n"
            "Specifically it will be given in the following format: Panel ID, followed directly by that panel's image. \n"
            "Your goal is to find the most PROVOCATIVE panel that would be the very best thumbnail for the story. Best here doesnt mean what encapsulates the story, rather it is the one that will get the MOST clicks [CRITICAL]. \n"
            "For the Thumbnail, you have to choose a captivating panel, it will typically contain our main character or a high-impact moment. \n"
            "Lastly you can only pick a single panel, so focus deeply on finding the very best one. \n"
            "All your output should be is the single panel id mapping to the optimal thumbnail panel. ONLY output the id number and nothing else. \n"
            "IT SHOULD NOT LOOK OVERLY CLUTTERED! \n"
            "[SUPER-CRITICAL]: You must select a panel as the thumbnail from the given panels. \n"
            "DO NOT RETURN AN EMPTY RESPONSE. \n"
        )

        panel_id_map = self.__get_panel_id_map(panels=panels)
        print("\n\n\n", panel_id_map, "\n\n\n")

        panels_prompt = self.__construct_panels_prompt(panels=panels)

        selected_panel = None

        try:
            contents.append(thumbnail_instruction)
            contents.extend(panels_prompt)
            contents = [item for item in contents if item is not None]

            self.logger.info(
                f"Requesting thumbnail selection from LLM for {len(panels)} panels."
            )
            response = self.generate_content_with_models(
                models=self.google_models,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=sys_prompt,
                    temperature=0.65,
                    # thinking_config=types.ThinkingConfig(thinking_budget=24576),
                ),
            )
            if response and hasattr(response, "text") and response.text:
                response_text = response.text.strip()
                self.logger.info(f"LLM raw response for thumbnail: '{response_text}'")

                match = re.search(r"\b(\d{4})\b", response_text)
                if not match:
                    match = re.search(r"(\d+)", response_text)

                if match:
                    try:
                        selected_id_str = match.group(1)
                        selected_id = int(selected_id_str)
                        self.logger.info(f"Parsed potential panel ID: {selected_id}")

                        if selected_id in panel_id_map:
                            selected_panel = panel_id_map[selected_id]
                            self.logger.info(
                                f"Successfully selected thumbnail panel ID: {selected_id:04d}, Ord: {selected_panel.ord}"
                            )
                        else:
                            self.logger.error(
                                f"LLM returned panel ID {selected_id}, but it was not found in the initial panel map. Possible collision, hash instability, or LLM error."
                            )
                            self.logger.debug(
                                f"Available panel IDs in map: {list(panel_id_map.keys())}"
                            )

                    except ValueError:
                        self.logger.error(
                            f"Could not convert extracted text '{selected_id_str}' to an integer ID."
                        )
                        Utils.pause("Continue? (y/n/e)")
                    except Exception as e:
                        self.logger.error(
                            f"Error processing extracted ID {selected_id_str}: {e}"
                        )
                        Utils.pause("Continue? (y/n/e)")

                else:
                    self.logger.error(
                        f"Could not find a valid panel ID number in the LLM response: '{response_text}'"
                    )
                    Utils.pause("Continue? (y/n/e)")

            else:
                self.logger.error("LLM response was empty or invalid.")

        except Exception as e:
            self.logger.error(
                f"Error during thumbnail selection API call or processing: {e}",
                exc_info=True,
            )
            Utils.pause("Continue? (y/n/e)")

        if selected_panel is None:
            self.logger.warning("Failed to select a thumbnail panel.")
            Utils.pause("Continue? (y/n/e)")

        return selected_panel
