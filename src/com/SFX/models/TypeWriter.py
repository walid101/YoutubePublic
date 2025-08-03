import os
import time
import srt
import torch
import torch.nn.functional as F
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, NamedTuple, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Lock, Thread, Event
import subprocess
from tqdm import tqdm
import gc
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

# Constants
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_FALLBACK_CHAR_WIDTH_RATIO = (
    0.6  # If a char is not in atlas, guess its width as ratio of atlas_char_cell_width
)


# Data Classes (ensure they are defined before TypeWriterAnimator)
@dataclass
class FontSettings:
    path: str
    size: int = 48
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)  # RGBA


@dataclass
class LayoutSettings:
    position_xy: Tuple[int, int] = (50, 50)
    align: str = "left"  # "left", "center", "right"
    max_width: Optional[int] = None
    total_height: Union[str, int] = "65%"
    line_spacing_multiplier: float = 1.2


@dataclass
class CursorSettings:
    style: str = "block"  # "block", "underscore", "pipe", "none"
    color: Tuple[int, int, int, int] = (255, 255, 255, 200)  # RGBA
    blink_hz: float = 2.0
    width_px: int = 2  # For pipe/underscore
    hide_after_typing_segment: bool = True


@dataclass
class AnimationParams:
    typing_cps: float = 15.0
    cursor: CursorSettings = field(default_factory=CursorSettings)
    segment_end_hold_sec: float = 0.5


@dataclass
class FrameOutputSettings:
    resolution_wh: Tuple[int, int] = (1920, 1080)
    fps: int = 30


@dataclass
class AnimationConfig:
    srt_file_path: str
    output_file: str
    font: FontSettings = field(default_factory=FontSettings)
    layout: LayoutSettings = field(default_factory=LayoutSettings)
    animation: AnimationParams = field(default_factory=AnimationParams)
    output: FrameOutputSettings = field(default_factory=FrameOutputSettings)
    max_workers: int = 4
    ffmpeg_threads: int = 2
    cache_limit_multiplier: float = 2.0
    use_fp16: bool = False
    encoder: str = "libx264"
    quality: int = 23
    preset: str = "medium"


class TextLine(NamedTuple):
    text: str
    y_offset: float
    width: float  # Actual pixel width of the text using the font


class FrameTextState(NamedTuple):
    full_segment_text_lines: List[TextLine]
    num_chars_to_display: int
    cursor_line_idx: int
    cursor_char_idx_in_line: int
    is_typing_done_for_segment: bool


class WorkerFrameTask(NamedTuple):
    frame_idx: int
    current_time_sec: float
    segment_srt_obj: Optional[srt.Subtitle]


# Structure for pre-calculated segment data
class PrecomputedSegmentData(NamedTuple):
    cleaned_text: str
    duration_sec: float
    conceptual_typing_end_time_sec: (
        float  # Time within segment when typing animation conceptually finishes
    )
    layout_lines: List[TextLine]
    block_width: float  # Max width of any line in this segment's layout
    block_height: float  # Total height of the laid-out block


class TypeWriterAnimator:
    def __init__(self, config: AnimationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.float16
            if config.use_fp16 and self.device.type == "cuda"
            else torch.float32
        )

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.max_workers = max(1, self.config.max_workers)
        self.results_cache_limit = max(
            2, int(self.max_workers * config.cache_limit_multiplier)
        )

        self._futures_dict: Dict[int, Any] = {}
        self._results_cache: Dict[int, Union[bytes, Exception]] = {}
        self._cache_lock = Lock()
        self._last_written_frame_index = -1

        self._processing_failed_event = Event()
        self._ffmpeg_writer_should_stop = Event()

        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._render_executor: Optional[ThreadPoolExecutor] = None
        self._ffmpeg_writer_thread: Optional[Thread] = None
        self._pbar: Optional[tqdm] = None

        self.pil_font: Optional[PIL.ImageFont.ImageFont] = None
        self.char_height: float = 0
        self.char_baseline_offset: float = 0
        self.atlas_char_cell_width: float = 0
        self.font_atlas: Optional[torch.Tensor] = None
        self.char_uvs: Dict[str, Tuple[float, float, float, float]] = {}

        # OPTIMIZATION: Store actual character widths
        self.char_actual_widths: Dict[str, float] = {}

        # OPTIMIZATION: Enhanced pre_layouts to store more segment-specific precomputed data
        self.precomputed_segment_cache: Dict[int, PrecomputedSegmentData] = {}

        self._ensure_dir_exists(os.path.dirname(config.output_file))
        self._preprocess_font()

    def _ensure_dir_exists(self, dir_path: str):
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def _get_char_width(self, char_code: str) -> float:
        """Gets pre-calculated width for a char, with fallback."""
        return self.char_actual_widths.get(
            char_code, self.atlas_char_cell_width * DEFAULT_FALLBACK_CHAR_WIDTH_RATIO
        )

    def _get_text_width(self, text: str) -> float:
        """Calculates width of a string using pre-calculated char widths."""
        if not text:
            return 0.0
        # This check ensures pil_font is available for _layout_text_block which still uses it directly
        # For other uses, _get_char_width should be primary
        if (
            not self.char_actual_widths and self.pil_font
        ):  # Fallback if char_actual_widths not populated yet (e.g. during early init)
            return self.pil_font.getlength(text)
        return sum(self._get_char_width(c) for c in text)

    def _preprocess_font(self):
        # More comprehensive character set for atlas
        chars = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
            "!@#$%^&*()_+-=[]{}|;':\",./<>?`~¡¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß"
        )
        try:
            self.pil_font = PIL.ImageFont.truetype(
                self.config.font.path, self.config.font.size
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to load font {self.config.font.path}: {e}. Using default."
            )
            self.pil_font = PIL.ImageFont.load_default()

        ascent, descent = self.pil_font.getmetrics()
        self.char_height = float(ascent + descent)
        self.char_baseline_offset = float(
            -ascent
        )  # PIL's ascent is positive, offset is negative

        # OPTIMIZATION: Pre-calculate actual widths for all chars in our set
        for char_code in chars:
            try:
                # Ensure getlength exists, especially for default font
                if hasattr(self.pil_font, "getlength"):
                    self.char_actual_widths[char_code] = self.pil_font.getlength(
                        char_code
                    )
                elif hasattr(self.pil_font, "getsize"):  # Fallback for older PIL/Pillow
                    self.char_actual_widths[char_code] = self.pil_font.getsize(
                        char_code
                    )[0]
                else:  # Absolute fallback
                    self.char_actual_widths[char_code] = (
                        self.config.font.size * 0.6
                    )  # Guess
            except Exception:  # Catch any error during getlength/getsize
                self.char_actual_widths[char_code] = (
                    self.config.font.size * 0.6
                )  # Guess

        # For atlas cell width, use max of pre-calculated widths or a robust fallback
        # Ensure 'M' is in char_actual_widths for reliable space width in atlas if ' ' has 0 width from font
        if "M" not in self.char_actual_widths and hasattr(self.pil_font, "getlength"):
            self.char_actual_widths["M"] = self.pil_font.getlength("M")
        elif "M" not in self.char_actual_widths and hasattr(self.pil_font, "getsize"):
            self.char_actual_widths["M"] = self.pil_font.getsize("M")[0]

        space_width_for_atlas = self.char_actual_widths.get(
            " ", self.char_actual_widths.get("M", self.config.font.size * 0.5)
        )

        valid_widths = [w for w in self.char_actual_widths.values() if w > 0]
        if not valid_widths:  # If all chars somehow have zero width (bad font?)
            self.atlas_char_cell_width = float(
                self.config.font.size * 0.6
            )  # Default guess
        else:
            self.atlas_char_cell_width = float(max(valid_widths))

        # Ensure atlas_char_cell_width is not zero if all chars were space or similar
        if self.atlas_char_cell_width <= 0:
            self.atlas_char_cell_width = self.config.font.size * 0.5

        cols = int(np.ceil(np.sqrt(len(chars))))
        rows = int(np.ceil(len(chars) / cols))
        atlas_width_px = int(cols * self.atlas_char_cell_width)
        atlas_height_px = int(rows * self.char_height)

        if atlas_width_px == 0 or atlas_height_px == 0:
            self.logger.error(
                "Font atlas dimensions are zero. Cannot proceed with font preprocessing."
            )
            # Potentially raise an error or set a flag that font init failed
            return

        atlas_img = PIL.Image.new(
            "RGBA", (atlas_width_px, atlas_height_px), (0, 0, 0, 0)
        )
        draw = PIL.ImageDraw.Draw(atlas_img)

        for i, char_code in enumerate(chars):
            col = i % cols
            row = i // cols
            x_pil = col * self.atlas_char_cell_width
            y_pil = row * self.char_height

            # Center character horizontally in its atlas cell for better rendering if actual width < cell_width
            actual_w = self._get_char_width(char_code)
            x_offset_for_centering = (self.atlas_char_cell_width - actual_w) / 2.0

            draw.text(
                (x_pil + x_offset_for_centering, y_pil),
                char_code,
                font=self.pil_font,
                fill=self.config.font.color,
            )

            self.char_uvs[char_code] = (
                x_pil / atlas_width_px,
                y_pil / atlas_height_px,
                self.atlas_char_cell_width
                / atlas_width_px,  # UV width is for the whole cell
                self.char_height / atlas_height_px,
            )

        atlas_np = np.array(atlas_img)
        self.font_atlas = (
            torch.from_numpy(atlas_np)
            .permute(2, 0, 1)
            .to(self.device, dtype=self.dtype)
            / 255.0
        )
        self.logger.info(
            f"Font atlas created: {self.font_atlas.shape} on {self.device}. Chars in map: {len(self.char_actual_widths)}"
        )

    def _parse_total_height(self) -> int:
        total_height_setting = self.config.layout.total_height
        if isinstance(total_height_setting, str) and total_height_setting.endswith("%"):
            percentage = float(total_height_setting.rstrip("%"))
            return int(self.config.output.resolution_wh[1] * percentage / 100)
        elif isinstance(total_height_setting, (int, float)):
            return int(total_height_setting)
        return int(self.config.output.resolution_wh[1] * 0.65)

    def _layout_text_block(self, text: str) -> Tuple[List[TextLine], float, float]:
        lines: List[TextLine] = []
        # Use pil_font.getlength here as it's the "source of truth" for layout,
        # _get_text_width using char_actual_widths is for rendering speed later.
        if not text or self.pil_font is None:
            return lines, 0.0, 0.0

        current_y_offset = 0.0
        max_block_width_used = 0.0
        line_height_pixels = (
            self.char_height * self.config.layout.line_spacing_multiplier
        )
        max_render_height_pixels = self._parse_total_height()

        words = text.split(" ")  # Already cleaned in pre-layout phase
        current_line_text = ""

        for i, word in enumerate(words):
            if not current_line_text:
                current_line_text = word
            else:
                test_line_text = current_line_text + " " + word
                test_line_width = self.pil_font.getlength(
                    test_line_text
                )  # Use PIL for layout decisions

                if (
                    self.config.layout.max_width is None
                    or test_line_width <= self.config.layout.max_width
                ):
                    current_line_text = test_line_text
                else:
                    if current_line_text:
                        if (
                            current_y_offset + line_height_pixels
                            <= max_render_height_pixels
                        ):
                            line_actual_width = self.pil_font.getlength(
                                current_line_text
                            )
                            lines.append(
                                TextLine(
                                    current_line_text,
                                    current_y_offset,
                                    line_actual_width,
                                )
                            )
                            max_block_width_used = max(
                                max_block_width_used, line_actual_width
                            )
                            current_y_offset += line_height_pixels
                        else:
                            break
                    current_line_text = word

            if i == len(words) - 1 and current_line_text:
                if current_y_offset + line_height_pixels <= max_render_height_pixels:
                    line_actual_width = self.pil_font.getlength(current_line_text)
                    lines.append(
                        TextLine(current_line_text, current_y_offset, line_actual_width)
                    )
                    max_block_width_used = max(max_block_width_used, line_actual_width)
                    current_y_offset += line_height_pixels

        if (
            not lines
            and text
            and (
                self.config.layout.max_width is None
                or self.pil_font.getlength(text)
                <= (self.config.layout.max_width or float("inf"))
            )
        ):
            if current_y_offset + line_height_pixels <= max_render_height_pixels:
                line_actual_width = self.pil_font.getlength(text)
                lines.append(TextLine(text, current_y_offset, line_actual_width))
                max_block_width_used = max(max_block_width_used, line_actual_width)
                current_y_offset += line_height_pixels

        return lines, max_block_width_used, current_y_offset

    def _calculate_conceptual_typing_end_time(
        self, text_len: int, segment_duration_sec: float
    ) -> float:
        """Helper to calculate when typing animation finishes within a segment."""
        typing_cps = self.config.animation.typing_cps
        segment_end_hold_sec = self.config.animation.segment_end_hold_sec

        available_typing_time = segment_duration_sec - segment_end_hold_sec
        available_typing_time = max(0.01, available_typing_time)

        effective_cps = typing_cps
        if text_len > 0 and typing_cps > 0:
            time_needed_at_set_cps = text_len / typing_cps
            if time_needed_at_set_cps > available_typing_time:
                effective_cps = text_len / available_typing_time

        return (text_len / effective_cps) if effective_cps > 0 else 0.0

    def _calculate_frame_state(
        self,
        segment_content_text_len: int,  # Length of pre-cleaned segment text
        conceptual_typing_end_time_sec: float,  # Pre-calculated
        layout_lines: List[TextLine],
        time_into_segment_sec: float,
    ) -> FrameTextState:
        # Effective CPS was implicitly used to calculate conceptual_typing_end_time_sec
        # Now determine chars_to_show based on that.
        if time_into_segment_sec <= 0:
            chars_to_show = 0
        elif conceptual_typing_end_time_sec == 0:  # No typing, or instant
            chars_to_show = segment_content_text_len if time_into_segment_sec > 0 else 0
        elif time_into_segment_sec >= conceptual_typing_end_time_sec:
            chars_to_show = segment_content_text_len
        else:  # Typing in progress
            # Estimate chars based on proportion of typing time elapsed
            chars_to_show = int(
                segment_content_text_len
                * (time_into_segment_sec / conceptual_typing_end_time_sec)
            )

        chars_to_show = min(chars_to_show, segment_content_text_len)
        is_typing_done = chars_to_show >= segment_content_text_len

        cursor_line_idx = 0
        cursor_char_idx_in_line = 0
        chars_counted = 0
        found_cursor_pos = False

        for line_idx, line in enumerate(layout_lines):
            if chars_counted + len(line.text) >= chars_to_show:
                cursor_line_idx = line_idx
                cursor_char_idx_in_line = chars_to_show - chars_counted
                found_cursor_pos = True
                break
            chars_counted += len(line.text)

        if not found_cursor_pos and layout_lines:
            cursor_line_idx = len(layout_lines) - 1
            cursor_char_idx_in_line = len(layout_lines[-1].text)
        elif not layout_lines:
            cursor_line_idx = 0
            cursor_char_idx_in_line = 0

        return FrameTextState(
            layout_lines,
            chars_to_show,
            cursor_line_idx,
            cursor_char_idx_in_line,
            is_typing_done,
        )

    def _get_visible_lines(self, state: FrameTextState) -> List[TextLine]:
        if state.num_chars_to_display == 0:
            return []

        visible_lines_data: List[TextLine] = []
        chars_shown_count = 0

        for line_obj in state.full_segment_text_lines:
            if chars_shown_count >= state.num_chars_to_display:
                break

            chars_to_take_from_this_line = (
                state.num_chars_to_display - chars_shown_count
            )

            if chars_to_take_from_this_line >= len(line_obj.text):
                visible_lines_data.append(line_obj)
                chars_shown_count += len(line_obj.text)
            else:
                partial_text = line_obj.text[:chars_to_take_from_this_line]
                # OPTIMIZATION: Use pre-calculated char widths sum instead of pil_font.getlength()
                partial_width = self._get_text_width(partial_text)
                visible_lines_data.append(
                    TextLine(partial_text, line_obj.y_offset, partial_width)
                )
                chars_shown_count += len(partial_text)
                break

        return visible_lines_data

    def _render_text_to_tensor(
        self, lines_to_draw: List[TextLine], segment_block_width: float
    ) -> torch.Tensor:
        res_w, res_h = self.config.output.resolution_wh
        frame_tensor = torch.zeros(
            4, res_h, res_w, device=self.device, dtype=self.dtype
        )

        if not lines_to_draw or self.font_atlas is None:
            return frame_tensor

        block_base_x = float(self.config.layout.position_xy[0])
        block_base_y = float(self.config.layout.position_xy[1])

        container_width_for_alignment = (
            self.config.layout.max_width
            if self.config.layout.max_width
            else segment_block_width
        )

        if self.config.layout.max_width is None:
            if self.config.layout.align == "center":
                block_base_x = (res_w - segment_block_width) / 2
            elif self.config.layout.align == "right":
                block_base_x = (
                    res_w - segment_block_width - self.config.layout.position_xy[0]
                )

        atlas_c, atlas_h_px, atlas_w_px = self.font_atlas.shape

        for line_obj in lines_to_draw:
            current_char_render_x = block_base_x

            if self.config.layout.align == "center":
                current_char_render_x += (
                    container_width_for_alignment - line_obj.width
                ) / 2.0
            elif self.config.layout.align == "right":
                current_char_render_x += container_width_for_alignment - line_obj.width

            render_pos_y = block_base_y + line_obj.y_offset

            for char_code in line_obj.text:
                if char_code in self.char_uvs:
                    u, v, cell_uv_w, cell_uv_h = self.char_uvs[
                        char_code
                    ]  # These are for the cell

                    atlas_px_y_start = int(v * atlas_h_px)
                    atlas_px_y_end = int((v + cell_uv_h) * atlas_h_px)
                    atlas_px_x_start = int(u * atlas_w_px)
                    atlas_px_x_end = int((u + cell_uv_w) * atlas_w_px)

                    char_sprite = self.font_atlas[
                        :,
                        atlas_px_y_start:atlas_px_y_end,
                        atlas_px_x_start:atlas_px_x_end,
                    ]
                    # sprite_c, sprite_h, sprite_w = char_sprite.shape # sprite_w is atlas_char_cell_width

                    # Actual width of this character for rendering placement if atlas cells are padded
                    actual_char_w_px = self._get_char_width(char_code)
                    # Sprite from atlas has width self.atlas_char_cell_width. We might only want to use actual_char_w_px part of it.
                    # For simplicity, we render the whole cell and advance by actual_char_w_px.
                    # Or, adjust sprite slicing from atlas if characters are centered in cells.
                    # Current _preprocess_font centers chars in atlas cells. So sprite has correct visual.

                    sprite_h = char_sprite.shape[1]  # Should be self.char_height
                    sprite_w = char_sprite.shape[
                        2
                    ]  # Should be self.atlas_char_cell_width

                    target_y_start = int(round(render_pos_y))
                    target_x_start = int(
                        round(current_char_render_x)
                    )  # Render current char at this x

                    target_y_end = target_y_start + sprite_h
                    target_x_end = (
                        target_x_start + sprite_w
                    )  # Render the full cell_width sprite

                    if (
                        target_x_start < res_w
                        and target_y_start < res_h
                        and target_x_end > 0
                        and target_y_end > 0
                    ):
                        src_x_offset, src_y_offset = 0, 0
                        draw_w, draw_h = sprite_w, sprite_h

                        if target_x_start < 0:
                            src_x_offset = -target_x_start
                            draw_w -= src_x_offset
                            target_x_start = 0
                        if target_y_start < 0:
                            src_y_offset = -target_y_start
                            draw_h -= src_y_offset
                            target_y_start = 0

                        draw_w = min(draw_w, res_w - target_x_start)
                        draw_h = min(draw_h, res_h - target_y_start)

                        if draw_w > 0 and draw_h > 0:
                            char_rgb = char_sprite[
                                :3,
                                src_y_offset : src_y_offset + draw_h,
                                src_x_offset : src_x_offset + draw_w,
                            ]
                            char_alpha = char_sprite[
                                3:4,
                                src_y_offset : src_y_offset + draw_h,
                                src_x_offset : src_x_offset + draw_w,
                            ]

                            current_rgb = frame_tensor[
                                :3,
                                target_y_start : target_y_start + draw_h,
                                target_x_start : target_x_start + draw_w,
                            ]
                            current_alpha = frame_tensor[
                                3:4,
                                target_y_start : target_y_start + draw_h,
                                target_x_start : target_x_start + draw_w,
                            ]

                            # Ensure all tensors for blending are of self.dtype
                            char_rgb = char_rgb.to(self.dtype)
                            char_alpha = char_alpha.to(self.dtype)
                            # current_rgb and current_alpha are already part of frame_tensor (self.dtype)

                            blended_rgb = char_rgb * char_alpha + current_rgb * (
                                1.0 - char_alpha
                            )
                            blended_alpha = char_alpha + current_alpha * (
                                1.0 - char_alpha
                            )

                            frame_tensor[
                                :3,
                                target_y_start : target_y_start + draw_h,
                                target_x_start : target_x_start + draw_w,
                            ] = blended_rgb
                            frame_tensor[
                                3:4,
                                target_y_start : target_y_start + draw_h,
                                target_x_start : target_x_start + draw_w,
                            ] = blended_alpha

                # OPTIMIZATION: Advance by pre-calculated actual width of the character for proportional spacing
                current_char_render_x += self._get_char_width(char_code)
        return frame_tensor

    def _render_cursor(
        self,
        frame_tensor: torch.Tensor,
        frame_state: FrameTextState,
        current_time_sec: float,
        segment_block_width: float,
    ):
        if self.config.animation.cursor.style == "none":
            return frame_tensor

        # Check if cursor should be visible based on blinking
        if self.config.animation.cursor.blink_hz > 0:
            blink_phase = (
                current_time_sec * self.config.animation.cursor.blink_hz
            ) % 1.0
            if blink_phase >= 0.5:
                return frame_tensor

        res_w, res_h = self.config.output.resolution_wh

        # Handle cases where there might be no lines yet (e.g. start of animation)
        if not frame_state.full_segment_text_lines:
            # If no lines but we expect to type, position cursor at the potential start of the first line.
            # This requires knowing the y_offset of the hypothetical first line.
            # For simplicity, if no lines, we can't accurately place cursor unless we pass a default y_offset.
            # Assuming if full_segment_text_lines is empty, it's an empty segment or before first char.
            cursor_line_obj = TextLine("", 0.0, 0.0)  # Dummy line at y_offset 0
            text_before_cursor_on_line = ""
        elif frame_state.cursor_line_idx >= len(frame_state.full_segment_text_lines):
            # This shouldn't happen if frame_state is calculated correctly. Log warning.
            self.logger.warning(
                f"Cursor line index {frame_state.cursor_line_idx} out of bounds for {len(frame_state.full_segment_text_lines)} lines."
            )
            return frame_tensor
        else:
            cursor_line_obj = frame_state.full_segment_text_lines[
                frame_state.cursor_line_idx
            ]
            text_before_cursor_on_line = cursor_line_obj.text[
                : frame_state.cursor_char_idx_in_line
            ]

        # OPTIMIZATION: Use pre-calculated char widths sum instead of pil_font.getlength()
        width_of_text_before_cursor = self._get_text_width(text_before_cursor_on_line)

        block_base_x = float(self.config.layout.position_xy[0])
        block_base_y = float(self.config.layout.position_xy[1])

        container_width_for_alignment = (
            self.config.layout.max_width
            if self.config.layout.max_width
            else segment_block_width
        )

        if self.config.layout.max_width is None:
            if self.config.layout.align == "center":
                block_base_x = (res_w - segment_block_width) / 2.0
            elif self.config.layout.align == "right":
                block_base_x = (
                    res_w - segment_block_width - self.config.layout.position_xy[0]
                )

        line_start_x_aligned = block_base_x
        if self.config.layout.align == "center":
            line_start_x_aligned += (
                container_width_for_alignment - cursor_line_obj.width
            ) / 2.0
        elif self.config.layout.align == "right":
            line_start_x_aligned += (
                container_width_for_alignment - cursor_line_obj.width
            )

        cursor_pixel_x = line_start_x_aligned + width_of_text_before_cursor
        cursor_pixel_y = (
            block_base_y + cursor_line_obj.y_offset
        )  # y_offset is top of line cell

        cursor_color_tensor = (
            torch.tensor(
                self.config.animation.cursor.color, device=self.device, dtype=self.dtype
            )
            / 255.0
        )

        cursor_render_w, cursor_render_h = 0.0, 0.0
        cursor_style = self.config.animation.cursor.style

        # For block/underscore, determine width. Use next char's width or a fallback.
        char_at_cursor_actual_width = self.atlas_char_cell_width  # Fallback width
        if frame_state.cursor_char_idx_in_line < len(cursor_line_obj.text):
            char_at_cursor = cursor_line_obj.text[frame_state.cursor_char_idx_in_line]
            char_at_cursor_actual_width = self._get_char_width(char_at_cursor)
        # If at end of line, block cursor can take width of a space or average char.
        elif frame_state.cursor_char_idx_in_line == len(cursor_line_obj.text):
            char_at_cursor_actual_width = self._get_char_width(
                " "
            )  # Use space width or fallback via _get_char_width

        if cursor_style == "block":
            cursor_render_w, cursor_render_h = (
                char_at_cursor_actual_width,
                self.char_height,
            )
        elif cursor_style == "underscore":
            cursor_render_w = char_at_cursor_actual_width
            cursor_render_h = float(self.config.animation.cursor.width_px)
            cursor_pixel_y += self.char_height - cursor_render_h
        elif cursor_style == "pipe":
            cursor_render_w, cursor_render_h = (
                float(self.config.animation.cursor.width_px),
                self.char_height,
            )

        y_s = int(round(cursor_pixel_y))
        y_e = int(round(cursor_pixel_y + cursor_render_h))
        x_s = int(round(cursor_pixel_x))
        x_e = int(round(cursor_pixel_x + cursor_render_w))

        y_s = max(0, min(y_s, res_h))
        y_e = max(0, min(y_e, res_h))
        x_s = max(0, min(x_s, res_w))
        x_e = max(0, min(x_e, res_w))

        if y_e > y_s and x_e > x_s:
            cursor_rgb = cursor_color_tensor[:3].view(3, 1, 1).to(self.dtype)
            cursor_alpha = cursor_color_tensor[3:4].view(1, 1, 1).to(self.dtype)

            current_rgb = frame_tensor[:3, y_s:y_e, x_s:x_e]  # Already self.dtype
            current_alpha = frame_tensor[3:4, y_s:y_e, x_s:x_e]  # Already self.dtype

            blended_rgb = cursor_rgb * cursor_alpha + current_rgb * (1.0 - cursor_alpha)
            blended_alpha = cursor_alpha + current_alpha * (1.0 - cursor_alpha)

            frame_tensor[:3, y_s:y_e, x_s:x_e] = blended_rgb
            frame_tensor[3:4, y_s:y_e, x_s:x_e] = blended_alpha

        return frame_tensor

    def _process_single_frame_task_on_worker(self, task: WorkerFrameTask) -> bytes:
        segment_data: Optional[PrecomputedSegmentData] = None
        time_into_segment_sec = 0.0

        if task.segment_srt_obj:
            segment_data = self.precomputed_segment_cache.get(
                task.segment_srt_obj.index
            )
            if not segment_data:
                self.logger.warning(
                    f"No precomputed data for segment index {task.segment_srt_obj.index}. This should not happen."
                )
                # Fallback or error handling might be needed here if precomputation failed for a segment
                res_w, res_h = self.config.output.resolution_wh
                empty_tensor = torch.zeros(
                    4, res_h, res_w, device=self.device, dtype=self.dtype
                )
                frame_bytes_hwc = (
                    (empty_tensor[:3] * 255.0).byte().cpu().permute(1, 2, 0)
                )
                return frame_bytes_hwc.numpy().tobytes()

            time_into_segment_sec = (
                task.current_time_sec - task.segment_srt_obj.start.total_seconds()
            )

            frame_state = self._calculate_frame_state(
                len(segment_data.cleaned_text),
                segment_data.conceptual_typing_end_time_sec,
                segment_data.layout_lines,
                time_into_segment_sec,
            )
            visible_lines = self._get_visible_lines(frame_state)

            show_cursor = self.config.animation.cursor.style != "none"
            if show_cursor and self.config.animation.cursor.hide_after_typing_segment:
                if frame_state.is_typing_done_for_segment:
                    # Use pre-calculated conceptual_typing_end_time_sec for cursor hiding logic
                    if (
                        time_into_segment_sec
                        > segment_data.conceptual_typing_end_time_sec
                        + self.config.animation.segment_end_hold_sec
                    ):
                        show_cursor = False

            output_tensor_rgba = self._render_text_to_tensor(
                visible_lines, segment_data.block_width
            )

            if show_cursor:
                output_tensor_rgba = self._render_cursor(
                    output_tensor_rgba,
                    frame_state,
                    task.current_time_sec,
                    segment_data.block_width,
                )

        else:
            res_w, res_h = self.config.output.resolution_wh
            output_tensor_rgba = torch.zeros(
                4, res_h, res_w, device=self.device, dtype=self.dtype
            )

        final_rgb_tensor = output_tensor_rgba[:3, :, :].clamp(0, 1)
        frame_bytes_hwc = (final_rgb_tensor * 255.0).byte().cpu().permute(1, 2, 0)
        return frame_bytes_hwc.numpy().tobytes()

    # ... ( _handle_worker_completion, _ffmpeg_writer_thread_func, _setup_ffmpeg_encoder remain largely the same) ...
    def _handle_worker_completion(
        self, frame_idx: int, future
    ):  # (Identical to previous version)
        if self._processing_failed_event.is_set():
            with self._cache_lock:
                if frame_idx in self._futures_dict:
                    del self._futures_dict[frame_idx]
            return
        try:
            result_bytes = future.result()
            with self._cache_lock:
                self._results_cache[frame_idx] = result_bytes
        except Exception as e:
            self.logger.error(
                f"Frame {frame_idx} processing failed in worker: {e}", exc_info=False
            )
            with self._cache_lock:
                self._results_cache[frame_idx] = e
                self._processing_failed_event.set()
        finally:
            with self._cache_lock:
                if frame_idx in self._futures_dict:
                    del self._futures_dict[frame_idx]

    def _ffmpeg_writer_thread_func(self):  # (Identical to previous version)
        self.logger.info("FFmpeg writer thread started.")
        frames_written_count = 0
        try:
            while not self._ffmpeg_writer_should_stop.is_set() or (
                self._ffmpeg_writer_should_stop.is_set()
                and frames_written_count < self._pbar.total
            ):

                if (
                    self._processing_failed_event.is_set()
                    and frames_written_count == self._last_written_frame_index + 1
                ):
                    if not any(
                        isinstance(self._results_cache.get(i), bytes)
                        for i in range(
                            self._last_written_frame_index + 1, self._pbar.total
                        )
                    ):
                        self.logger.warning(
                            "Processing failed and no more valid frames in cache for FFmpeg writer."
                        )
                        break

                frame_to_write_idx = self._last_written_frame_index + 1
                frame_data = None

                with self._cache_lock:
                    if frame_to_write_idx in self._results_cache:
                        frame_data = self._results_cache.pop(frame_to_write_idx)

                if frame_data is not None:
                    if isinstance(frame_data, bytes):
                        try:
                            if self._ffmpeg_process and self._ffmpeg_process.stdin:
                                self._ffmpeg_process.stdin.write(frame_data)
                                self._last_written_frame_index = frame_to_write_idx
                                frames_written_count += 1
                                if self._pbar:
                                    self._pbar.update(1)
                            else:
                                self.logger.error(
                                    "FFmpeg process or stdin not available for writing."
                                )
                                self._processing_failed_event.set()
                                break
                        except (IOError, BrokenPipeError, ValueError) as e:
                            self.logger.error(
                                f"FFmpeg stdin write error for frame {frame_to_write_idx}: {e}"
                            )
                            self._processing_failed_event.set()
                            break
                    elif isinstance(frame_data, Exception):
                        self.logger.error(
                            f"Skipping frame {frame_to_write_idx} due to render error: {frame_data}"
                        )
                        self._last_written_frame_index = frame_to_write_idx
                        frames_written_count += 1
                        if self._pbar:
                            self._pbar.update(1)

                    time.sleep(0.0001)
                else:
                    if (
                        self._ffmpeg_writer_should_stop.is_set()
                        and not self._results_cache
                    ):
                        is_drained = True
                        with self._cache_lock:
                            if self._futures_dict:
                                is_drained = False
                        if is_drained:
                            break
                    time.sleep(0.005)
        except Exception as e:
            self.logger.error(
                f"FFmpeg writer thread encountered an error: {e}", exc_info=True
            )
            self._processing_failed_event.set()
        finally:
            self.logger.info(
                f"FFmpeg writer thread finished. Total frames written attempt: {frames_written_count}."
            )

    def _setup_ffmpeg_encoder(
        self,
    ) -> Optional[subprocess.Popen]:  # (Identical to previous version)
        res_w, res_h = self.config.output.resolution_wh
        fps = self.config.output.fps
        output_path = self.config.output_file

        if res_w % 2 != 0:
            res_w += 1
        if res_h % 2 != 0:
            res_h += 1

        input_args = [
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{res_w}x{res_h}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-threads",
            str(self.config.ffmpeg_threads),
        ]
        output_args = []
        encoder_to_use = self.config.encoder.lower()

        if encoder_to_use == "prores_ks":
            output_path = (
                output_path.replace(".mp4", ".mov")
                if ".mp4" in output_path
                else output_path + ".mov"
            )
            output_args = [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "3",  # Profile 3 is ProRes 422 HQ, good balance
                "-pix_fmt",
                "yuv422p10le",
            ]
        elif encoder_to_use == "h264_nvenc" and torch.cuda.is_available():
            output_args = [
                "-c:v",
                "h264_nvenc",
                "-preset",
                self.config.preset,
                "-rc",
                "vbr",
                "-cq",
                str(self.config.quality),
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
            ]
        elif encoder_to_use == "libx264":
            output_args = [
                "-c:v",
                "libx264",
                "-preset",
                self.config.preset,
                "-crf",
                str(self.config.quality),
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
            ]
        else:
            self.logger.warning(
                f"Unsupported encoder '{self.config.encoder}', defaulting to libx264."
            )
            output_args = [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
            ]

        ffmpeg_cmd = ["ffmpeg"] + input_args + output_args + [output_path]
        self.logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

        try:
            return subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self.logger.error(
                "FFmpeg executable not found. Please ensure it's in your PATH."
            )
        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg: {e}")
        return None

    def generate(self):
        self.logger.info(f"Starting animation generation: {self.config.output_file}")
        start_time = time.time()

        try:
            with open(self.config.srt_file_path, "r", encoding="utf-8") as f:
                srt_content = f.read()
            segments = list(srt.parse(srt_content))
            if not segments:
                self.logger.error("No segments found in SRT file.")
                return None
            self.logger.info(f"Parsed {len(segments)} SRT segments.")

            # OPTIMIZATION: Pre-layout and pre-calculate segment data in parallel
            self.logger.info("Starting parallel pre-computation of segment data...")

            # Group segments by unique content to avoid redundant layout
            unique_content_to_segments: Dict[str, List[srt.Subtitle]] = {}
            for seg in segments:
                # Important: Use original newlines for layout, then clean for typing length.
                # _layout_text_block expects newlines to be handled by its logic (splits by space).
                # For conceptual_typing_end_time, we need length of displayable text.
                content_for_layout = seg.content.replace("\r\n", " ").replace(
                    "\n", " "
                )  # Spaces for layout
                if content_for_layout not in unique_content_to_segments:
                    unique_content_to_segments[content_for_layout] = []
                unique_content_to_segments[content_for_layout].append(seg)

            # Temporary executor for pre-computation
            with ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="PrecompWorker"
            ) as precomp_executor:
                futures_to_content = {}
                for (
                    content_key,
                    srt_objs_for_content,
                ) in unique_content_to_segments.items():
                    # All srt_objs_for_content under same content_key will share layout
                    # But duration and conceptual_typing_end_time might differ if srt_objs have different timings for same text

                    # Layout is based on content_key
                    future_layout = precomp_executor.submit(
                        self._layout_text_block, content_key.strip()
                    )  # Strip for layout
                    futures_to_content[future_layout] = (
                        content_key,
                        srt_objs_for_content,
                    )

                for future_layout in futures_to_content:
                    content_key, srt_objs_for_content = futures_to_content[
                        future_layout
                    ]
                    try:
                        lines, block_w, block_h = future_layout.result()

                        for srt_obj in srt_objs_for_content:
                            cleaned_text_for_typing = (
                                content_key.strip()
                            )  # For length calculation
                            seg_duration = (srt_obj.end - srt_obj.start).total_seconds()
                            conceptual_typing_end = (
                                self._calculate_conceptual_typing_end_time(
                                    len(cleaned_text_for_typing), seg_duration
                                )
                            )
                            self.precomputed_segment_cache[srt_obj.index] = (
                                PrecomputedSegmentData(
                                    cleaned_text=cleaned_text_for_typing,
                                    duration_sec=seg_duration,
                                    conceptual_typing_end_time_sec=conceptual_typing_end,
                                    layout_lines=lines,
                                    block_width=block_w,
                                    block_height=block_h,
                                )
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to pre-compute for content '{content_key[:50]}...': {e}",
                            exc_info=True,
                        )
                        self._processing_failed_event.set()
                        # For robustness, could add dummy PrecomputedSegmentData for these srt_obj.index

            if self._processing_failed_event.is_set():
                self.logger.error("Pre-computation failed. Aborting generation.")
                return None
            self.logger.info(
                f"Pre-computation complete. Cached data for {len(self.precomputed_segment_cache)} segment occurrences."
            )

            self._ffmpeg_process = self._setup_ffmpeg_encoder()
            if not self._ffmpeg_process:
                return None

            self._render_executor = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="RenderWorker"
            )

            total_duration_sec = segments[-1].end.total_seconds()
            total_frames = int(total_duration_sec * self.config.output.fps)
            self.logger.info(f"Total frames to render: {total_frames}")
            if total_frames == 0:
                self.logger.warning("Total frames is 0, nothing to render.")
                return (
                    self.config.output_file
                )  # Or None if an empty file is not desired

            self._pbar = tqdm(total=total_frames, desc="Rendering frames", unit="frame")

            self._ffmpeg_writer_thread = Thread(
                target=self._ffmpeg_writer_thread_func, name="FFmpegWriter"
            )
            self._ffmpeg_writer_thread.start()

            current_segment_idx = 0
            for frame_idx in range(total_frames):
                if self._processing_failed_event.is_set():
                    self.logger.warning(
                        f"Processing failed event set. Stopping frame submission at frame {frame_idx}."
                    )
                    break

                while (
                    len(self._results_cache) + len(self._futures_dict)
                    >= self.results_cache_limit
                    and not self._processing_failed_event.is_set()
                ):
                    time.sleep(0.01)
                    if self._processing_failed_event.is_set():
                        break
                if self._processing_failed_event.is_set():
                    break

                current_time_sec = frame_idx / self.config.output.fps
                current_timedelta = timedelta(seconds=current_time_sec)

                active_segment_obj: Optional[srt.Subtitle] = None
                while (
                    current_segment_idx < len(segments)
                    and segments[current_segment_idx].end <= current_timedelta
                ):  # Use <= to ensure segment is chosen if time is exactly end
                    current_segment_idx += 1

                if (
                    current_segment_idx < len(segments)
                    and segments[current_segment_idx].start
                    <= current_timedelta
                    < segments[current_segment_idx].end
                ):
                    active_segment_obj = segments[current_segment_idx]

                worker_task = WorkerFrameTask(
                    frame_idx, current_time_sec, active_segment_obj
                )

                future = self._render_executor.submit(
                    self._process_single_frame_task_on_worker, worker_task
                )
                with self._cache_lock:
                    self._futures_dict[frame_idx] = future
                future.add_done_callback(
                    lambda f, idx=frame_idx: self._handle_worker_completion(idx, f)
                )

            self.logger.info("All frame tasks submitted. Waiting for completion...")
            # Refined waiting logic
            while not self._processing_failed_event.is_set():
                with self._cache_lock:
                    # Check if all submitted futures are done and all results written by FFmpeg writer
                    if (
                        not self._futures_dict
                        and self._last_written_frame_index >= total_frames - 1
                    ):
                        break  # All done
                    # Check if all futures are done but FFmpeg writer might still be working or stalled
                    if (
                        not self._futures_dict
                        and not self._results_cache
                        and self._last_written_frame_index < total_frames - 1
                    ):
                        # If FFmpeg writer is dead, then we are stuck
                        if (
                            self._ffmpeg_writer_thread
                            and not self._ffmpeg_writer_thread.is_alive()
                        ):
                            self.logger.error(
                                "FFmpeg writer thread died before completion."
                            )
                            self._processing_failed_event.set()  # Signal failure
                            break
                        # Otherwise, FFmpeg writer is just slow or idle, keep waiting for it.
                time.sleep(0.2)

            if self._processing_failed_event.is_set():
                self.logger.error("Generation process encountered an error.")
            else:
                self.logger.info(
                    "All render tasks and FFmpeg writing seem complete based on tracking."
                )

        except Exception as e:
            self.logger.error(f"Unhandled exception in generate(): {e}", exc_info=True)
            self._processing_failed_event.set()
        finally:
            self.logger.info("Initiating cleanup...")

            self._ffmpeg_writer_should_stop.set()
            if self._ffmpeg_writer_thread and self._ffmpeg_writer_thread.is_alive():
                self.logger.info("Waiting for FFmpeg writer thread to finish...")
                self._ffmpeg_writer_thread.join(timeout=30)
                if self._ffmpeg_writer_thread.is_alive():
                    self.logger.warning(
                        "FFmpeg writer thread did not terminate gracefully."
                    )

            if self._render_executor:
                self.logger.info("Shutting down render executor...")
                self._render_executor.shutdown(
                    wait=not self._processing_failed_event.is_set(),
                    cancel_futures=self._processing_failed_event.is_set(),
                )

            if self._pbar:
                final_pbar_val = self._last_written_frame_index + 1
                if (
                    final_pbar_val < self._pbar.total
                    and not self._processing_failed_event.is_set()
                ):
                    # If we exited loop early but no error, update pbar to total assuming success
                    # Or, if errored, pbar reflects what was processed.
                    self.logger.info(
                        f"Progress bar might not reflect total frames due to early exit/error. Last written: {self._last_written_frame_index}"
                    )
                self._pbar.n = min(final_pbar_val, self._pbar.total)  # Cap at total
                self._pbar.refresh()
                self._pbar.close()

            if self._ffmpeg_process:
                self.logger.info("Closing FFmpeg process...")
                if self._ffmpeg_process.stdin:
                    try:
                        self._ffmpeg_process.stdin.close()
                    except Exception as e_stdin:
                        self.logger.warning(f"Error closing FFmpeg stdin: {e_stdin}")

                try:
                    stdout, stderr = self._ffmpeg_process.communicate(timeout=25)
                    if self._ffmpeg_process.returncode == 0:
                        self.logger.info("FFmpeg process completed successfully.")
                    else:
                        self.logger.error(
                            f"FFmpeg process exited with error code {self._ffmpeg_process.returncode}."
                        )
                        self.logger.error(
                            f"FFmpeg stdout:\n{stdout.decode(errors='ignore') if stdout else 'None'}"
                        )
                        self.logger.error(
                            f"FFmpeg stderr:\n{stderr.decode(errors='ignore') if stderr else 'None'}"
                        )
                        self._processing_failed_event.set()
                except subprocess.TimeoutExpired:
                    self.logger.warning("FFmpeg process timed out. Forcing kill.")
                    self._ffmpeg_process.kill()
                    stdout, stderr = self._ffmpeg_process.communicate()
                except Exception as e_comm:
                    self.logger.error(f"Error during FFmpeg communicate: {e_comm}")

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            end_time = time.time()
            self.logger.info(
                f"Animation generation finished in {end_time - start_time:.2f} seconds."
            )

            if self._processing_failed_event.is_set():
                self.logger.error("Typewriter animation generation FAILED.")
                return None

            # Check if output file exists and has size, basic sanity check
            final_output_path = self.config.output_file
            if self.config.encoder.lower() == "prores_ks":
                final_output_path = (
                    final_output_path.replace(".mp4", ".mov")
                    if ".mp4" in final_output_path
                    else final_output_path + ".mov"
                )

            if (
                os.path.exists(final_output_path)
                and os.path.getsize(final_output_path) > 100
            ):  # 100 bytes as a tiny threshold
                self.logger.info(f"Output video: {final_output_path}")
                return final_output_path
            else:
                self.logger.error(
                    f"Output file {final_output_path} not found or is empty. FFmpeg likely failed."
                )
                return None
