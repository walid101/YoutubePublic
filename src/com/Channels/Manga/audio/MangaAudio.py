import os
import numpy as np
import srt
import torch
import logging
import librosa
from scipy import signal
import soundfile as sf
from moviepy.editor import AudioFileClip
from pathlib import Path
from typing import Callable, Iterable, List, Union
from datetime import timedelta
from rapidfuzz.fuzz import ratio
from ozkit.src.com.network.Network import Network
from ozkit.src.com.Utils import Utils
from ozkai_tts.AbstractTTSModel import AbstractTTSModel
from ozkai_tts.AbstractRemoteTTSModel import AbstractRemoteTTSModel
from ozkai_tts.WHISPERSRTModel import WHISPERSRTModel
from ozkai_tts.WHISPERXSRTModel import WHISPERXSRTModel
from src.com.channels.manga.datamodels.models import Timestamp


class MangaAudio:
    def __init__(
        self,
        tts: Union[AbstractTTSModel, AbstractRemoteTTSModel],
        whisper: Union[WHISPERSRTModel, WHISPERXSRTModel],
    ):
        self.tts = tts
        self.whisper = whisper
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @Network.proxy
    def construct(
        self,
        dialogue: List[str],
        filepath: str,
        speed: float = 1.0,
        time_between: float = 0.33,
    ) -> str:
        """
        Construct the video audio via. Eleven Labs.

        @params
        model_id: the elvn labs model id to generate tts. \n
        voice: the voice key to generate tts. \n
        dialogue: the dialogue for tts. SPLITS BY IDX. \n
        filepath: file to save tts. \n
        """
        if isinstance(self.tts, AbstractTTSModel):
            subtitles = self.tts.chunk_tts(
                text_segments=dialogue,
                filepath=filepath,
                speed=speed,
                time_between=time_between,
            )
            return subtitles
        else:
            self.tts.chunk_tts(input_segments=dialogue, filepath=filepath, speed=speed)
            return None  # Subtitles no available for remote tts models.

    def get_subtitles(
        self,
        audio_path: str,
        output_path: str,
        ref_text: str,
    ) -> str:
        if isinstance(self.whisper, WHISPERSRTModel):
            self.whisper.generate_srt_from_tts(
                audio_path=audio_path,
                output_path=output_path,
                text=ref_text,
                word_timestamps=False,
                fp16=True,  # Custom
                vad="silero:v3.1",
                refine_whisper_precision=0.02,  # Custom
            )
            self.logger.info(f"Successfully generated subtitles at {output_path}")
        else:
            kwargs = {
                "language": "en",
                "batch_size": 8,
                "vad_filter": True,
                "vad_parameters": {"threshold": 0.5},
            }
            self.whisper.generate_srt_from_tts(
                audio_path=audio_path, output_path=output_path, text=ref_text, **kwargs
            )
        return output_path

    def get_subtitles_by_word(
        self, audio_path: str, output_path: str, ref_text: str
    ) -> str:
        if isinstance(self.whisper, WHISPERSRTModel):
            self.whisper.generate_word_level_srt(
                audio_path=audio_path,
                output_path=output_path,
                text=ref_text,
                refine_whisper_precision=0.5,
            )
        else:
            kwargs = {
                "language": "en",
                "batch_size": 8,
                "vad_filter": True,
                "vad_parameters": {"threshold": 0.5},
            }
            self.whisper.generate_word_level_srt(
                audio_path=audio_path, output_path=output_path, text=ref_text, **kwargs
            )
        self.logger.info(
            f"Successfully generated subtitles at {output_path} by word level"
        )
        return output_path

    @staticmethod
    def srt_time_to_seconds(timecode: str) -> float:
        hh_mm_ss, ms = timecode.split(",")
        hh, mm, ss = map(int, hh_mm_ss.split(":"))
        return hh * 3600 + mm * 60 + ss + float("0." + ms)

    def align_srt_by_reference(
        self, subtitle: Union[Path, str], reference: List[str], max_lookahead: int = 100
    ):
        """
        Aligns SRT subtitles to reference sentences by merging subtitles optimally.

        Args:
            subtitle: Path to SRT file
            reference: List of reference sentences to align with
            max_lookahead: Maximum number of subsequent subtitles to check for merging.
                           Limits the inner loop search depth for performance. (Default: 50)
        """
        try:
            with open(subtitle, "r", encoding="utf-8") as file:
                raw_srt = file.read()
        except FileNotFoundError:
            self.logger.error(f"Subtitle file not found: {subtitle}")
            return []
        except Exception as e:
            self.logger.error(f"Error reading subtitle file {subtitle}: {e}")
            return []

        try:
            srt_list = list(srt.parse(raw_srt))
            if not srt_list:
                self.logger.warning(
                    f"Subtitle file {subtitle} appears empty or invalid."
                )
                return []
        except Exception as e:
            self.logger.error(f"Error parsing SRT file {subtitle}: {e}")
            return []

        srt_pointer = 0
        ref_pointer = 0
        processed_srt = []

        cleaned_references = [Utils.clean(ref) for ref in reference]

        while ref_pointer < len(reference) and srt_pointer < len(srt_list):
            best_similarity = -1.0
            best_idx = srt_pointer

            search_limit_idx = min(len(srt_list) + 1, srt_pointer + 1 + max_lookahead)

            current_combined = ""

            for idx in range(srt_pointer + 1, search_limit_idx):
                if idx > srt_pointer + 1:
                    if current_combined and srt_list[idx - 1].content:
                        current_combined += " "
                    current_combined += srt_list[idx - 1].content
                else:
                    current_combined = srt_list[srt_pointer].content

                target_ref_cleaned = cleaned_references[ref_pointer]
                current_combined_cleaned = Utils.clean(current_combined)

                if not current_combined_cleaned or not target_ref_cleaned:
                    current_sim = 0.0
                else:
                    current_sim = ratio(current_combined_cleaned, target_ref_cleaned)

                if current_sim > best_similarity:
                    best_similarity = current_sim
                    best_idx = idx

            if best_similarity > -1.0 and best_idx > srt_pointer:
                optimal = srt_list[srt_pointer:best_idx]
                combined_sub = MangaAudio.combine_subtitles(optimal)
                if combined_sub:
                    processed_srt.append(combined_sub)
                    self.logger.info(f"SUBTITLE: {combined_sub.content}")
                    srt_pointer = best_idx
                    ref_pointer += 1
                else:
                    self.logger.warning(
                        f"Failed to combine subtitles {srt_pointer}-{best_idx-1}. Advancing SRT pointer by 1."
                    )
                    srt_pointer += 1
                    ref_pointer += 1

            else:
                self.logger.debug(
                    f"Fallback: Processing single subtitle {srt_pointer} for ref {ref_pointer}"
                )
                combined_sub = MangaAudio.combine_subtitles([srt_list[srt_pointer]])
                if combined_sub:
                    processed_srt.append(combined_sub)
                else:
                    self.logger.warning(
                        f"Failed to process single subtitle {srt_pointer}."
                    )

                srt_pointer += 1
                ref_pointer += 1

        if srt_pointer < len(srt_list):
            self.logger.warning(
                f"Finished alignment with {len(srt_list) - srt_pointer} unused subtitles remaining."
            )
        if ref_pointer < len(reference):
            self.logger.warning(
                f"Finished alignment with {len(reference) - ref_pointer} unused references remaining."
            )

        return processed_srt

    def merge_subtitles_by_gap(
        self,
        subtitles_input: Union[
            Path, str, Iterable[srt.Subtitle]
        ],  # Changed input name and type
        minimal_gap: float = 0.2,
    ) -> List[srt.Subtitle]:
        """
        Merges adjacent subtitles if the time gap between them is <= minimal_gap.
        Accepts either a path to an SRT file (str or Path) or an iterable of srt.Subtitle objects.
        Assumes the input list (if provided or parsed) is sorted by start time. Re-indexes the output.
        """
        subtitles: List[srt.Subtitle] = []  # Initialize empty list

        # --- Input Handling ---
        if isinstance(subtitles_input, (str, Path)):
            # Input is a file path
            try:
                with open(subtitles_input, "r", encoding="utf-8") as f:
                    raw_srt = f.read()
                # Use srt.parse to get the list of Subtitle objects
                # Ensure srt.parse handles potential errors or returns an empty list
                subtitles = list(srt.parse(raw_srt))
            except FileNotFoundError:
                MangaAudio.logger.error(f"Subtitle file not found: {subtitles_input}")
                return []  # Return empty list on error
            except Exception as e:
                MangaAudio.logger.error(
                    f"Error reading or parsing SRT file {subtitles_input}: {e}"
                )
                return []  # Return empty list on error
        elif isinstance(subtitles_input, Iterable):
            # Assume it's already an iterable of Subtitle objects (like a list)
            # We convert to list to ensure we can index it; filter out non-Subtitle objects
            subtitles = [
                sub for sub in subtitles_input if isinstance(sub, srt.Subtitle)
            ]
            if (
                not subtitles and subtitles_input
            ):  # Check if filtering removed everything
                MangaAudio.logger.warning(
                    "Input iterable contained no valid srt.Subtitle objects."
                )
        else:
            MangaAudio.logger.error(
                f"Invalid input type for subtitles_input: {type(subtitles_input)}"
            )
            return []

        # --- Merging Logic (applied to the 'subtitles' list) ---
        if not subtitles:
            return []  # Return empty if no valid subtitles were found/parsed

        # Ensure subtitles are sorted by start time, as merging logic depends on it
        subtitles.sort(key=lambda x: x.start)

        minimal_gap_td = timedelta(seconds=minimal_gap)
        merged_srt = []

        # Start with a copy of the first subtitle to avoid modifying originals if input was a list
        current_sub = srt.Subtitle(
            index=subtitles[0].index,  # Index will be corrected later
            start=subtitles[0].start,
            end=subtitles[0].end,
            content=subtitles[0].content,
            proprietary=subtitles[0].proprietary,
        )

        for i in range(1, len(subtitles)):
            next_sub = subtitles[i]
            gap = next_sub.start - current_sub.end

            if gap >= timedelta(0) and gap <= minimal_gap_td:
                # Merge: Combine content and update end time
                # Clean content during merge for cleaner output
                combined_content = f"{Utils.clean(current_sub.content)} {Utils.clean(next_sub.content)}"
                current_sub.end = next_sub.end
                current_sub.content = combined_content  # clean already strips
            else:
                # Finalize the previous subtitle and start a new one
                # Only add if content is not empty after cleaning
                if Utils.clean(current_sub.content):
                    merged_srt.append(current_sub)
                # Start new current_sub as a copy of next_sub
                current_sub = srt.Subtitle(
                    index=next_sub.index,
                    start=next_sub.start,
                    end=next_sub.end,
                    content=next_sub.content,
                    proprietary=next_sub.proprietary,
                )

        # Append the last subtitle being processed, checking content
        if Utils.clean(current_sub.content):
            merged_srt.append(current_sub)

        # Re-index the final list sequentially
        for i, sub in enumerate(merged_srt):
            sub.index = i + 1

        return merged_srt

    @staticmethod
    def parse_srt(srt_path: Union[str, Path]) -> List[Timestamp]:
        timestamps = []
        current_block = {"index": None, "time": None, "text": []}

        with open(srt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:  # Empty line marks end of subtitle block
                    if current_block["time"]:
                        start_str, end_str = current_block["time"].split(" --> ")
                        start = MangaAudio.srt_time_to_seconds(start_str)
                        end = MangaAudio.srt_time_to_seconds(end_str)
                        timestamps.append(
                            {"start": start, "end": end, "duration": end - start}
                        )
                    current_block = {"index": None, "time": None, "text": []}
                    continue

                if current_block["index"] is None:  # Index line
                    current_block["index"] = line
                elif current_block["time"] is None:  # Timestamp line
                    current_block["time"] = line
                else:  # Text content
                    current_block["text"].append(line)

        return timestamps

    @staticmethod
    def combine_subtitles(subtitles: List[srt.Subtitle]) -> srt.Subtitle:
        subtitles_sorted = sorted(subtitles, key=lambda x: x.start)
        combined_text = " ".join([subtitle.content for subtitle in subtitles_sorted])
        return srt.Subtitle(
            index=subtitles_sorted[0].index,
            start=subtitles_sorted[0].start,
            end=subtitles_sorted[-1].end,
            content=combined_text,
        )

    @staticmethod
    def split_subtitle(subtitle: srt.Subtitle, prefix: str) -> List[srt.Subtitle]:
        """
        Splits subtitle on a prefix string into prefix and postfix parts with proportional timing.
        Ensures valid subtitle durations and content.
        """
        # Early return for empty prefix or non-matching content
        if not prefix or not subtitle.content.startswith(prefix):
            return [subtitle]

        # Calculate total duration in microseconds accurately
        total_duration = float((subtitle.end - subtitle.start).total_seconds() * 1e6)
        total_length = len(subtitle.content)

        # Guard against division by zero and empty content
        if total_length == 0:
            return [subtitle]

        prefix_length = len(prefix)
        postfix_length = total_length - prefix_length

        # Prevent invalid split when prefix equals content
        if postfix_length == 0:
            return [subtitle]

        # Calculate proportional timing (ensure minimum duration)
        prefix_duration = int(total_duration * prefix_length / total_length)
        postfix_duration = total_duration - prefix_duration

        # Validate durations before creating subtitles
        if prefix_duration <= 0 or postfix_duration <= 0:
            return [subtitle]

        # Create prefix subtitle with valid timing
        prefix_end = subtitle.start + timedelta(microseconds=prefix_duration)
        prefix_sub = srt.Subtitle(
            index=subtitle.index, start=subtitle.start, end=prefix_end, content=prefix
        )

        # Create postfix content without stripping whitespace
        postfix_content = subtitle.content[len(prefix) :]

        # Validate postfix content
        if not postfix_content.strip():
            return [prefix_sub]

        # Create postfix subtitle with valid timing
        postfix_sub = srt.Subtitle(
            index=subtitle.index + 1,
            start=prefix_end,
            end=subtitle.end,
            content=postfix_content,
        )

        # Final validation of subtitle durations
        if prefix_sub.end <= prefix_sub.start or postfix_sub.end <= postfix_sub.start:
            return [subtitle]

        return [prefix_sub, postfix_sub]

    @staticmethod
    def convert_subtitles_to_srt(subtitles: List[srt.Subtitle], save_path: str):
        """
        Converts list of Subtitles to .srt File

        Args:
            subtitles (List[srt.Subtitle]): List of subtitle objects to convert
            save_path (str): Path where the .srt file should be saved

        Returns:
            None

        Raises:
            IOError: If there's an error writing to the specified path
            ValueError: If the subtitles list is empty or contains invalid entries
        """
        if not subtitles:
            raise ValueError("Subtitles list cannot be empty")

        if not all(isinstance(sub, srt.Subtitle) for sub in subtitles):
            raise ValueError("All entries must be valid Subtitle objects")

        try:
            sorted_subtitles = sorted(subtitles, key=lambda x: x.start)
            srt_content = srt.compose(sorted_subtitles)

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(srt_content)

            print(f"Saved subtitles to {save_path}")

        except IOError as e:
            raise IOError(f"Error writing to file {save_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during SRT conversion: {str(e)}")

    def resample(
        self, filepath: str, start_sample: int, end_sample: int, output_path: str = None
    ) -> str:
        """
        Resamples audio from a media file, changing the sample rate from start_sample to end_sample.

        Args:
            filepath: Path to the input audio or video file (mp4, mp3, wav, etc.)
            start_sample: Original sample rate
            end_sample: Target sample rate for resampling
            output_path: Path to save the resampled audio/video. If None, creates path based on input file

        Returns:
            str: Path to the resampled file

        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If start_sample >= end_sample or other invalid parameters
            RuntimeError: If resampling operation fails
        """

        if not os.path.exists(filepath):
            self.logger.error(f"Input file not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")

        if start_sample >= end_sample:
            self.logger.error(
                f"Invalid sample rates: start_sample {start_sample} >= end_sample {end_sample}"
            )
            raise ValueError("start_sample must be less than end_sample")

        # Generate output path if not provided
        if output_path is None:
            basename = os.path.splitext(os.path.basename(filepath))[0]
            output_path = f"{basename}_resampled_{start_sample}_to_{end_sample}.wav"

        is_video = filepath.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        output_is_video = output_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

        try:
            import subprocess

            if is_video:
                if output_is_video:
                    # For video to video resampling, use ffmpeg directly
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-i",
                        filepath,
                        "-af",
                        f"aresample=osr={end_sample}",  # Audio filter to resample
                        "-c:v",
                        "copy",  # Copy video stream without re-encoding
                        "-y",  # Overwrite output if exists
                        output_path,
                    ]

                    process = subprocess.run(
                        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )

                    if process.returncode != 0:
                        error_msg = process.stderr.decode()
                        self.logger.error(f"FFmpeg failed: {error_msg}")
                        raise RuntimeError(f"Video resampling failed: {error_msg}")

                    self.logger.info(f"Successfully resampled video to {output_path}")
                    return output_path

                else:
                    # Video to audio resampling
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-i",
                        filepath,
                        "-vn",  # No video
                        "-ar",
                        str(end_sample),  # Set output sample rate
                        "-y",  # Overwrite output if exists
                        output_path,
                    ]

                    process = subprocess.run(
                        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )

                    if process.returncode != 0:
                        error_msg = process.stderr.decode()
                        self.logger.error(f"FFmpeg failed: {error_msg}")
                        raise RuntimeError(
                            f"Audio extraction and resampling failed: {error_msg}"
                        )

                    self.logger.info(
                        f"Successfully extracted and resampled audio to {output_path}"
                    )
                    return output_path

            else:
                # Audio file processing
                # Load the audio file
                y, sr = librosa.load(filepath, sr=start_sample)

                # Resample to target sample rate
                y_resampled = librosa.resample(
                    y, orig_sr=start_sample, target_sr=end_sample
                )

                # Save resampled audio
                sf.write(output_path, y_resampled, end_sample, format="WAV")

                self.logger.info(f"Successfully resampled audio to {output_path}")
                return output_path

        except Exception as e:
            self.logger.error(f"Error during resampling: {str(e)}")
            raise RuntimeError(f"Resampling failed: {str(e)}")

    def enhance_high_frequency(self, input_path: str, output_path: str = None) -> str:
        """
        Enhances speech clarity by applying targeted EQ boosts to high frequencies.

        Args:
            input_path: Path to the input audio file
            output_path: Path to save the enhanced audio. If None, creates path based on input file

        Returns:
            str: Path to the enhanced audio file

        Raises:
            FileNotFoundError: If the input file doesn't exist
            RuntimeError: If processing fails
        """

        if not os.path.exists(input_path):
            self.logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"File not found: {input_path}")

        # Generate output path if not provided
        if output_path is None:
            basename = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"{basename}_enhanced_eq.wav"

        try:
            # Load the audio file
            y, sr = librosa.load(input_path, sr=None)

            # Design the EQ filters
            # First filter: 3dB boost at 3-4kHz with medium Q
            center_freq1 = 3500  # 3.5kHz (middle of 3-4kHz range)
            Q1 = 1.2  # Medium Q factor
            gain1 = 3.0  # 3dB boost

            # Second filter: 2dB shelf boost starting at 6kHz
            shelf_freq = 6000  # 6kHz
            gain2 = 2.0  # 2dB boost

            # Convert to filter coefficients (biquad filters)
            # Peaking EQ filter
            b1, a1 = signal.iirpeak(center_freq1 / (sr / 2), Q1, gain1)

            # High shelf filter
            b2, a2 = signal.iirfilter(
                2,
                shelf_freq / (sr / 2),
                btype="highshelf",
                ftype="butter",
                fs=sr,
                output="ba",
                gain=gain2,
            )

            # Apply the filters sequentially
            y_eq1 = signal.lfilter(b1, a1, y)
            y_eq = signal.lfilter(b2, a2, y_eq1)

            # Normalize to prevent clipping (optional, adjust as needed)
            if np.max(np.abs(y_eq)) > 0.98:
                y_eq = y_eq / np.max(np.abs(y_eq)) * 0.98

            # Save the enhanced audio
            sf.write(output_path, y_eq, sr, format="WAV")

            self.logger.info(f"Successfully enhanced high frequencies in {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error during high frequency enhancement: {str(e)}")
            raise RuntimeError(f"Audio processing failed: {str(e)}")

    def apply_multiband_clarity_compression(
        self, input_path: str, output_path: str = None
    ) -> str:
        """
        Applies multiband compression focused on the 2-8kHz range to enhance clarity.

        Args:
            input_path: Path to the input audio file
            output_path: Path to save the compressed audio. If None, creates path based on input file

        Returns:
            str: Path to the compressed audio file

        Raises:
            FileNotFoundError: If the input file doesn't exist
            RuntimeError: If processing fails
        """

        if not os.path.exists(input_path):
            self.logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"File not found: {input_path}")

        # Generate output path if not provided
        if output_path is None:
            basename = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"{basename}_clarity_compressed.wav"

        try:
            # Load the audio file
            y, sr = librosa.load(input_path, sr=None)

            # Design bandpass filter for 2-8kHz range
            low_freq = 2000  # 2kHz
            high_freq = 8000  # 8kHz

            # Create bandpass filter
            sos = signal.butter(
                4,
                [low_freq / (sr / 2), high_freq / (sr / 2)],
                btype="bandpass",
                output="sos",
            )

            # Split signal into bands
            mid_high_band = signal.sosfilt(sos, y)
            rest_of_signal = y - mid_high_band

            # Compression parameters
            threshold = 0.15
            ratio = 2.0  # 2:1 compression ratio
            attack_time = 0.008  # 8ms
            release_time = 0.065  # 65ms

            # Convert time constants to samples
            attack_samples = int(attack_time * sr)
            release_samples = int(release_time * sr)

            # Prepare compression
            gain_reduction = np.ones_like(mid_high_band)
            env = np.zeros_like(mid_high_band)

            # Simple envelope follower for compression
            for i in range(1, len(mid_high_band)):
                # Calculate instantaneous level
                level = abs(mid_high_band[i])

                # Attack/release envelope
                if level > env[i - 1]:
                    # Attack phase
                    env[i] = env[i - 1] + (level - env[i - 1]) / attack_samples
                else:
                    # Release phase
                    env[i] = env[i - 1] + (level - env[i - 1]) / release_samples

                # Apply compression if above threshold
                if env[i] > threshold:
                    # Calculate gain reduction (in linear scale)
                    gain_reduction[i] = threshold + (env[i] - threshold) / ratio
                    gain_reduction[i] /= env[i]
                else:
                    gain_reduction[i] = 1.0

            # Apply compression to mid-high band only
            compressed_band = mid_high_band * gain_reduction

            # Make-up gain (to compensate for gain reduction)
            makeup_gain = 1.2  # Small makeup gain to bring out the details
            compressed_band *= makeup_gain

            # Recombine with the rest of the signal
            y_compressed = compressed_band + rest_of_signal

            # Normalize to prevent clipping
            if np.max(np.abs(y_compressed)) > 0.98:
                y_compressed = y_compressed / np.max(np.abs(y_compressed)) * 0.98

            # Save the compressed audio
            sf.write(output_path, y_compressed, sr, format="WAV")

            self.logger.info(
                f"Successfully applied clarity compression in {output_path}"
            )
            return output_path

        except Exception as e:
            self.logger.error(f"Error during clarity compression: {str(e)}")
            raise RuntimeError(f"Audio processing failed: {str(e)}")
