import logging
import math
import os
from pathlib import Path
import re
from abc import abstractmethod
import subprocess
import tempfile
from typing import List, Optional, Tuple, Union
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
)


class AudioSource:
    """
    Audio Source is an abstract class that defines all aspects of an audio source in a video.
    """

    def __init__(self, audio: Union[Path, str] = None):
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        if audio:
            self.audio = AudioFileClip(
                str(Path(audio) if isinstance(audio, str) else audio)
            )

    @abstractmethod
    def get_audio_metadata():
        """
        Return feature set of this audio class object.
        """
        pass

    def get_audio(self):
        """
        Return audio clip as AudioFileClip.
        """
        return self.audio

    @staticmethod
    def overlay(
        media_source: Union[str, VideoFileClip],
        audio_source: Union[str, AudioFileClip],
        output_path: Optional[Union[Path, str]] = None,
        start_time: Union[str, float] = 0,
        end_time: Optional[Union[str, float]] = None,
        volume: float = 0.5,
    ) -> str:
        temp_files_to_clean = []

        _final_output_path_str: str
        if output_path is None:
            if not isinstance(media_source, str):
                raise ValueError(
                    "output_path must be specified for non-string media_source or non-in-place operations."
                )
            _final_output_path_str = media_source
        else:
            _final_output_path_str = str(output_path)

        final_dir = os.path.dirname(_final_output_path_str)
        if final_dir and not os.path.exists(final_dir):
            os.makedirs(final_dir, exist_ok=True)

        final_basename = os.path.basename(_final_output_path_str)
        base_name, ext = os.path.splitext(final_basename)
        _temp_ffmpeg_output_path_str = os.path.join(
            final_dir if final_dir else ".",
            f"{base_name}_ffmpeg_temp_{os.urandom(4).hex()}{ext}",
        )
        temp_files_to_clean.append(_temp_ffmpeg_output_path_str)

        actual_media_path: str
        if isinstance(media_source, str):
            actual_media_path = media_source
        elif isinstance(media_source, VideoFileClip):
            # Ensure moviepy is actually available if its objects are used
            if "VideoFileClip" in str(type(media_source)) and not hasattr(
                media_source, "write_videofile"
            ):
                raise ImportError(
                    "moviepy.editor.VideoFileClip used but MoviePy seems not fully imported or object is a dummy."
                )
            with tempfile.NamedTemporaryFile(
                suffix="_rs_media.mp4", delete=False
            ) as tmp_f:
                temp_media_path = tmp_f.name
            media_source.write_videofile(temp_media_path, codec="libx264", audio_codec="aac")  # type: ignore
            actual_media_path = temp_media_path
            temp_files_to_clean.append(actual_media_path)
        else:
            raise TypeError(f"Unsupported media_source type: {type(media_source)}")

        actual_audio_path: str
        if isinstance(audio_source, str):
            actual_audio_path = audio_source
        elif isinstance(audio_source, AudioFileClip):
            if "AudioFileClip" in str(type(audio_source)) and not hasattr(
                audio_source, "write_audiofile"
            ):
                raise ImportError(
                    "moviepy.editor.AudioFileClip used but MoviePy seems not fully imported or object is a dummy."
                )
            with tempfile.NamedTemporaryFile(
                suffix="_rs_audio.wav", delete=False
            ) as tmp_f:
                temp_audio_path = tmp_f.name
            audio_source.write_audiofile(temp_audio_path)  # type: ignore
            actual_audio_path = temp_audio_path
            temp_files_to_clean.append(actual_audio_path)
        else:
            raise TypeError(f"Unsupported audio_source type: {type(audio_source)}")

        if not os.path.exists(actual_media_path):
            raise FileNotFoundError(
                f"Processed media source not found: {actual_media_path}"
            )
        if not os.path.exists(actual_audio_path):
            raise FileNotFoundError(
                f"Processed audio source not found: {actual_audio_path}"
            )

        def get_duration(file_path: str) -> float:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                file_path,
            ]
            try:
                duration_str = subprocess.run(
                    cmd, check=True, capture_output=True, text=True
                ).stdout.strip()
                duration = float(duration_str)
                if duration <= 0:
                    raise ValueError(f"Duration non-positive ({duration}s).")
                return duration
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"ffprobe failed for {file_path}: {e.stderr.decode() if e.stderr else str(e)}"
                )
            except ValueError as e:
                raise RuntimeError(f"Could not parse duration for {file_path}: {e}")
            except Exception as e:
                raise RuntimeError(f"Could not determine duration for {file_path}: {e}")

        video_duration = get_duration(actual_media_path)
        new_audio_duration = get_duration(actual_audio_path)

        def parse_time(time_val: Union[str, float], context: str) -> float:
            if isinstance(time_val, str):
                match = re.match(
                    r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$", time_val
                )
                if match:
                    h_str, m_str, s_str = match.groups()
                    h = int(h_str) if h_str else 0
                    m = int(m_str)
                    s = float(s_str)
                    if not (0 <= m < 60 and 0 <= s < 60):
                        raise ValueError(
                            f"Invalid {context} time component in '{time_val}'."
                        )
                    return h * 3600 + m * 60 + s
                try:
                    return float(time_val)
                except ValueError:
                    raise ValueError(f"{context} time '{time_val}' format error.")
            return float(time_val)

        start_seconds = parse_time(start_time, "Start")
        if start_seconds < 0:
            start_seconds = 0.0
        if start_seconds >= video_duration:
            raise ValueError(
                f"Start time {start_seconds:.3f}s is at or after video duration {video_duration:.3f}s."
            )

        looping_required = False
        if end_time is not None:
            end_seconds = parse_time(end_time, "End")
            if end_seconds <= start_seconds:
                raise ValueError(
                    f"End time {end_seconds:.3f}s must be after start time {start_seconds:.3f}s."
                )
            effective_end_seconds = min(end_seconds, video_duration)
            new_audio_play_duration = effective_end_seconds - start_seconds
            if new_audio_play_duration > new_audio_duration:
                looping_required = True
        else:
            max_possible_play_duration = video_duration - start_seconds
            new_audio_play_duration = min(
                new_audio_duration, max_possible_play_duration
            )

        if new_audio_play_duration <= 0:
            if _final_output_path_str != actual_media_path:
                import shutil

                shutil.copy2(actual_media_path, _final_output_path_str)
            return _final_output_path_str

        probe_cmd_has_audio = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            actual_media_path,
        ]
        try:
            probe_result = subprocess.run(
                probe_cmd_has_audio, check=False, capture_output=True, text=True
            )
            video_has_original_audio = (
                probe_result.returncode == 0 and probe_result.stdout.strip() == "audio"
            )
        except Exception:
            video_has_original_audio = False

        delay_ms_str = f"{int(start_seconds * 1000)}"
        filter_complex_parts = []

        if not looping_required:
            filter_complex_parts.append(
                f"[1:a]volume={volume:.3f},atrim=duration={new_audio_play_duration:.3f}[fg_audio_trimmed]"
            )
            filter_complex_parts.append(
                f"[fg_audio_trimmed]adelay={delay_ms_str}|{delay_ms_str}[fg_audio_final]"
            )
        else:
            filter_complex_parts.append(f"[1:a]volume={volume:.3f}[vol_adj_audio]")
            filter_complex_parts.append(
                f"[vol_adj_audio]atrim=duration={new_audio_duration:.3f}[loop_unit]"
            )
            filter_complex_parts.append(
                f"[loop_unit]aloop=loop=-1:size=2147483647[infinite_loop]"
            )
            filter_complex_parts.append(
                f"[infinite_loop]adelay={delay_ms_str}|{delay_ms_str}[delayed_loop]"
            )
            filter_complex_parts.append(
                f"[delayed_loop]atrim=duration={new_audio_play_duration:.3f}[fg_audio_final]"
            )

        if video_has_original_audio:
            filter_complex_parts.append(
                f"[0:a][fg_audio_final]amix=inputs=2:duration=longest:normalize=1[a_out]"
            )
        else:
            filter_complex_parts.append(f"[fg_audio_final]asetpts=PTS-STARTPTS[a_out]")

        filter_complex = ";".join(filter_complex_parts)

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i",
            actual_media_path,
            "-i",
            actual_audio_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[a_out]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            _temp_ffmpeg_output_path_str,
        ]

        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)

            try:
                os.replace(_temp_ffmpeg_output_path_str, _final_output_path_str)
            except (
                OSError
            ):  # Fallback for os.replace, e.g. different filesystems (though unlikely here)
                import shutil

                shutil.move(_temp_ffmpeg_output_path_str, _final_output_path_str)

            if _temp_ffmpeg_output_path_str in temp_files_to_clean:
                temp_files_to_clean.remove(_temp_ffmpeg_output_path_str)
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"FFmpeg command failed (ret {e.returncode}):\nCmd: {' '.join(e.cmd)}\n"
            )
            if e.stderr:
                error_msg += f"Stderr: {e.stderr}\n"
            raise RuntimeError(error_msg)
        finally:
            for temp_file_path in temp_files_to_clean:
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception:
                        pass

        return _final_output_path_str

    def convert_time(
        self, time_str: Union[str, int, float], media_duration: float
    ) -> float:
        """
        Validates the time string in "HH:mm:ss" format or accepts an integer
        representing seconds and converts it to seconds.

        Parameters:
        time_str (Union[str, int]): Time in "HH:mm:ss" format or an integer in seconds.
        media_duration (float): The duration of the media file in seconds.

        Returns:
        float: The time in seconds.

        Raises:
        ValueError: If the time format is invalid or exceeds the media duration.
        """
        if time_str is None:
            return None
        # If the input is already an integer or float, return it as is
        if isinstance(time_str, (int, float)):
            if time_str < 0:
                raise ValueError("Time in seconds cannot be negative.")
            if time_str > media_duration:
                raise ValueError(
                    f"Time {time_str} exceeds the media duration of {media_duration} seconds."
                )
            return float(time_str)

        # Regex to match the format HH:mm:ss
        time_format = re.compile(r"^\d{2}:\d{2}:\d{2}$")

        if not time_format.match(time_str):
            raise ValueError(f"Time {time_str} is not in 'HH:mm:ss' format.")

        # Convert the time to seconds
        hours, minutes, seconds = map(int, time_str.split(":"))
        total_seconds = hours * 3600 + minutes * 60 + seconds

        if total_seconds > media_duration:
            raise ValueError(
                f"Time {time_str} exceeds the media duration of {media_duration} seconds."
            )

        return total_seconds
