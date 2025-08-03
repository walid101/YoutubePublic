import os
import cv2
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from typing import Union, List, Tuple
from moviepy.editor import ImageClip, VideoFileClip, CompositeVideoClip


class VideoSource:
    """
    Professional video overlay processor with advanced background removal.
    """

    def __init__(self, video: Union[Path, str, VideoFileClip] = None):
        if isinstance(video, (Path, str)):
            self.video = VideoFileClip(str(video))
        elif isinstance(video, VideoFileClip):
            self.video = video
        else:
            self.video = None
        self.supported_formats = [".mp4", ".mov", ".avi"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save(self, output_path: Union[Path, str]):
        if not self.video:
            raise ValueError("No video loaded. Cannot save.")

        output_path = Path(output_path)
        file_extension = output_path.suffix.lower()
        if file_extension not in self.supported_formats:
            raise ValueError(
                f"Unsupported format: {file_extension}. Supported formats: {self.supported_formats}"
            )

        self.video.write_videofile(
            str(output_path), codec="libx264", audio_codec="aac", threads=4
        )

    def overlay(
        self,
        bg: Union[VideoFileClip, str],
        fg: Union[VideoFileClip, str],
        timestamp: Tuple[Union[int, str, float], Union[int, str, float]] = (0, 0),
        color_index: List[int] = None,
        color_dist: int = 0,
        color_match: bool = True,
        brightness_threshold: int = 0,
    ):
        """
        Overlay a foreground VideoFileClip onto a background VideoFileClip with optional chroma keying and brightness filtering.

        Args:
            bg: Background Video can be either a VideoFileClip or a filepath to one
            fg: Foreground Video can be either a VideoFileClip or a filepath to one
            timestamp: Tuple of (start, end) times in seconds or "HH:MM:SS" format.
                    If not provided, will use (0, background_duration)
            color_index: RGB values to use as chroma key [R,G,B], or None if no chroma keying
            tolerance: Tolerance for color matching in chroma key
            brightness_threshold: Minimum brightness value a pixel in the foreground must have, or 0 if no thresholding

        Returns:
            VideoFileClip: Processed video with overlay applied
        """
        if not isinstance(bg, VideoFileClip):
            bg = VideoFileClip(bg)

        if not isinstance(fg, VideoFileClip):
            fg = VideoFileClip(fg)

        if timestamp == (0, 0):  # Check for the default timestamp
            timestamp = (0, bg.duration)

        # Convert timestamps
        start_time = self.convert_time(timestamp[0], bg.duration)
        end_time = self.convert_time(timestamp[1], bg.duration)

        bg_clip = bg.subclip(start_time, end_time)

        if fg.size != bg.size:
            fg = fg.resize(bg.size)

        def create_mask(frame):
            if len(frame.shape) == 3:
                frame_rgb = frame

                if color_index is not None:
                    color_diff = frame_rgb - np.array(color_index)
                    color_distance = np.linalg.norm(
                        color_diff, axis=2
                    )  # Euclidean distance
                    color_mask = color_distance < color_dist
                    if not color_match:
                        color_mask = ~color_mask
                else:
                    color_mask = np.ones(frame_rgb.shape[:2], dtype=bool)

                if brightness_threshold > 0:
                    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                    brightness_mask = frame_gray >= brightness_threshold
                else:
                    brightness_mask = np.ones(frame_rgb.shape[:2], dtype=bool)

                # Combine both masks: pixel must pass both the chroma key mask and brightness threshold
                final_mask = color_mask & brightness_mask

                return final_mask.astype("float32")
            return np.ones(frame.shape[:2], dtype="float32")

        fg = fg.set_duration(end_time - start_time)
        mask_clip = fg.fl_image(create_mask)
        mask_clip.ismask = True

        masked_clip = fg.set_mask(mask_clip)
        masked_clip = masked_clip.set_position(("center", "center"))

        # Composite videos
        final_clip = CompositeVideoClip([bg_clip, masked_clip], size=bg_clip.size)
        final_clip = final_clip.set_duration(end_time - start_time)

        return final_clip

    def torch_overlay(
        self,
        bg: Union[VideoFileClip, str],
        fg: Union[VideoFileClip, str],
        timestamp: Tuple[Union[int, str, float], Union[int, str, float]] = (0, 0),
        color_index: List[int] = None,
        color_dist: int = 0,
        color_match: bool = True,
        brightness_threshold: int = 0,
        sharpen: bool = True,
    ):
        """
        Overlay a foreground VideoFileClip onto a background VideoFileClip with optional chroma keying and brightness filtering.
        """
        if not isinstance(bg, VideoFileClip):
            bg = VideoFileClip(bg)

        if not isinstance(fg, VideoFileClip):
            fg = VideoFileClip(fg)

        if timestamp == (0, 0):
            timestamp = (0, bg.duration)

        start_time = self.convert_time(timestamp[0], bg.duration)
        end_time = self.convert_time(timestamp[1], bg.duration)

        bg_clip = bg.subclip(start_time, end_time)

        if fg.size != bg.size:
            fg = fg.resize(bg.size)

        def create_mask(frame):
            if len(frame.shape) == 3:
                frame_tensor = torch.from_numpy(frame).float()
                if torch.cuda.is_available():
                    frame_tensor = frame_tensor.cuda()

                # Handle color keying
                if color_index is not None:
                    color_ref = torch.tensor(color_index, dtype=torch.float32)
                    if torch.cuda.is_available():
                        color_ref = color_ref.cuda()

                    # Calculate color difference per pixel
                    color_diff = torch.abs(frame_tensor - color_ref)
                    color_distance = torch.sum(
                        color_diff, dim=2
                    )  # Sum across color channels
                    color_mask = color_distance < color_dist
                    if not color_match:
                        color_mask = ~color_mask
                else:
                    color_mask = torch.ones(frame_tensor.shape[:2], dtype=torch.bool)
                    if torch.cuda.is_available():
                        color_mask = color_mask.cuda()

                if brightness_threshold > 0:
                    brightness = torch.mean(frame_tensor, dim=2)
                    brightness_mask = brightness >= brightness_threshold
                else:
                    brightness_mask = torch.ones(
                        frame_tensor.shape[:2], dtype=torch.bool
                    )
                    if torch.cuda.is_available():
                        brightness_mask = brightness_mask.cuda()

                final_mask = color_mask & brightness_mask

                if sharpen:
                    conv_input = frame_tensor.permute(2, 0, 1).unsqueeze(0)
                    sharpened_frame = self.sharpen(conv_input)
                    frame_tensor = sharpened_frame.squeeze(0).permute(1, 2, 0)

                final_mask = final_mask.cpu().numpy().astype(np.float32)

                return final_mask  # Return a 2D mask without expanding to 3 channels
            return np.ones(
                frame.shape[:2], dtype=np.float32
            )  # Return a 2D mask by default

        fg = fg.set_duration(end_time - start_time)

        mask_clip = fg.fl_image(create_mask)
        mask_clip.ismask = True

        masked_clip = fg.set_mask(mask_clip)
        masked_clip = masked_clip.set_position(("center", "center"))

        final_clip = CompositeVideoClip([bg_clip, masked_clip], size=bg_clip.size)
        final_clip = final_clip.set_duration(end_time - start_time)

        return final_clip

    def sharpen(self, frame_tensor):
        """
        Apply sharpening to the input tensor using a 3x3 kernel.
        Expects input shape: [batch, channels, height, width]
        """
        base_kernel = torch.tensor(
            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=torch.float32
        )

        if torch.cuda.is_available():
            base_kernel = base_kernel.cuda()
            frame_tensor = frame_tensor.cuda()

        in_channels = frame_tensor.size(1)  # Should be 3 for RGB
        sharpening_kernel = base_kernel.unsqueeze(0).unsqueeze(0)
        sharpening_kernel = sharpening_kernel.repeat(in_channels, 1, 1, 1)
        sharpened_frame = F.conv2d(
            frame_tensor, sharpening_kernel, padding=1, groups=in_channels
        )
        sharpened_frame = torch.clamp(sharpened_frame, 0, 255)
        return sharpened_frame

    @staticmethod
    def convert_img_to_vid(
        input_path, output_path, duration=5, fps=60, resize_height=None
    ):
        image_clip = ImageClip(input_path)
        image_clip = image_clip.set_duration(duration)

        if resize_height:
            image_clip = image_clip.resize(
                height=resize_height
            )  # Resize while keeping aspect ratio

        image_clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
        )

    def convert_time(
        self, time_input: Union[str, int, float], media_duration: float
    ) -> float:
        """
        Converts time formats to seconds.

        Args:
            time_input: Time in seconds or "HH:MM:SS" format
            media_duration: Total duration of the video in seconds

        Returns:
            float: Time in seconds

        Raises:
            ValueError: If time format is invalid or outside video duration
        """
        if isinstance(time_input, (int, float)):
            if not 0 <= time_input <= media_duration:
                raise ValueError(f"Time {time_input} outside valid range")
            return float(time_input)

        try:
            hours, minutes, seconds = map(int, time_input.split(":"))
            total_seconds = hours * 3600 + minutes * 60 + seconds
            if not 0 <= total_seconds <= media_duration:
                raise ValueError(f"Time {time_input} outside valid range")
            return float(total_seconds)
        except:
            raise ValueError(f"Invalid time format: {time_input}")

    def convert_exr_to_mp4(exr_folder, output_video_path, fps=90):
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        exr_files = sorted([f for f in os.listdir(exr_folder) if f.endswith(".exr")])

        if not exr_files:
            print("No EXR files found in the specified directory.")
            return

        first_exr = cv2.imread(
            os.path.join(exr_folder, exr_files[0]), cv2.IMREAD_UNCHANGED
        )
        height, width = first_exr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'XVID' for .avi files
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for exr_file in exr_files:
            img = cv2.imread(os.path.join(exr_folder, exr_file), cv2.IMREAD_UNCHANGED)
            img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_normalized = np.uint8(img_normalized)
            out.write(img_normalized)

        # Release the VideoWriter object
        out.release()
        print(f"Video saved as {output_video_path}")
