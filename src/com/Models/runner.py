import os
import pandas as pd
from moviepy.editor import VideoFileClip
from src.com.models.VideoSource import VideoSource
from src.com.models.AudioSource import AudioSource
from ozkmcro.src.com.youtube.youtube import Youtube
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
import cv2


def aud_overlay():
    aud = AudioSource()

    combined_aud_clip = aud.overlay(
        media_source=r"D:\Projects\Youtube\src\com\channels\Music\bgs\0717.mp4",
        audio_source=r"D:\Projects\Youtube\src\com\channels\Music\clips\Light Years Away Main Ext.mp3",
    )

    combined_aud_clip = aud.overlay(
        combined_aud_clip,
        audio_source=r"D:\Projects\Youtube\gentle_rain\rain_sounds_dark.mp3",
        volume=0.3,
    )

    # Create the output directory if it doesn't exist
    output_directory = "output2"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Specify the output file path
    output_file_path = os.path.join(output_directory, "example.mp4")

    # Write the modified video to a file
    combined_aud_clip.write_videofile(
        output_file_path,
        codec="libx264",
        audio_codec="aac",
    )


def video_overlay():
    vid = VideoSource()
    combined_vid_clip = None

    # Apply the video overlay
    combined_vid_clip = vid.torch_overlay(
        bg=r"D:\Projects\Youtube\output\trimmed.mp4",  # r"D:\Projects\Youtube\outputImageRain\output.mp4",
        fg=r"E:\SFX\freezy\vecteezy_rain-effect-overlay-stock-footage_3407210.mp4",
        brightness_threshold=10,
    )

    # Create the output directory if it doesn't exist
    output_directory = "D:\Projects\Youtube\outputSharpImageRain"
    os.makedirs(output_directory, exist_ok=True)

    # Specify the output file path
    output_file_path = os.path.join(output_directory, "rain_overlayed_10.mp4")

    # Write the video file with specific parameters
    combined_vid_clip.write_videofile(
        output_file_path,
        threads=16,
        codec="libx264",  # Change to a faster codec like libx264
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
        verbose=True,
        fps=combined_vid_clip.fps,
        audio=combined_vid_clip.audio is not None,
        bitrate="1000k",  # Lower the bitrate for faster processing (adjust as needed)
    )

    print(f"Video successfully saved to: {output_file_path}")


def temp():
    yt = Youtube()
    yt.get_ytaudio_by_timestamp(
        yt_url="https://www.youtube.com/watch?v=q76bMs-NwRk",
        output_path="gentle_rain",
        timestamps=[("00:00:00", "00:04:00")],
    )


def res():
    # Define the input path for the video
    input_path = r"E:\SFX\freezy\vecteezy_rain-effect-overlay-stock-footage_3407210.mp4"

    # Load the video
    video = VideoFileClip(input_path)

    # Trim the video to get the segment from 5 to 15 seconds
    trimmed = video.subclip(5, 15)

    # Initialize an array to accumulate pixel intensities
    hist_accumulator = np.zeros((256,))
    hist_raw = np.zeros((256,))

    # Analyze each frame in the 10-second segment
    for frame in trimmed.iter_frames(fps=24, dtype="uint8"):
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Calculate histogram for the current frame
        histogram = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist_accumulator += histogram.flatten()
        hist_raw += histogram.flatten()  # Save raw values for printing

    # Normalize histogram (optional for the plot)
    hist_normalized = hist_accumulator / hist_accumulator.sum()

    # Plot the combined normalized histogram
    plt.figure(figsize=(10, 6))
    plt.plot(hist_normalized, color="blue")
    plt.title("Grayscale Histogram (5-15 seconds)")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Normalized Frequency")
    plt.grid(True)
    plt.show()

    # Convert raw frequencies to percentages
    total_pixels = hist_raw.sum()
    hist_percentage = (hist_raw / total_pixels) * 100

    # Generate a table of pixel intensities and their frequencies in percentages
    pixel_values = np.arange(256)
    frequency_table = pd.DataFrame(
        {"Pixel Intensity": pixel_values, "Frequency (%)": hist_percentage}
    )

    # Print the table (frequencies in percentage)
    print(frequency_table.to_string(index=False))

    # Save the table to a file in the current working directory
    output_file = "frequency_table_percentage.txt"
    frequency_table.to_csv(output_file, sep="\t", index=False)
    print(f"\nFrequency table saved to '{output_file}'")

    # Return the table if needed
    return frequency_table


def main():
    video_overlay()
    # res()
    # from PIL import Image

    # # Create a black image with dimensions 1920x1080
    # image = Image.new("RGB", (1920, 1080), color="black")

    # # Save the image
    # image.save("black_image.jpg")

    # output_dir = r"D:\Projects\Youtube\outputImageRain"
    # output_file = os.path.join(output_dir, "output.mp4")

    # # Ensure the output directory exists
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # VideoSource.convert_img_to_vid(
    #     input_path="black_image.jpg",
    #     output_path=output_file,
    # )


if __name__ == "__main__":
    main()
