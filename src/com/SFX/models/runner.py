import logging
import time

from src.com.sfx.models.TypeWriter import *


def run_typewriter():
    config = AnimationConfig(
        srt_file_path=r"H:\Youtube\Videos\Manga\production\Almark\Almark_8_8_aligned_srt.srt",
        output_file="typewriter_animation3_faster.mov",  # MOV for alpha channel
        font=FontSettings(path="fonts/arial.ttf", size=64, color=(255, 255, 255, 255)),
        layout=LayoutSettings(position_xy=(100, 100), align="left", max_width=800),
        animation=AnimationParams(
            typing_cps=20.0, cursor=CursorSettings(style="pipe", blink_hz=1.5)
        ),
        output=FrameOutputSettings(resolution_wh=(1920, 1080), fps=60),
        max_workers=6,
        use_fp16=True,
        encoder="prores_4444",  # For alpha channel support
    )

    animator = TypeWriterAnimator(config)
    result = animator.generate()
    print(f"Generated: {result}")


run_typewriter()
