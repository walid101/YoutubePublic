from .MangaAudio import MangaAudio
from ozkai_tts.KKROTTSModel import KKROTTSModel


def run_manga_audio():
    audio = MangaAudio()
    audio.construct(
        model_id="eleven_flash_v2_5",
        voice="JBFqnCBsd6RMkjVDRZzb",
        dialogue="A monstrous blue canine with glowing red eyes sinks its teeth into a person's arm.",
        filepath=r"D:\Projects\Youtube\src\com\channels\Manga\audio\tts\tts.mp3",
    )


def run_kkro():
    name = "onyx"
    model_path = r"D:\Models\TTS\Kokoro-82M\kokoro-v1_0.pth"
    config_path = r"D:\Models\TTS\Kokoro-82M\config.json"
    voice_path = (
        r"D:\Models\TTS\Kokoro-82M\voices\am_"
        + name
        + r".pt"  # (Adam) Eric is the anime voice
    )
    filepath = f"H:\Youtube\Videos\KKRO\output_{name}.mp3"
    kkro = KKROTTSModel(model=model_path, config=config_path)
    kkro.kkro_tts(
        text="Hello is this kokoro speaking? If so say Hi!",
        voice=voice_path,
        filepath=filepath,
    )


def test_clean():
    print(
        "CLEANED: ",
        MangaAudio.clean(
            'Inside, Nat sighs, "Again?" as Shiggy observes, "Yeah, she started again."'
        ),
    )


test_clean()
# run_kkro()
