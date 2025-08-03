## Run
Composer
1. python -m src.com.channels.manga.composer.runner

2. yes y | python -m src.com.channels.manga.composer.runner composer --manga-title "The Corporate Slave Sword Master Becomes a Broadcaster" --manga-id "d4841c9b-5239-42e9-b452-87f1b5b50ab5" --chapter-range "1-6.4"

3. yes y | python -m src.com.channels.manga.composer.runner composer --manga-title "The Rising of the Commoner Origin Officer Beat Up All the Incompetent Noble Superiors!" --chapter-range "1-13"

Mangacon
1. python -m src.com.channels.manga.composer.runner mangacon --title "Gachiakuta" --chapter-range "1-20" --manga-id "6037/gachiakuta" --provider "mangapill"

2. python -m src.com.channels.manga.composer.runner mangacon --title "Sword God from the Ruined World" --chapter-range "1-20" --provider mangahere --manga-id "sword_god_from_the_ruined_world" --provider "mangahere"

Title
1.  python -m src.com.channels.manga.composer.runner title --manga-title "Any Highly Advanced Medicine Is Indistinguishable From Magic" --chapter-range "1-10"

Thumbnail
1. python -m src.com.channels.manga.composer.runner thumbnail --manga-title "You're So Sloppy, Hotta-sensei" --chapter-range "1-10"

2. Editor - https://clippingmagic.com/

## TODO's

1. ~~Add characters to Manga Meta, this should include all characters (limit 50.) => limits need to call Jikan API.~~ 

~~1.5. Add characters to Chapter meta. As in all character references from all previous chapters AND this chapter. Via. LLM! This must be a preprocess step. (FROM MANGA COMPOSER)~~

~~1.6. Use Chapter meta characters are default character list when summarizing panels within that chapter.~~

~~2.5. Investigate if there is a reason why the panel segmenter segments a full panel in half?~~

6. ~~Add page images to input.~~

7. ~~Add Magi Segmenter to Manga Composer.~~
### TODO
04/02/25
1. ~~Use Magi Character identifyer and textbox mapper as input to LLM (Helps LLM keep naming and dialogue consistent). Exp. Time: 1 day.~~

04/14/25
TODO:
2. WHEN ALIGNING SUBTITLES MAKE SURE TO TAKE OUT ANY INNER QUOTATIONS, ex - 
~~"Riko excitedly bursts in, \"Yap! It's started!\"" <- What the LLM Generated~~
vs
~~"Rike excitedly bursts in, Yap! It's started <- what is in the subtitles from Whisper~~
^ This can make a descrpancy.

3.~~ [MangaAudio]: In our ElevenLabs TTS: The issue with chunking and then combining is that the speaker Adds a pause when coming to the end of one sentence (so if we end it in between a sentence we will get a pause there for no reason).~~
# TODO - 04/18/25

~~1. Over-reliance at times on panel text to be the narrative content (finetune this.) -> Temp solution check.~~

~~2. Still sometimes inconsistent assigning of Text bubbles to characters -> Temp solution check. ~~

07/20/25:
- Investigate Orpheus TTS Options - 
https://colab.research.google.com/drive/1_FQ41NbrnUbYwmR93fLVZYuqYtXOhWsF

3. 05/23/25
- We need to make our manga and page fetching MORE reliable. We are doing this by revamping the Manga and MangaCon classes. We are currently at the stage where we can populate UP to page meta data. However when we try to fetch the page image we get an error due to these websites requiring a direct referrer from their OWN website. Consumet can already do this, it knows the correct headers to pass such that we get images back. We simply need to pass that to the PageDataModel.fetch_page_image(...), etc. Follow this - 
Okay, to enable passing specific headers (which you might get from Consumet's /read endpoint) when fetching images to potentially solve the 403 errors, we need to modify the classes/methods involved in the image fetching pipeline.
Your Network.safe_request is already capable of accepting headers via **kwargs. The changes will be in how MangaCon and your data models handle and pass these headers down to Network.safe_request.
Here are the methods/classes that will need changes:
MangaCon.get_pages:
Reason: This is where you parse the response from the Consumet /read endpoint. If this response includes a headers object for each page image, get_pages is the place to extract it.
Change: When creating PageDataModel instances, you'll pass these extracted headers to the PageDataModel constructor.
PageDataModel.__init__:
Reason: To store the headers received from MangaCon.get_pages.
Change: Add a new parameter (e.g., request_headers: Optional[Dict[str, str]] = None) to the constructor and store it as an instance attribute (e.g., self.request_headers).
PageDataModel methods that trigger image downloads (e.g., fetch_page_image, save_page_image):
Reason: These methods are responsible for initiating the download of the image associated with the page. They need to use the stored self.request_headers.
Change: When these methods call the underlying image fetching/saving logic (which might be in ImageModel or directly call Network.safe_request), they need to pass self.request_headers along.
ImageModel methods that call Network.safe_request (e.g., fetch_image, save_image, fetch_image_static):
Reason: These are the methods that directly interact with Network.safe_request. They need to be able to accept a headers argument and pass it through.
Change:
If they are instance methods (like fetch_image, save_image in your ImageModel context), they should be modified to accept an optional headers: Optional[Dict[str, str]] = None parameter. Inside, they would pass this headers to Network.safe_request.
If they are staticmethods (like fetch_image_static), they also need to be modified to accept an optional headers parameter.
Let's illustrate with the key parts of MangaCon.get_pages and how it would flow down (conceptually, as I don't have your exact ImageModel and PageDataModel fetching methods here, but I'll use the ones you provided as a base):





# WHISPER EFFICIENCY PARAMS:
To enhance both efficiency and accuracy, you can adjust the following parameters in the provided transcribe_timestamped function. Here's a breakdown of what you can modify:

1. FP16 (Floating Point Precision)
Parameter: fp16

Change: Set fp16=True to use half-precision (FP16) for faster processing with minimal loss in accuracy. If the model is running on a supported GPU (especially one with tensor cores), this will speed up the process and may still yield good accuracy. If accuracy is paramount, you might leave it as None or False to use full precision.

2. VAD (Voice Activity Detection)
Parameter: vad

Change: Set vad=True or use a more sophisticated method (like "silero:3.1"). This will help reduce hallucinations and improve accuracy by removing non-speech segments (silence). It can make the transcription more focused on actual speech.

3. Naive Approach
Parameter: naive_approach

Change: Set naive_approach=True if you need to decode twice: once for transcription and once for alignment. This can improve accuracy, especially in the word alignment, but it will be slower. If the performance is acceptable and accuracy is paramount, enable this.

4. Word Alignment
Parameter: word_alignment_most_top_layers

Change: You can experiment with increasing word_alignment_most_top_layers from 6 to a higher value (e.g., 8-12) to improve the granularity of word-level timestamps and alignment accuracy. Be cautious, as this can increase computation time.

5. Refining Whisper Precision
Parameter: refine_whisper_precision

Change: Set this to a value like 0.02 or 0.04 to refine the segment positions. Lower values will make the alignment more accurate but also more time-consuming. You may want to tweak this depending on the precision needed.

6. Minimum Word Duration
Parameter: min_word_duration

Change: Set this to a lower value (e.g., 0.02) to help ensure that shorter words aren't discarded. This can help if you want more precise word-level transcriptions.

7. Sampling Temperature
Parameter: temperature

Change: Set temperature=0.0 for deterministic results (which could help in ensuring more consistent outputs). If you need variability or randomness in the transcription, increase it slightly (e.g., 0.2 or 0.4).

8. Best of and Beam Size
Parameters: best_of, beam_size

Change: If you’re using beam search, set beam_size=5 or higher, and best_of=5. This will enhance the accuracy by exploring more potential candidates but can reduce speed. If accuracy is the primary goal, these should be tuned accordingly.

9. Disfluency Detection
Parameter: detect_disfluencies

Change: Set this to True if you want the model to detect filler words, repetitions, or other disfluencies. This can help in producing a cleaner and more accurate transcription.

10. Temperature Sampling (Randomness)
Parameter: temperature

Change: Set temperature=0.0 for deterministic transcriptions (no randomness), which can ensure higher accuracy, especially for critical tasks where randomness is less desirable.

11. Include Punctuation in Confidence
Parameter: include_punctuation_in_confidence

Change: Set this to True if you want the punctuation’s confidence score to be included in the word confidence calculation. This could give you more accurate overall word confidence.

12. Trust Whisper Timestamps
Parameter: trust_whisper_timestamps

Change: Set trust_whisper_timestamps=True if you want to use Whisper's pre-computed timestamps. This can improve speed but may impact accuracy in certain cases. If you want maximum accuracy, you might opt for recalculating timestamps manually or with higher precision.

13. Backend Timestamps
Parameter: use_backend_timestamps

Change: Set this to True to use the timestamps provided by the backend model (OpenAI Whisper or transformers), which can sometimes be more accurate than those computed by heuristics.

14. Segment Compression and No-Speech Thresholds
Parameter: compression_ratio_threshold, no_speech_threshold

Change: Tweak these thresholds based on your audio quality. For instance, increasing compression_ratio_threshold can help filter out bad-quality transcriptions.



### MANGAS
05/21/25
- https://mangadex.org/chapter/14eee2ed-da07-4604-9101-50213c4cb6ab/15 (Used)
- https://mangadex.org/title/202718c6-525e-4cec-a469-c945548cacc3/anta-to-osananajimitte-dake-demo-iyananoni-zekkou-kara-hajimaru-s-kyuu-bishoujo-to-no-gakuen (Used)
- https://mangadex.org/title/a967d6e2-e269-4ab8-9306-04ffc4c2cb52/tora-wa-ryuu-wo-mada-tabenai (Used 06/20)
- https://mangadex.org/title/7e4e5e8a-f3f0-4612-b04d-257e2ac0b7eb (Used - 05/21)

05/28/25
- COLORED
- https://mangadex.org/chapter/024dce18-7b4f-409c-87f2-8992f93073e7/1 (Used - 06/20)

- UNCOLORED
- https://mangapill.com/chapters/6698-10001100/tsuihou-sareta-onimotsu-tamer-sekai-yuiitsu-no-necromancer-ni-kakusei-suru-chapter-1.1

06/20/25
- https://mangadex.org/title/28c77530-dfa1-4b05-8ec3-998960ba24d4/otome-game-sekai-wa-mob-ni-kibishii-sekai-desu

- ** https://mangadex.org/title/93f59355-6c69-416f-a5fb-84927eb31d42/danshi-kinsei-game-sekai-de-ore-ga-yarubeki-yuiitsu-no-koto-yuri-no-ma-ni-hasamaru-otoko-to-shite **

## MANHWAS - Might need to use mangacon
- (https://mangadex.org/title/49d31d52-66ed-474e-89d9-5268ec5a610e/boss-in-school)
- https://mangadex.org/title/a796fb50-cdc9-4c15-a48d-bc1d733d89b2/the-player-who-can-t-level-up
- https://mangadex.org/title/274a8e39-71a8-4bd2-af19-4572518fe44f/leviathan