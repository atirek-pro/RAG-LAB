"""
ffmpeg-library enables you to use FFmpeg in Python to manipulate various media files for different purposes like building comprehensive 
multimedia applications, preprocessing media files.

MoviePy is a Python library for video editing, enabling cutting, concatenations, title insertions, video compositing, and effects like 
animations or color grading.

Pytube is a Python library used for downloading videos from YouTube. It supports downloading in various formats, resolutions, and also 
direct audio extraction.

Pydub is a Python library for audio manipulation, enabling easy loading, editing, and exporting of audio files in various formats with
minimal code.

The SpeechRecognition library in Python allows you to convert spoken language into text using various engines and APIs, such as 
Google Speech Recognition, IBM Speech to Text, etc.

SoundFile is a Python library for reading from and writing to audio files, supporting many formats through the libsndfile library, 
ideal for high-quality audio processing.

FTFY (Fix Text For You) is a Python library that fixes broken Unicode text and mojibake (garbled text due to encoding issues), 
making text legible again.

OpenAI Whisper is a robust, multilingual speech recognition model developed by OpenAI. It converts speech into text and supports various 
languages with high accuracy.

pprint is a Python module that provides a capability to "pretty-print" complex data structures in a well-formatted and more readable 
way than the basic print function.
"""


import os
from pathlib import Path
import subprocess
import speech_recognition as sr
import yt_dlp
from moviepy.editor import VideoFileClip

from llama_index.core import StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.lancedb import LanceDBVectorStore

# Configuration
video_url = "https://www.youtube.com/watch?v=U_85TaXbeIo"

BASE_DIR = Path(
    r"D:\AI Engineering\RAG-LAB\Multi-Modal-RAG\Video-Processing-App"
)

output_video_path = BASE_DIR / "video_data"
output_folder = output_video_path / "mixed_data"
output_audio_path = output_folder / "output_audio.wav"
filepath = output_video_path / "input_vid.mp4"


def download_video(url, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    output_template = str(output_path / "input_vid.%(ext)s")

    ydl_opts = {
        "format": (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]"
            "/best[ext=mp4]/best"
        ),
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "ffmpeg_location": r"C:\ffmpeg-2026-10-12-git-615b07170e-full_build\bin",
        "cookiefile": str(
            Path(__file__).resolve().parent / "cookies.txt"
        ),
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
    }


    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    metadata = {
        "Author": info.get("channel") or info.get("uploader"),
        "Title": info.get("title"),
        "Views": info.get("view_count"),
    }

    # yt-dlp merges the streams into MP4 when needed.
    final_path = output_path / "input_vid.mp4"

    if not final_path.exists():
        raise FileNotFoundError(
            f"Downloaded MP4 was not found at: {final_path}"
        )

    return metadata, final_path


def video_to_images(video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with VideoFileClip(str(video_path)) as clip:
        clip.write_images_sequence(
            str(output_dir / "frame%04d.png"),
            fps=0.2,
        )


def video_to_audio(video_path, audio_path):
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    with VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            raise RuntimeError("The video does not contain an audio track.")

        clip.audio.write_audiofile(str(audio_path))


def audio_to_text(audio_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(str(audio_path)) as source:
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_whisper(audio_data)
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio."


# Execute pipeline
metadata_vid, filepath = download_video(
    video_url, output_video_path
)

print("Video metadata:")
print(metadata_vid)

video_to_images(filepath, output_folder)
video_to_audio(filepath, output_audio_path)

text_data = audio_to_text(output_audio_path)

with open(output_folder + "output_text.txt", "w") as file:
        file.write(text_data)
print("Text data saved to file")
file.close()


