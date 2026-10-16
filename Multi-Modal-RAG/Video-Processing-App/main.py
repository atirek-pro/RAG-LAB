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
import json
from pathlib import Path
import subprocess
import speech_recognition as sr
from PIL import Image
import yt_dlp
from moviepy.editor import VideoFileClip
from google.genai import types

from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from llama_index.core.schema import ImageNode
from llama_index.core.response.notebook_utils import display_source_node

import matplotlib.pyplot as plt

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock



from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from your .env file")

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=GEMINI_API_KEY,
    embed_batch_size=5,
    http_options=types.HttpOptions(timeout=120000),
)

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

text_output_path = output_folder / "output_text.txt"

with open(text_output_path, "w", encoding="utf-8") as file:
    file.write(text_data)

print(f"Text data saved to: {text_output_path}")
file.close()

# ============================================================
# LanceDB / Index configuration
# ============================================================

LANCEDB_DIR = BASE_DIR / "lancedb"
TEXT_TABLE = "text_collection"
IMAGE_TABLE = "image_collection"
INDEX_METADATA_PATH = BASE_DIR / "index_metadata.json"
INDEX_STORAGE_DIR = BASE_DIR / "index_storage"

EMBEDDING_MODEL_NAME = "gemini-embedding-001"


def lance_table_exists(db_path, table_name):
    """Check whether a LanceDB table exists on disk."""

    db_path = Path(db_path)

    if not db_path.exists():
        return False

    table_path = db_path / f"{table_name}.lance"

    return table_path.exists()


def index_is_valid():
    """Check whether an existing index belongs to this video/model."""

    if not INDEX_METADATA_PATH.exists():
        return False

    if not lance_table_exists(LANCEDB_DIR, TEXT_TABLE):
        return False

    if not lance_table_exists(LANCEDB_DIR, IMAGE_TABLE):
        return False

    try:
        with open(
            INDEX_METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            metadata = json.load(f)

    except (json.JSONDecodeError, OSError):
        return False

    if metadata.get("video_url") != video_url:
        return False

    if metadata.get("embedding_model") != EMBEDDING_MODEL_NAME:
        return False

    return True


# ============================================================
# Create / reuse LanceDB
# ============================================================

text_store = LanceDBVectorStore(
    uri=str(LANCEDB_DIR),
    table_name=TEXT_TABLE,
)

image_store = LanceDBVectorStore(
    uri=str(LANCEDB_DIR),
    table_name=IMAGE_TABLE,
)


if index_is_valid() and INDEX_STORAGE_DIR.exists():

    # --------------------------------------------------------
    # EXISTING INDEX
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EXISTING LANCEDB INDEX FOUND")
    print("=" * 60)

    print(f"Video: {video_url}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print("Loading existing index...")
    print("Skipping embedding generation...")
    print("=" * 60)

    # Load the previously persisted LlamaIndex structure.
    # This does NOT generate embeddings again.
    storage_context = StorageContext.from_defaults(
        persist_dir=str(INDEX_STORAGE_DIR),
        vector_store=text_store,
        image_store=image_store,
    )

    index = load_index_from_storage(
        storage_context,
    )

    print("Existing index loaded successfully.")


else:

    # --------------------------------------------------------
    # CREATE NEW INDEX
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("NO VALID LANCEDB INDEX FOUND")
    print("Creating embeddings...")
    print("=" * 60)

    documents = SimpleDirectoryReader(
        input_dir=str(output_folder),
        required_exts=[".txt", ".png"],
    ).load_data()

    storage_context = StorageContext.from_defaults(
        vector_store=text_store,
        image_store=image_store,
    )

    # Expensive operation
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    # --------------------------------------------------------
    # PERSIST LLAMAINDEX INDEX STRUCTURE
    # --------------------------------------------------------
    #
    # This is different from the LanceDB vectors.
    #
    # LanceDB stores the embeddings.
    # index_storage stores the LlamaIndex structure needed
    # to reconstruct the index on the next application run.
    #

    index.storage_context.persist(
        persist_dir=str(INDEX_STORAGE_DIR)
    )

    # --------------------------------------------------------
    # SAVE CACHE METADATA
    # --------------------------------------------------------

    index_metadata = {
        "video_url": video_url,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "text_table": TEXT_TABLE,
        "image_table": IMAGE_TABLE,
    }

    with open(
        INDEX_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            index_metadata,
            f,
            indent=2,
        )

    print("LanceDB index created successfully.")
    print("LlamaIndex index structure persisted.")
    print("Index metadata saved.")

retriever_engine=index.as_retriever(similarity_top_k=1, image_similarity_top_k=5)

def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text

query="can you tell me what is bayes theorm? explain equation of it"

img,text=retrieve(retriever_engine,query)

def plot_images(images_path):
  images_shown = 0
  plt.figure(figsize=(16, 9))
  for img_path in images_path:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 5:
                break

plot_images(img)

qa_tmpl_str=(
    "Based on the provided information, including relevant images and retrieved context from the video, \
    accurately and precisely answer the query without any additional prior knowledge.\n"

    "---------------------\n"
    "Context: {context_str}\n"
    "Metadata for video: {metadata_str} \n"

    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)

metadata_str = json.dumps(metadata_vid)
context_str = "\n".join(text)

prompt = qa_tmpl_str.format(
    query_str=query,
    metadata_str=metadata_str,
    context_str=context_str,
)

blocks = [
    TextBlock(text=prompt)
]

for image_path in img[:5]:
    blocks.append(
        ImageBlock(path=str(image_path))
    )

message = ChatMessage(
    role="user",
    blocks=blocks,
)

mm_llm = GoogleGenAI(
    model="gemini-3.8-flash",
    api_key=GEMINI_API_KEY,
    max_tokens=1500,
)

response = mm_llm.chat([message])

print(response.message.content)