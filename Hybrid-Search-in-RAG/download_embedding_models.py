"""
Download the all-MiniLM-L6-v2 sentence-transformer model from Hugging Face
and save it locally inside the `embedding_model` folder.

Usage:
    python download_embedding_models.py
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent

# Hugging Face model ID
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Saved in its own subfolder so it doesn't overwrite other models in embedding_model/
MODEL_DIR = BASE_DIR / "embedding_model" / "all-MiniLM-L6-v2"


def download_model(model_name: str = MODEL_NAME, model_dir: Path = MODEL_DIR) -> Path:
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"Model already exists at: {model_dir}")
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{model_name}' from Hugging Face...")
    model = SentenceTransformer(model_name)
    model.save(str(model_dir))
    print(f"Model downloaded and saved at: {model_dir}")

    return model_dir


if __name__ == "__main__":
    path = download_model()

    # Quick sanity check: load from the local folder and embed a sentence
    model = SentenceTransformer(str(path))
    embedding = model.encode("Hybrid search combines dense and sparse retrieval.")
    print(f"Loaded local model OK. Embedding dimension: {len(embedding)}")
