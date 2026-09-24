from sentence_transformers import SentenceTransformer
from pathlib import Path


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = Path("embeddings_model")


print("Downloading model...")

model = SentenceTransformer(MODEL_NAME)

model.save_pretrained(MODEL_PATH)

print(f"Model downloaded successfully to: {MODEL_PATH}")