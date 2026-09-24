import math
from sentence_transformers import SentenceTransformer

# Load Local Embedding Model
MODEL_PATH = "./embeddings_model"

model = SentenceTransformer(MODEL_PATH)

# Create Embedding
def get_embedding(text):
    """Convert text into a sentence embedding."""
    return model.encode(text)

# Calculate Cosine Similarity
def get_cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors"""

    # Check if the two input vectors are of same dimensions
    if len(vector_a) != len(vector_b):
        raise ValueError("Both vectors must have the same dimensions.")

    # calculate dot product
    dot_product = 0
    for i in range(len(vector_a)):
        dot_product += vector_a[i] * vector_b[i]

    # calculate the magnitude of A
    magnitude_a = 0
    for value in vector_a:
        magnitude_a += value ** 2
    magnitude_a = math.sqrt(magnitude_a)

    # calculate the magnitude of B
    magnitude_b = 0
    for value in vector_b:
        magnitude_b += value ** 2
    magnitude_b = math.sqrt(magnitude_b)

    # Avoid Division by Zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    # Calculate Cosine Similarity
    similarity =  dot_product / (magnitude_a * magnitude_b)

    return similarity