import json
import math
import requests
from data import corpus_of_documents
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

def return_most_similar_docs(query, corpus, top_k):
    query_embedding = get_embedding(query)
    similarities = []
    for doc in corpus:
        document_embedding = get_embedding(doc)
        similarity = get_cosine_similarity(
            query_embedding,
            document_embedding
        )
        similarities.append((doc, similarity))
    
    # Sort by similarity in descending order
    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    relevent_docs = [similarities[i][0] for i in range(top_k)]

    return relevent_docs

query = "i like fresh air."
relevant_documents=return_most_similar_docs(query,corpus_of_documents, 3)
print(relevant_documents)

# Convert Retrieved Documents Into Prompt-Friendly Format
relevant_document_text = "\n".join(
    f"- {document}"
    for document in relevant_documents
)

prompt = """
You are a bot that makes recommendations for activities.
You answer in very short sentences and do not include extra information.
These are the recommended activities:
{relevant_document}
The user input is:
{user_input}
Compile a recommendation to the user based on the recommended activities
and the user input.
"""

# Ollama API
url = "http://localhost:11434/api/generate"

data = {
    "model": "llama2",
    "prompt": prompt.format(
        user_input=query,
        relevant_document=relevant_document_text
    ),
    "stream": True
}

headers = {
    "Content-Type": "application/json"
}

# Send Request
full_response = []

response = requests.post(
    url,
    data=json.dumps(data),
    headers=headers,
    stream=True
)


# Read Streaming Response
try:

    for line in response.iter_lines():

        if not line:
            continue

        decoded_line = json.loads(
            line.decode("utf-8")
        )

        # Only process chunks containing "response"
        if "response" in decoded_line:

            full_response.append(
                decoded_line["response"]
            )

        # Ollama sends this when generation is complete
        if decoded_line.get("done", False):
            break

finally:

    response.close()


# Final Response
print("\nRecommendation:")
print("".join(full_response))