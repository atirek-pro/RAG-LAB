import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Persistent vector database directory
VECTOR_DIR = BASE_DIR / "vector_index"

# Create directory if it doesn't exist
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "agentic_design_patterns"

# Local model directory
MODEL_DIR = BASE_DIR / "embedding_model"

# Hugging Face model ID
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Download model if it doesn't exist locally
if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):

    print("Downloading embedding model...")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    # Save model locally
    model.save(str(MODEL_DIR))

    print(f"Model downloaded and saved at: {MODEL_DIR}")

else:
    print(f"Loading existing embedding model from: {MODEL_DIR}")


doc_path = r'D:\AI Engineering\RAG-LAB\Hybrid Search in RAG\data\Agentic_Design_Patterns.pdf'

loader = PyPDFLoader(doc_path)
pdf_text = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap=30)

chunks = splitter.split_documents(pdf_text)

# Initialize embeddings from local directory
embeddings = HuggingFaceEmbeddings(
    model_name=str(MODEL_DIR),
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

print("Local embedding model initialized successfully!")

# Check whether an existing vector index is available
if (VECTOR_DIR / "chroma.sqlite3").exists():

    print("Existing ChromaDB index found. Loading existing embeddings...")

    vectore_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR)
    )

    # Check whether the collection contains documents
    if vectore_store._collection.count() > 0:
        print(
            f"Loaded existing vector index with "
            f"{vectore_store._collection.count()} embeddings."
        )

    else:
        print("Existing collection is empty. Creating embeddings...")

        vectore_store.add_documents(chunks)

else:

    print("No existing vector index found. Creating a new one...")

    vectore_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR)
    )

    print("Vector index created successfully!")

vectore_store_retriever = vectore_store.as_retriever(
    search_kwargs={"k": 3}
)

keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k = 3

hybrid_retriever = EnsembleRetriever(retrievers=[vectore_store_retriever, keyword_retriever], weights=[0.7, 0.3])

model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

        Answer the user's question using ONLY the provided context.

        If the answer is not available in the context,
        clearly state that you don't know.

        Context:
        {context}
        """
    ),
    ("human", "{input}")
])

# Create document combining chain
combine_docs_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt
)

# Create retrieval chain
hybrid_chain = create_retrieval_chain(
    hybrid_retriever,
    combine_docs_chain
)

query = "What is RAG architecture?"

# Test hybrid retrieval independently
retrieved_docs = hybrid_retriever.invoke(query)

print("\nRETRIEVED DOCUMENTS:")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nDocument {i}")
    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    print("Content:", doc.page_content)
    print("-" * 60)

# Run complete pipeline
response1 = hybrid_chain.invoke({
    "input": query
})

print("\nGENERATED ANSWER:")
print(response1["answer"])