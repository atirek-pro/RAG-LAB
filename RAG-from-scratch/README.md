# Introduction to RAG

RAG is called Retrieval augmented generation, is an architecture used to help LLMs model like gpt- 4, Gemini, Gemma, LLAMA2,MISTRAL to provide a better response by using relevant information from additional sources and reduces the chance that llm will lead to incorrect information.

RAG is a technique that enhances language model generation by incorporating external knowledge.This is typically done by retrieving relevant information from a large corpus of documents and using that information to inform the generation process.

With RAG, the LLM is able to leverage knowledge and information that is not necessarily in its weights means it is not inside the training.

# Why we should use RAG?

1. Limited knowledge access
2. Lack of transparency: LLMs struggle to provide transparent or relevant information
3. Hallucinations in answers

# RAG Architecture.

### Phase 1 --> Ingestion

![Ingestion Flow](./assets/Ingestion.png)

##### Lets understand the Ingestion process

#### 1. Document:

In a typical RAG pipeline, we have knowledge sources, such as Local Files, CSV, TSV, JSON, PDF, DOCs,Web pages,XML, Databases, Cloud Storage, Any Remote Location etc.

#### 2. Chunking:

We collect the data from various sources, split the data, into the chunks.

#### Why chunking is required?

We can fed the entire document also but why we are just passing the paragraph so here the reason is LLM will not be overloaded with information and Even it is having some limits in terms of tokens.

#### How to figuring out the Ideal Chunk Size?

Too small a chunk won't provide you info. It is not sufficient and These chunks can be defined either by a fixed size, such
as a specific number of characters, sentences or paragraphs. Larger chunks might include irrelevant information, introducing noise and potentially reducing the retrieval accuracy. By controlling the chunk size RAG can maintain a balance between comprehensiveness and precision.

_Based on these factors you can decide the size of chunk_

1. Data Characteristics
2. Retriever Constraints
3. Memory and Computational Resources
4. Task Requirements
5. xperimentation
6. Overlap Consideration

[For more details on Chunking Stratergies click here](https://www.pinecone.io/learn/chunking-strategies/)

#### 3. Embedding:

After chunking we convert it into vector embedding . vector embeddingare numerical representations of the data.

#### Types of embedding

1. _Frequency based embeddings_
   BOW - BAG of WORDS
   TD-IDF - Term Frequenc-Inverse Document Frequency
   N-GRAMS

2. _Neural Network based embedding_
   WORD2VEC
   FAST TEXT
   BERT
   ELMO
   OPENAI EMBEDDINGS
   GEMINI EMBEDDINGS

#### Token level embedding vs Sentence level embedding

#### How sentence transformers differ compared to token-level embedding models such as BERT?

Sentence trañsformers are specifically optimized for producing representations at the sentence level, focusing on capturing the overall semantics of sentences, which makes them particularly useful for tasks involving sentence similarit and clustering. This contrasts with token-level models like BERT, which are more focused on understanding an representing the meaning of individual tokens within their wider context.

_Checkout the links:_

https://huggingface.co/sentence-transformers
https://www.sbert.net/

[To know which embedding model you need to select click here](https://huggingface.co/spaces/mteb/leaderboard)

#### 4. Vector Embedding Indexing

A vector index is a data structure used in computer science and information retrieval to efficiently store and retrieve high dimensional vector data, enabling fast similarity searches and nearest neighbor queries.

![Vector Indexing](./assets/Vector-Indexing.png)

[Click here to Learn More](https://www.datastax.com/quides/what-is-a-vector-index)

#### 5. Database or Retriever:

The retriever here could be any of the following depending on the need

_**Vector database**: A vector database indexes and stores vector embeddings for fast retrieval and similarity search, with capabilities like CRUD operations, metadata filtering, horizontal scaling, and serverless._

Vector databases are best suited for managing and querying high-dimensional data in use cases that require similarity searches.

_**Graph database**: Graph databases are designed to represent and store data as graphs. This makes it easy to represent people, products, and events along with what ties them together. Search engines, logistics businesses, and social networks typically use graph databases to understand connections in their data._

**Nodes** are the primary entities in a graph database. Each node holds all data about a person, product, business, event, or another entity.

**Edges** are the connecting parts of graph databases. They show similarities, relationships, and commonalities. You can define the properties and weights of edges to fit your purpose.

Graph databases bring you the full power of relationships in data.

_**Regular SQL Databases:**_ Offers structured data storage and retrieval but might lack the semantic flexibility of vector databases.

It can be hard to make the oice and go with either graph or vector technology for your database. With generative AI, large language models (LLM), and real-time data playing an increasing part in modern applications, we're seeing an increase in combined solutions.

With generative AI, large language models (LLM), and real-time data playing an increasing part in modern applications, we're seeing an increase in combined solutions.

This is why Neo4i recently added the ability to perform vector similarity search. They aim to make more sense of data and combat LLM hallucinations by blending similar feature vectors to input vectors found through lookups in the knowledggraph

[To know a detailed comparison of different databases click here](https://superlinked.com/vector-db-comparison/)

### Phase 2 --> Retrieval

### Phase 3 --> Generation

![Ingestion Flow](./assets/Retrieval-and-Generation.png)
