from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

load_dotenv()

# Data Ingestion 
with open("states_of_union.txt","r", encoding="utf8") as f:
  data = f.read()

loder=TextLoader('states_of_union.txt', encoding="utf8")
document=loder.load()


# Chunking of the Data
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text_chunks=text_splitter.split_documents(document)

# Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# Local Vector Index
vectorstore=FAISS.from_documents(text_chunks, embeddings)

# Retriever
retriever=vectorstore.as_retriever()

# Prompt template
template="""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

prompt=ChatPromptTemplate.from_template(template)

output_parser=StrOutputParser()

# LLm Model
llm_model=ChatGoogleGenerativeAI(model="gemini-3.7-flash")

# RAG Chain
rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)

# Response
print(rag_chain.invoke("How is the United States supporting Ukraine economically and militarily?"))