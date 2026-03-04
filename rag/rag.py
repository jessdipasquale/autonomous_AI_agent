"""
Retrieval-Augmented Generation (RAG) module.

Implements document ingestion (chunking, embedding, and upserting into Pinecone)
and semantic retrieval. This module provides the knowledge layer that the agent 
uses to ground its reasoning and scoring decisions.
"""


import os
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI


# RAG constants:
INDEX_NAME = "market-entry-agent"
EMBED_MODEL = "text-embedding-ada-002"
DIMENSION = 1536
CHUNK_SIZE_CHARS = 2000
DOCS_FOLDER = "docs"

# Ensures the Pinecone vector index exists.
# Creates the index if it does not already exist.
# This prepares the vector database for storing document embeddings.
def create_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = "market-entry-agent"
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✅ Index '{index_name}' created!")
    else:
        print(f"ℹ️ Index '{index_name}' already exists.")

# Reads documents from the docs folder, splits them into chunks,
# creates embeddings for each chunk, and uploads them to Pinecone.
# This builds the internal knowledge base for semantic search.
def load_and_embed_documents():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("market-entry-agent")
    
    docs_folder = "docs/"

    for root, dirs, files in os.walk(docs_folder):
        for filename in files:
            if filename.startswith("."):
                continue
            filepath = os.path.join(root, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            
            chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
            
            for i, chunk in enumerate(chunks):
                response = client.embeddings.create(
                    input=chunk,
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                
                index.upsert(vectors=[{
                    "id": f"{filename}_chunk_{i}",
                    "values": embedding,
                    "metadata": {"filename": filename, "chunk": i, "text": chunk}
                }])
                print(f"✅ Uploaded: {filename} — chunk {i}")


# Retrieves the most relevant document chunks for a given query.
# Converts the query into an embedding and performs a vector search in Pinecone.
# Returns the top-k matching chunks with source and similarity score.
def retrieve(query: str, k: int = 5):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    response = client.embeddings.create(
        input=query,
        model=EMBED_MODEL,
    )
    query_vector = response.data[0].embedding

    results = index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True,
    )

    cleaned = []
    for match in results.get("matches", []):
        metadata = match.get("metadata") or {}
        cleaned.append({
            "text": metadata.get("text", ""),
            "source": metadata.get("filename", ""),
            "score": match.get("score", 0.0),
        })

    return cleaned


    
# Test Block > REMOVE LATER!
if __name__ == "__main__":
    print("🔹 Ensuring index exists...")
    create_pinecone_index()

    print("🔹 Ingesting documents...")
    load_and_embed_documents()

    print("🔹 Testing retrieval...")

    queries = [
        "Define Market Attractiveness scoring from 1 to 5",
        "What are the sections of our final report?",
        "What does Go / Explore / No-Go mean in our scoring?"
    ]

    for q in queries:
        print("\n====================")
        print("QUERY:", q)
        print("====================")

        results = retrieve(q, 5)

        for r in results:
            print(r["score"], r["source"])
            print(r["text"][:200])
            print("------")
