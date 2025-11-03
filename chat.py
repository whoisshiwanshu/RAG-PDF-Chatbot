from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import tiktoken
from qdrant_client import QdrantClient
import os

load_dotenv()

openai_client = OpenAI()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

qdrant_client = QdrantClient(url=QDRANT_URL)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    client=qdrant_client,
    collection_name="learning rag",
)

def chat_with_pdf(user_query):
    search_results = vector_db.similarity_search(query=user_query, k=10)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    context_chunks = []
    total_tokens = 0
    MAX_CONTEXT_TOKENS = 90000

    for result in search_results:
        chunk_text = f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}"
        chunk_tokens = len(tokenizer.encode(chunk_text))
        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        context_chunks.append(chunk_text)
        total_tokens += chunk_tokens

    context = "\n\n".join(context_chunks)

    SYSTEM_PROMPT = (
        "You are a helpful assistant that answers questions based ONLY on the provided PDF context.\n"
        "Be concise, accurate, and if unsure, say 'Not mentioned in the document.'\n\n"
        f"Context:\n{context}\n\n"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=800,
        temperature=0.4,
    )

    return response.choices[0].message.content
