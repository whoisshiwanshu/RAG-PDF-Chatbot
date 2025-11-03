from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import tiktoken

load_dotenv()

openai_client = OpenAI()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vector_db = QdrantVectorStore.from_existing_collection(

     embedding=embedding_model,
     url = "http://localhost:6333" ,
     collection_name = "learning rag",
)







def chat_with_pdf(user_query):
    # 1️⃣ Similarity search for most relevant PDF chunks
    search_results = vector_db.similarity_search(query=user_query, k=10)

    # 2️⃣ Initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # 3️⃣ Prepare and token-count each chunk
    context_chunks = []
    total_tokens = 0
    MAX_CONTEXT_TOKENS = 90000   # stay safe below 128k

    for result in search_results:
        chunk_text = f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}"
        chunk_tokens = len(tokenizer.encode(chunk_text))
        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        context_chunks.append(chunk_text)
        total_tokens += chunk_tokens

    context = "\n\n".join(context_chunks)

    # 4️⃣ Prepare system prompt
    SYSTEM_PROMPT = (
        "You are a helpful assistant that answers questions based ONLY on the provided PDF context.\n"
        "Be concise, accurate, and if unsure, say 'Not mentioned in the document.'\n\n"
        f"Context:\n{context}\n\n"
    )

    # 5️⃣ Measure full message length before sending
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    full_text = SYSTEM_PROMPT + user_query
    total_tokens_used = len(tokenizer.encode(full_text))

    if total_tokens_used > 120000:
        # If still too large, truncate context safely
        trimmed_context = tokenizer.decode(tokenizer.encode(context)[:100000])
        SYSTEM_PROMPT = (
            "You are a helpful assistant that answers questions based ONLY on the provided PDF context.\n\n"
            f"Context:\n{trimmed_context}\n\n"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

    # 6️⃣ Generate response
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=800,
        temperature=0.4,
    )

    return response.choices[0].message.content


