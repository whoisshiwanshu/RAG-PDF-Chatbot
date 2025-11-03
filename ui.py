import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import tempfile

# --- Load environment variables ---
load_dotenv()
client = OpenAI()

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📄",
    layout="centered"
)

# --- Custom Apple-Style CSS ---
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', sans-serif;
    }
    .main {
        background-color: #0e1117;
        padding: 2rem;
        border-radius: 20px;
    }
    h1 {
        font-weight: 700 !important;
        text-align: center;
        color: #f5f5f7;
        font-size: 2.5rem !important;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        background-color: #1c1c1e;
        color: #fff;
        border-radius: 12px;
        border: 1px solid #3a3a3c;
        padding: 12px 16px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #007aff, #0a84ff);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0a84ff, #5ac8fa);
        transform: scale(1.05);
    }
    .stFileUploader > div > div {
        background-color: #1c1c1e;
        border: 1px dashed #3a3a3c;
        border-radius: 15px;
        padding: 1.5rem;
        color: #f5f5f7;
    }
    .chat-bubble {
        background-color: #1c1c1e;
        border-radius: 15px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .assistant {
        background-color: #007aff22;
        border: 1px solid #007aff33;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<h1>📘 Chat with your PDF</h1>", unsafe_allow_html=True)
st.write("Upload a PDF and ask anything about it using AI 🤖")

# --- PDF Upload Section ---
uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    reader = PdfReader(tmp_path)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text() or ""

    st.success("✅ PDF uploaded successfully!")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_db = QdrantVectorStore.from_texts(
        texts=[pdf_text],
        embedding=embedding_model,
        url="http://localhost:6333",
        collection_name="uploaded_pdf"
    )

    st.session_state["vector_db"] = vector_db
# --- Chat Input (Press Enter or Click Submit) ---
st.markdown("<br>", unsafe_allow_html=True)

query = st.text_input(
    "Ask something about your PDF...",
    placeholder="Type your question and press Enter..."
)

# Create a submit button too
submit_button = st.button("Submit")

# Treat either Enter or Button click as submission
if query or submit_button:

    if not query.strip():
        st.warning("Please enter a question first.")
    elif "vector_db" not in st.session_state:
        st.error("Please upload a PDF first.")
    else:
        vector_db = st.session_state["vector_db"]
        results = vector_db.similarity_search(query)
        context = "\n".join([r.page_content for r in results])
        context = context[:6000]  # Limit context to avoid overflow

        system_prompt = f"You are an AI assistant answering from the uploaded PDF context.\n\nContext:\n{context}"

        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ]
                )
                st.markdown(
                    "<div class='chat-bubble assistant'><b>🤖 Answer:</b><br>"
                    + response.choices[0].message.content
                    + "</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error: {e}")
