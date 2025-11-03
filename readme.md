# 🧠 RAG-based AI Chatbot with PDF Support

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot that lets users **upload PDFs and ask context-aware questions**.  
Built using **LangChain**, **OpenAI embeddings**, and **Qdrant Vector Database** for powerful semantic search — all wrapped in a **modern Streamlit UI** with an Apple-like minimalist aesthetic.

---

## 🚀 Features

- 📄 **PDF Upload & Parsing** — Upload any document and extract relevant context.  
- 💬 **AI-Powered Q&A** — Ask questions and get precise, contextual answers using RAG.  
- 🔍 **Vector Search** — Powered by Qdrant for lightning-fast semantic retrieval.  
- ⚙️ **LangChain Integration** — Efficient retrieval pipeline for better accuracy.  
- 🎨 **Sleek Streamlit UI** — Clean, responsive, and minimal interface.  
- 🔒 **Secure Keys** — Environment variables handled via `.env` file.

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| Language Model | OpenAI GPT |
| Framework | LangChain |
| Vector Database | Qdrant |
| Frontend | Streamlit |
| Language | Python |

---

## ⚙️ Installation & Setup

Follow these simple steps to set up locally:

### 1️⃣ Clone the Repository
# bash
git clone https://github.com/whoisshiwanshu/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot

2️⃣ Create a Virtual Environment (Optional but Recommended)
# bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On macOS/Linux

3️⃣ Install Dependencies
# bash
pip install -r requirements.txt

4️⃣ Add Your API Key

Create a .env file in the root directory and add:

OPENAI_API_KEY=your_openai_api_key_here

5️⃣ Run Qdrant (Vector Database)

If you have Docker installed:
# bash
docker-compose up -d

6️⃣ Run the Application
# bash
streamlit run ui.py

Your app will launch on http://localhost:8501
