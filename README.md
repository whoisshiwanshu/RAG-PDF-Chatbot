📄 RAG PDF Chatbot

A simple RAG-based PDF Chatbot built with Streamlit, LangChain, OpenAI, and Qdrant.
Upload PDFs, index them, and chat with their content.

⚙️ Setup

git clone https://github.com/whoisshiwanshu/rag-pdf-chatbot.git
cd rag-pdf-chatbot
pip install -r requirements.txt


> CREATE A NEW .ENV FILE

OPENAI_API_KEY=your_openai_api_key


🧩 Commands

1️⃣ Index PDF

python index.py


2️⃣ Run Chatbot

streamlit run ui.py

🧠 Tech

Streamlit , LangChain , OpenAI API , Qdrant

📦REQUIREMENTS.TXT

streamlit
langchain
langchain-community
langchain-openai
langchain-qdrant
python-dotenv
pypdf
qdrant-client
