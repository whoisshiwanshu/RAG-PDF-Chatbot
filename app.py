import streamlit as st
from ui import main as ui_main

# Streamlit page config
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")

st.title("📄 RAG-based AI Chatbot with PDF Support")

# Call your UI or logic function
ui_main()
