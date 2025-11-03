import streamlit as st
import ui

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")

st.title("📄 RAG-based AI Chatbot with PDF Support")

if __name__ == "__main__":
    ui.main_ui()
