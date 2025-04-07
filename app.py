import streamlit as st
from streamlit.extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

with st.sidebar:
    st.title("This is a pdf chatting app biatch")
    st.markdown('''
    ## about
    this app is an LLM powered chatbot built using:
    using streamlit
    using langchain
    and using together ai api llama turbo LLM
    ''')
    add_vertical_space(5)
    st.write('made with a lot of hatred and anger')