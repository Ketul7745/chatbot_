import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import cohere
from langchain.embeddings.base import Embeddings
import os
from dotenv import load_dotenv

load_dotenv()
# Initialize Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY") 
if not COHERE_API_KEY:
    st.error("Cohere api key not found") # Replace with your Cohere API key
cohere_client = cohere.Client(COHERE_API_KEY)

# print("Cohere api key:" ,COHERE_API_KEY)

class CohereEmbeddings(Embeddings):
    def __init__(self, cohere_client, model="embed-english-light-v2.0"):
        self.cohere_client = cohere_client
        self.model = model

    def embed_documents(self, texts):
        # Cohere API expects a list of strings
        response = self.cohere_client.embed(texts=texts, model=self.model)
        return response.embeddings

    def embed_query(self, text):
        # For single query embedding
        response = self.cohere_client.embed(texts=[text], model=self.model)
        return response.embeddings[0]

# Set up the page configuration
st.set_page_config(page_title="Langgraph AI Agent", page_icon=":robot:", layout="wide")

# Sidebar for model settings
st.sidebar.title("Model Settings")
system_prompt = st.sidebar.text_area("Define your AI agent:", height=75, placeholder="Type your prompt here")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]
provider = st.sidebar.selectbox("Select a provider", ['GROQ', 'OPENAI'])

if provider == 'GROQ':
    model_name = st.sidebar.selectbox("Select a model", MODEL_NAMES_GROQ)
else:
    model_name = st.sidebar.selectbox("Select a model", MODEL_NAMES_OPENAI)

allow_search = st.sidebar.checkbox("Allow search")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload for documents:
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = ""

# Main chat interface
st.title("Langgraph AI Chatbot")
st.write("Start chatting with the AI below!")

# PDF upload section
st.subheader("Upload a PDF Document")
uploaded_file = st.file_uploader("Choose a pdf file", type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    try:
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    except Exception as e:
        st.error(f"Failed to extract text as the PDF is too large")

    chunk_size = st.sidebar.number_input("Chunk size", min_value=1, max_value=1000, value=200)
    chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=500, value=20)
    if chunk_overlap >= chunk_size:
        st.error("Chunk overlap must be less than the chunk size.")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(pdf_text)

    # Generate embeddings using Cohere
    def get_cohere_embeddings(texts):
        response = cohere_client.embed(texts=texts, model="embed-english-light-v2.0")  # Use a Cohere embedding model
        return response.embeddings

    embeddings = get_cohere_embeddings(chunks)

    # Initialize the custom Cohere embeddings wrapper
    cohere_embeddings = CohereEmbeddings(cohere_client)

    # Store embeddings in FAISS vector store
    vector_store = FAISS.from_texts(chunks, cohere_embeddings)

    # Store the vector store in session state
    st.session_state.vector_store = vector_store
    st.success("PDF file uploaded and processed successfully!")

# Display chat history dynamically
st.subheader("Chat History")
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "ai":
            st.markdown(f"**AI:** {message['content']}")

# Input for user query
user_query = st.text_input("Your message:", placeholder="Type your query here and press Enter")

API_URL = "http://127.0.0.1:8000/chat"

# Automatically handle user input when Enter is pressed
if user_query.strip():
    # Add the user's query to the chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    context = ""
    if st.session_state.vector_store is not None:
        docs = st.session_state.vector_store.similarity_search(user_query, k=1)
        context = "\n".join(doc.page_content[:1000] for doc in docs)  # Limit each document to 1000 characters
    else:
        st.warning("No PDF document uploaded. Please upload a PDF.")

    # Limit chat history to the last 5 messages
    max_history = 5
    recent_history = st.session_state.chat_history[-max_history:]

    # Prepare the payload for the backend
    payload = {
        "model_name": model_name,
        "model_provider": provider,
        "system_prompt": system_prompt,
        "messages": [context] + [msg["content"] for msg in recent_history if msg["role"] == "user"],
        "allow_search": allow_search,
    }

    # Debug: Monitor token usage
    total_tokens = len(context.split()) + sum(len(msg["content"].split()) for msg in recent_history)
    st.write(f"Total tokens in request: {total_tokens}")

    try:
        # Send the request to the backend
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                if isinstance(response_data, dict):
                    ai_response = response_data.get("content", "No response")
                else:
                    ai_response = str(response_data)
                # Add the AI's response to the chat history
                st.session_state.chat_history.append({"role": "ai", "content": ai_response})
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")

    # Clear the input box after sending the message
