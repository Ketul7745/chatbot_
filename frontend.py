import streamlit as st

st.set_page_config(page_title="Langgraph AI Agent", page_icon=":robot:", layout = "centered")
st.title("langgraph AI Agent")
st.write("Welcome to the langgraph AI Agent")

system_prompt = st.text_area("define your ai agent:", height = 60, placeholder="type your prompt here")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]
provider = st.selectbox("Select a provider", ['openai', 'groq'])

if provider == 'openai':
    model_name = st.selectbox("Select a model", MODEL_NAMES_OPENAI)
else:
    model_name = st.selectbox("Select a model", MODEL_NAMES_GROQ)

allow_search = st.checkbox("Allow search")

user_query = st.text_area("define your query:", height = 150, placeholder="type your query here")

API_URL = "http://localhost:8501/chat"

if st.button("Ask Agent?"):
    if user_query.strip():
        #connecting with backend
        import requests
        payload = {
            "model_name": model_name,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_search,
        }
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                response_data = response.json()
                if "error" in response_data:
                    st.error(response_data["error"])
                else:
                    st.subheader("Agent response")
                    st.markdown(response_data)  # Display the actual response content
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")

