#setting up pydantic model
from pydantic import BaseModel
from typing import List

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str  # Changed from system_prompy to system_prompt
    messages: List[str]
    allow_search: bool


#setup ai agent from frontend request
from fastapi import FastAPI
from app import get_response_from_ai_agent
ALLOWED_MODEL_NAMES = ["llama-3.3-70b-versatile", "gpt-4o-mini"]

# Add CORS middleware imports
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Langgraph AI agent")

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
@app.post("/chat")
def chat_endpoint(request_state: RequestState):
    """
    This endpoint takes a request state and returns a response from the AI agent.
    """
    if request_state.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Auukad mai model select kar"}

    llm_id = request_state.model_name
    query = request_state.messages
    allow_search = request_state.allow_search
    system_prompt = request_state.system_prompt  # Note: matches the class attribute name
    provider = request_state.model_provider

    #create ai agent and get response
    response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "127.0.0.1", port = 8000)

