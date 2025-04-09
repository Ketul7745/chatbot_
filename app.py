import os
from dotenv import load_dotenv
load_dotenv()

# GROQ_API_KEY = "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"
# TAVILY_API_KEY = "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"
# OPENAI_API_KEY = "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"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Groq API key not found")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("Tavily API key not found")
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("OpenAI API key not found")

print("API keys loaded")
#setting up llms
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

groq_llm = ChatGroq(model = "llama-3.3-70b-versatile", groq_api_key = GROQ_API_KEY)
openai_llm = ChatOpenAI(model = "gpt-4o-mini", openai_api_key = OPENAI_API_KEY)


system_prompt = "Act as an AI agent who is just plain smart"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider.lower() == "groq":  # Changed to case-insensitive comparison
        llm = ChatGroq(model=llm_id, groq_api_key=GROQ_API_KEY)
    elif provider.lower() == "openai":  # Changed to case-insensitive comparison
        llm = ChatOpenAI(model=llm_id, openai_api_key=OPENAI_API_KEY)
    else:
        return "Invalid provider specified"

    # Move agent creation outside the OpenAI condition
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import AIMessage
    
    tools = [TavilySearchResults(max_results=2, tavily_api_key=TAVILY_API_KEY)] if allow_search else []
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )

    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1] if ai_messages else "No response"


