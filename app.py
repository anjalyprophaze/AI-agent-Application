import os
import requests
import streamlit as st
import certifi
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")
st.markdown("Search + Weather AI Agent using LangChain & Local LLM (Ollama)")

# ==========================================
# TOOLS
# ==========================================
@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except requests.RequestException as e:
        return f"Weather API request failed: {str(e)}"

    if "current" not in data:
        return f"Could not fetch weather data for {city}: {data.get('error', {}).get('info', 'Unknown error')}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

# ==========================================
# CACHED AGENT INITIALIZATION
# ==========================================
@st.cache_resource
def load_agent():
    search_tool = TavilySearch(max_results=2)

    llm = ChatOllama(
        model="qwen3-coder:30b-64k",   # Must be a tool-calling model pulled in Ollama (e.g., llama3.2, qwen2.5, mistral)
        temperature=0,
        base_url="http://10.10.10.100:11434"
    )

    tools = [search_tool, get_weather_data]
    return create_agent(model=llm, tools=tools)

agent = load_agent()

# ==========================================
# UI INPUT
# ==========================================
user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the capital of India and current weather"
)

# ==========================================
# RUN AGENT WITH LIVE STATUS
# ==========================================
if st.button("Run Agent"):
    if user_query:
        with st.status("Agent is working...", expanded=True) as status:
            try:
                st.write("⏳ Sending request to Local LLM (Ollama)...")
                final_output = ""
                for chunk in agent.stream(
                    {"messages": [("user", user_query)]},
                    stream_mode="values"
                ):
                    latest_msg = chunk["messages"][-1]
                    if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                        for call in latest_msg.tool_calls:
                            st.write(f"🔧 Calling tool: **{call['name']}**")
                    elif latest_msg.type == "ai" and latest_msg.content:
                        final_output = latest_msg.content

                status.update(label="Response Generated!", state="complete", expanded=False)

                st.markdown("## Final Response")
                st.write(final_output)

            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query")