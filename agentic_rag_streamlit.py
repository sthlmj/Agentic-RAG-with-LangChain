# ─── Section 1: Imports & Env Setup ─────────────────────────────────────────────
import os
from dotenv import load_dotenv

import streamlit as st

from supabase.client import create_client, Client

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain import hub
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# ─── Section 2: Supabase & Embeddings Initialization ──────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embedding_model,
    table_name="documents",
    query_name="match_documents",
)

# ─── Section 3: LLM & Prompt Loading ────────────────────────────────────────────
# Use gpt-4o with temperature=0 for deterministic responses
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Pull pre-built tool-calling prompt
tool_calling_prompt: ChatPromptTemplate = hub.pull("hwchase17/openai-functions-agent")

# ─── Section 4: Define Retrieval Tool ─────────────────────────────────────────
@tool(response_format="content_and_artifact")
def retrieve(query: str) -> tuple[str, list]:
    """
    Search for the top-2 semantically similar docs
    and return a formatted string plus raw docs.
    """
    top_docs = vector_store.similarity_search(query, k=2)

    formatted_entries = []
    for doc in top_docs:
        formatted_entries.append(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
        )

    result_text = "\n\n".join(formatted_entries)
    return result_text, top_docs

# Register tools
tools = [retrieve]

# Create the agent and executor
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=tool_calling_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ─── Section 5: Streamlit UI ───────────────────────────────────────────────────
def initialize_session():
    """Ensure chat history exists in session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def render_chat_history():
    """Replay past messages in the Streamlit chat."""
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

def handle_user_input(user_input: str):
    """Add human message, invoke agent, and add AI response."""
    # Display & store the user’s message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append(HumanMessage(user_input))

    # Invoke the agent with chat history
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": st.session_state.messages
    })
    ai_text = response["output"]

    # Display & store the AI’s response
    with st.chat_message("assistant"):
        st.markdown(ai_text)
    st.session_state.messages.append(AIMessage(ai_text))

def main():
    # Page config & title
    st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
    st.title("🦜 Agentic RAG Chatbot")

    initialize_session()
    render_chat_history()

    # Input box
    user_input = st.chat_input("Ask me anything...")
    if user_input:
        handle_user_input(user_input)

if __name__ == "__main__":
    main()
