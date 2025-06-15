# ─── Section 1: Imports & Environment ──────────────────────────────────────────
import os
from dotenv import load_dotenv

from supabase.client import create_client, Client

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain_core.tools import tool

# Load environment variables from a .env file
load_dotenv()

# ─── Section 2: Initialize Supabase & Embeddings ───────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Create a Supabase client
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Create an embedding model (converts text → vectors)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a vector store in Supabase for semantic search
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embedding_model,
    table_name="documents",
    query_name="match_documents",
)

# ─── Section 3: Language Model & Prompt Setup ───────────────────────────────────
# Instantiate ChatOpenAI with deterministic responses (temperature=0)
llm = ChatOpenAI(temperature=0)

# Pull a pre-built “tool-calling” prompt template from the LangChain Hub
tool_calling_prompt: ChatPromptTemplate = hub.pull("hwchase17/openai-functions-agent")

# ─── Section 4: Define Retrieval “Tool” ─────────────────────────────────────────
@tool(response_format="content_and_artifact")
def retrieve(question: str) -> tuple[str, list]:
    """
    Retrieve the top-2 most semantically similar documents
    from the vector store & format the output.
    """
    # Perform similarity search
    top_docs = vector_store.similarity_search(question, k=2)

    # Build a human-readable string from the results
    formatted = []
    for doc in top_docs:
        src = doc.metadata or {}
        text = doc.page_content
        formatted.append(f"Source: {src}\nContent: {text}\n")

    # Join entries with blank lines
    result_text = "\n".join(formatted)
    return result_text, top_docs  # content_and_artifact format

# ─── Section 5: Agent Creation & Execution ────────────────────────────────────
# Register our one tool
tools = [retrieve]

# Build an “agent” that knows how to call our retrieve tool
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=tool_calling_prompt,
)

# Wrap it in an executor to manage calls & streaming
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    # Ask a sample question and print the agent’s answer
    question = "Why is agentic RAG better than naive RAG?"
    result = agent_executor.invoke({"input": question})
    print(result["output"])
