from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import os

# === Configuration ===
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = "gpt-4"
AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-large"

openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-07-01-preview"

# === FastAPI Setup ===
app = FastAPI()

# === Pydantic Models ===
class UserQuery(BaseModel):
    user_id: str
    query: str

# === LangChain Setup ===
llm = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT_NAME,
    temperature=0,
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2023-07-01-preview",
    openai_api_key=AZURE_OPENAI_API_KEY,
)

embedding_model = AzureOpenAIEmbeddings(
    deployment=AZURE_EMBEDDING_DEPLOYMENT,
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2023-07-01-preview",
    openai_api_key=AZURE_OPENAI_API_KEY,
)

# Load pre-built FAISS index
vector_store = FAISS.load_local("./faiss_snow_index", embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# === Prompt Setup ===
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an enterprise IT support assistant. Respond based on SNOW ticket data."),
    ("human", "{query}")
])

# === LangChain Chain ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
)

# === LangGraph State Machine ===
class ChatState(Dict):
    query: str
    user_id: str
    result: Any

def input_parser(state: ChatState) -> ChatState:
    # Add classification logic if needed
    return state

def run_query(state: ChatState) -> ChatState:
    query = state["query"]
    result = qa_chain.run(query)
    state["result"] = result
    return state

def format_output(state: ChatState) -> str:
    return state["result"]

workflow = StateGraph(ChatState)
workflow.add_node("parse_input", input_parser)
workflow.add_node("query_snow", run_query)
workflow.add_node("format_output", format_output)

workflow.set_entry_point("parse_input")
workflow.add_edge("parse_input", "query_snow")
workflow.add_edge("query_snow", "format_output")
workflow.add_edge("format_output", END)

app_graph = workflow.compile()

# === REST Endpoint ===
@app.post("/query")
async def query_snow(data: UserQuery):
    result = app_graph.invoke({"query": data.query, "user_id": data.user_id})
    return JSONResponse({"response": result})

# === WebSocket Support ===
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_query = UserQuery(**data)
            result = app_graph.invoke({"query": user_query.query, "user_id": user_query.user_id})
            await websocket.send_json({"response": result})
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
