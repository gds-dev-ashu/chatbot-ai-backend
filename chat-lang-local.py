import os
import tempfile
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
if not AZURE_OPENAI_ENDPOINT.endswith("/openai/"):
    AZURE_OPENAI_ENDPOINT += "/openai/"

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
FAISS_DB_PATH = ".local/faiss_index"

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Setup LLM and embedding
llm = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT_NAME,
    temperature=0,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2023-07-01-preview",
    openai_api_key=AZURE_OPENAI_API_KEY,
    model_name="gpt-4",
)

embedding_model = AzureOpenAIEmbeddings(
    deployment=AZURE_EMBEDDING_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2023-07-01-preview",
    openai_api_key=AZURE_OPENAI_API_KEY,
)

# Or use local model
data_embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that answers based on SNOW ticket data."),
    ("human", "Context: {context}\n\nQuestion: {query}")
])


@app.post("/load-tickets/")
async def load_excel_to_vector_db(file: UploadFile = File(...)):
    tmp_path = None
    try:
        # Write uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load and process the Excel file
        loader = UnstructuredExcelLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)

        # Create and save FAISS index
        vector_store = FAISS.from_documents(split_docs, data_embedding_model)
        vector_store.save_local(FAISS_DB_PATH)

        return JSONResponse(content={"status": "success", "message": "Vector DB created and saved successfully."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/chat")
async def retrieve_from_db(data: QueryRequest):
    try:
        vector_db = FAISS.load_local(FAISS_DB_PATH, data_embedding_model, allow_dangerous_deserialization=True)
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})

        print(data)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
        )

        output = qa_chain.invoke({"query": data.query})  # returns dict with 'result' and 'source_documents'

        return JSONResponse(content={
            "query": data.query,
            "answer": output["result"],
            "sources": [doc.metadata for doc in output["source_documents"]],
        })
    except Exception as e:
        tb_str = traceback.format_exc()
        print(tb_str)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
