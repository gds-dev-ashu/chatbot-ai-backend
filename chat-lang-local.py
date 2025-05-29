import os
import tempfile
import traceback
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import pandas as pd

load_dotenv()

# Azure Configuration - DO NOT MODIFY ENDPOINT FORMAT
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_BASE_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION='2024-12-01-preview'
FAISS_DB_PATH = ".local/faiss_index"
app = FastAPI()

## Base Query class
class QueryRequest(BaseModel):
    query: str

# Validate Azure configuration
if not AZURE_OPENAI_ENDPOINT:
    raise EnvironmentError("AZURE_OPENAI_ENDPOINT is missing in environment variables")
if not AZURE_OPENAI_API_KEY:
    raise EnvironmentError("AZURE_OPENAI_API_KEY is missing in environment variables")

# Initialize Azure services - FIXED ENDPOINT USAGE
try:
    llm = AzureChatOpenAI(
        deployment_name=AZURE_DEPLOYMENT_NAME,
        temperature=0,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY,
        model_name="gpt-4o",
    )

    embeddings = AzureOpenAIEmbeddings(
        deployment=AZURE_EMBEDDING_DEPLOYMENT_NAME,
        azure_endpoint=f'{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_EMBEDDING_DEPLOYMENT_NAME}/embeddings?api-version={AZURE_OPENAI_API_VERSION}',
        openai_api_version=AZURE_OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY,
    )
except Exception as e:
    raise RuntimeError(f"Azure service initialization failed: {str(e)}")

# SNOW table headers
SNOW_HEADERS = [
    "Number", "Opened", "Short description", "Requested by", "Requested for",
    "Location", "Priority", "State", "Category", "Assignment group",
    "Assigned to", "Updated", "Updated by", "Comments and Work notes",
    "Initial Priority", "On hold reason", "Closed"
]

@app.post("/load-tickets")
async def load_excel_to_vector_db(file: UploadFile = File(...)):
    tmp_path = None
    try:
        # Validate file
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Only Excel files accepted (.xlsx, .xls)"
            )

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            tmp.write(content)
            tmp_path = tmp.name

        # Read the Excel file into a DataFrame
        df = pd.read_excel(tmp_path)
        
        # Validate required columns
        missing_columns = [col for col in SNOW_HEADERS if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )

        # Create documents
        documents = []
        for _, row in df.iterrows():
            # Convert all values to string to handle NaNs and other types
            row_data = {col: str(row.get(col, 'N/A')) for col in SNOW_HEADERS}
            
            content = "\n".join(
                f"{header}: {row_data[header]}" 
                for header in SNOW_HEADERS
            )
            metadata = {
                "ticket_id": row_data["Number"],
                "status": row_data["State"],
                **row_data
            }
            documents.append(Document(page_content=content, metadata=metadata))

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\nNumber:", "\nState:"]
        )
        split_docs = splitter.split_documents(documents)

        # Create vector store - REMOVED NORMALIZE_L2 FOR COMPATIBILITY
        vector_store = FAISS.from_documents(
            split_docs,
            embeddings
        )
        os.makedirs(os.path.dirname(FAISS_DB_PATH), exist_ok=True)
        vector_store.save_local(FAISS_DB_PATH)

        return JSONResponse(
            content={
                "status": "success",
                "message": f"Processed {len(split_docs)} ticket chunks from {len(df)} tickets",
                "stats": {
                    "tickets": len(df),
                    "chunks": len(split_docs),
                    "avg_chunk_size": sum(len(d.page_content) for d in split_docs) // len(split_docs)
                }
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Processing error: {str(e)}"}
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/query")
async def query_tickets(data: QueryRequest):
    try:
        # Validate query
        if not data.query or len(data.query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Query must be at least 3 characters"
            )

        # Check if vector store exists
        if not os.path.exists(FAISS_DB_PATH):
            raise HTTPException(
                status_code=404,
                detail="No ticket data found. Please ingest tickets first using /load-tickets"
            )

        # Load vector store
        vector_db = FAISS.load_local(
            FAISS_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Create retriever
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
        
        # Retrieve relevant documents
        docs = retriever.invoke(data.query)

        def format_docs(docs):
            formatted = []
            for doc in docs:
                # Safely get metadata values
                ticket_num = doc.metadata.get("Number", "UNKNOWN")
                state = doc.metadata.get("State", "UNKNOWN")
                opened = doc.metadata.get("Opened", "N/A")
                description = doc.metadata.get("Short description", "N/A")
                requester = doc.metadata.get("Requested by", "N/A")
                assignment_group = doc.metadata.get("Assignment group", "N/A")
                assigned_to = doc.metadata.get("Assigned to", "N/A")
                priority = doc.metadata.get("Priority", "N/A")
                initial_priority = doc.metadata.get("Initial Priority", "N/A")
                updated = doc.metadata.get("Updated", "N/A")
                updated_by = doc.metadata.get("Updated by", "N/A")
                comments = doc.metadata.get("Comments and Work notes", "N/A")
                comments = comments[:200] + "..." if len(comments) > 200 else comments
                
                formatted.append(
                    f"Ticket {ticket_num} ({state}):\n"
                    f"Opened: {opened}\n"
                    f"Description: {description}\n"
                    f"Requester: {requester}\n"
                    f"Assignment: {assignment_group} -> {assigned_to}\n"
                    f"Priority: {priority} (Initial: {initial_priority})\n"
                    f"Last Update: {updated} by {updated_by}\n"
                    f"Comments: {comments}"
                )
            return "\n\n---\n\n".join(formatted)

        # Define prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a ServiceNow (SNOW) ticket assistant. Use ONLY the following context to answer.
        Current date: {current_date}
        
        Context:
        {context}
        
        Question: {question}
        
        Answer in this structured format:
        Summary: [concise response summarizing relevant tickets]
        Relevant Tickets: [comma-separated list of ticket numbers with status in parentheses]
        Analysis: [brief analysis of common patterns or issues]
        Next Steps: [recommended actions if applicable]
        """)

        # Create chain
        chain = (
            RunnableMap({
                "context": lambda x: format_docs(docs),
                "question": lambda x: x["question"],
                "current_date": lambda _: datetime.now().strftime("%Y-%m-%d")
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        # Execute
        response = chain.invoke({"question": data.query})

        # Prepare sources
        sources = []
        for doc in docs:
            sources.append({
                "ticket_id": doc.metadata.get("Number", "UNKNOWN"),
                "status": doc.metadata.get("State", "UNKNOWN"),
                "priority": doc.metadata.get("Priority", "N/A")
            })

        return JSONResponse(content={
            "query": data.query,
            "response": response,
            "sources": sources
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Query error: {str(e)}"}
        )

@app.post("/test-azure")
async def test_azure():
    """Test endpoint for Azure connectivity"""
    try:
        # Test embeddings
        print('### >>>>>>>>>> Testting Embedding')
        test_embedding = embeddings.embed_query("test")
        if not test_embedding or len(test_embedding) < 1:
            raise ValueError("Embedding test failed")
        else :
            print('>>>>> Embedding test passed')
        
        # Test chat
        print('### >>>>>>>>>> Testting LLM')
        response = await llm.ainvoke([HumanMessage(content="What is 2+2?")])
        if not response.content:
            raise ValueError("LLM response empty")
        
        return JSONResponse(content={
            "status": "success",
            "embeddings_test": f"Vector size: {len(test_embedding)}",
            "llm_test": response.content
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Azure connection test failed: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)