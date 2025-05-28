from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import AzureOpenAI

# Load .env variables
load_dotenv()

client = AzureOpenAI(
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
)

deployed_model = os.getenv("DEPLOYMENT_NAME", "gpt-4o")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and response models
class UserMessage(BaseModel):
    message: str

class BotReply(BaseModel):
    reply: str

@app.get("/", response_model=BotReply)
async def index():
    try:
        return {"reply": "Hello World!!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=BotReply)
async def chat(user_input: UserMessage):
    try:
        response = client.chat.completions.create(
            model=deployed_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input.message}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        reply = response.choices[0].message.content
        return BotReply(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
