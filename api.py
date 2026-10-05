from fastapi import FastAPI
from pydantic import BaseModel

from app import agent_answer

app = FastAPI(
    title="Rider Recruitment Agent API",
    description="An API for a Rider Recruitment Agent",
    version="1.0.0"
)

class Query(BaseModel):
    query: str



# ==========================================
# 2. Request schema
# ==========================================

class ChatRequest(BaseModel):
    query: str


# ==========================================
# 3. Response schema
# ==========================================

class ChatResponse(BaseModel):
    answer: str


# ==========================================
# 4. Health check
# ==========================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Rider Recruitment Agent API is running"
    }


# ==========================================
# 5. Agent endpoint
# ==========================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    answer = agent_answer(
        request.query
    )

    return ChatResponse(
        answer=answer
    )