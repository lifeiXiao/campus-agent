from fastapi import APIRouter
from pydantic import BaseModel
from services.llm_service import get_llm_response
from services.db_service import save_message

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/chat")
async def chat_with_agent(data: ChatRequest):
    question = data.question
    answer = await get_llm_response(question)
    await save_message("test_user", question, answer)
    return {"answer": answer}
