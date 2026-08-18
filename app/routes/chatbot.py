from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.ai.chat import ChatService
from app.mcp import client as mcp_client

router = APIRouter(tags=["Chatbot"])


@router.post("/chatbot")
async def chatbot(user_prompt: str):
    tools = mcp_client.get_tools()
    service = ChatService(user_prompt=user_prompt, tools=tools)

    return StreamingResponse(
        service.chat(),
        media_type="text/event-stream",
        status_code=200,
    )
