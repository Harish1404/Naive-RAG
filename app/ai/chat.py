import logging
from typing import AsyncGenerator

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# LLM factory with automatic fallback chain
# ─────────────────────────────────────────────────────────

def create_llm_with_fallbacks(temperature: float = 0.7, max_tokens: int = 1024):
    """
    Builds a LangChain chat model with built-in fallbacks.
    Priority: Mistral -> Gemini -> Groq.
    """
    candidate_models = []

    if settings.MISTRAL_API_KEY:
        candidate_models.append(
            ChatMistralAI(
                model="mistral-large-2407",
                api_key=settings.MISTRAL_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=2,
            )
        )

    if settings.GEMINI_API_KEY:
        candidate_models.append(
            ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.GEMINI_API_KEY,
                temperature=temperature,
                max_output_tokens=max_tokens,
                max_retries=2,
            )
        )

    if settings.GROQ_API_KEY:
        candidate_models.append(
            ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=settings.GROQ_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=2,
            )
        )

    if not candidate_models:
        raise ValueError("No valid API keys configured for Mistral, Gemini, or Groq.")

    primary = candidate_models[0]
    fallbacks = candidate_models[1:]

    if fallbacks:
        return primary.with_fallbacks(fallbacks=fallbacks)
    return primary


# ─────────────────────────────────────────────────────────
# ChatService — MCP-powered ReAct agent with streaming
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant with access to GitHub tools.
When the user asks about GitHub repositories, issues, pull requests, or code,
use the available tools to fetch real data. For general questions, answer directly.
Be concise and helpful."""


class ChatService:
    """
    Handles user chat messages using a LangGraph ReAct agent.
    The agent has access to MCP tools (GitHub) and can decide
    whether to call tools or respond directly.
    """

    def __init__(self, user_prompt: str, tools: list):
        self.user_prompt = user_prompt
        self.tools = tools

        llm = create_llm_with_fallbacks()
        self.agent = create_react_agent(model=llm, tools=self.tools)

    async def chat(self) -> AsyncGenerator[str, None]:
        """
        Streams response tokens from the ReAct agent.
        Uses astream_events to capture only the final LLM text output.
        """
        input_messages = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=self.user_prompt),
            ]
        }

        try:
            async for event in self.agent.astream_events(input_messages, version="v2"):
                kind = event.get("event")

                # Only yield text chunks from the chat model stream
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield chunk.content

        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            yield f"\n[ERROR: Agent failed — {str(e)}]"
