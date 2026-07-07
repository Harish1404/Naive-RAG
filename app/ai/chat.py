import logging
import asyncio
import random

from litellm import acompletion, stream_chunk_builder
from litellm.exceptions import RateLimitError, APIError

from app.prompts.rag_prompt import RAG_SYSTEM_PROMPT, build_context_message
from app.core.config import settings
from app.rag.rag_pipeline import rag_pipeline


# logger lets us print debug/error messages to the console with proper labels
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# SECTION 1: ChatService  (streaming chat — already built)
# ─────────────────────────────────────────────────────────

class ChatService:
    """
    Handles a single user chat message.
    Streams the reply back token-by-token using LiteLLM.
    Falls back to a second model if the first one fails.
    """

    def __init__(self, model_type: str, user_prompt: str):

        self.user_prompt = user_prompt

        self.gemini_key = settings.gemini_api_key
        self.groq_key   = settings.groq_api_key

        # We always try Gemini first, then fall back to Groq
        self.fallback_chain = [
            ("gemini/gemini-2.5-flash",   self.gemini_key),
            ("groq/llama-3.1-8b-instant", self.groq_key),
        ]

        # RAG step: pull the most relevant chunks from the ingested documents
        # (e.g. the resume) and hand them to the LLM alongside the question.
        retrieved_chunks = rag_pipeline.retrieve(self.user_prompt)
        context_message = build_context_message(self.user_prompt, retrieved_chunks)

        # The message list that gets sent to the LLM
        self.messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": context_message},
        ]

    async def stream_retry(self, model: str, api_key: str, messages: list, max_retries: int = 3, **kwargs):
        """
        Tries to stream a response from the given model.
        If we hit a rate limit, it waits and retries (up to max_retries times).
        Yields text chunks as they arrive so the user sees output in real time.
        """

        for attempt in range(1, max_retries + 1):

            try:
                # Call the LLM with streaming enabled
                response = await acompletion(
                    model       = model,
                    api_key     = api_key,
                    messages    = messages,
                    temperature = kwargs.get("temperature", 0.7),
                    max_tokens  = kwargs.get("max_tokens", 500),
                    stream      = True
                )

                collected_chunks = []

                try:
                    # Loop over every chunk that arrives from the LLM
                    async for chunk in response:
                        collected_chunks.append(chunk)

                        choices = getattr(chunk, "choices", [])
                        if choices:
                            delta_content = getattr(choices[0].delta, "content", "")
                            if delta_content:
                                yield delta_content   # send this piece to the caller

                    return  # streaming finished successfully — stop retrying

                except Exception as stream_err:
                    logger.error(f"[{model}] Stream interrupted: {stream_err}")
                    yield f"\n[ERROR: Stream interrupted — {stream_err}]"

                finally:
                    # Rebuild the full response object for logging (runs whether or not there was an error)
                    if collected_chunks:
                        try:
                            stream_chunk_builder(collected_chunks)
                        except Exception:
                            pass

            except RateLimitError as e:
                logger.warning(f"[{model}] Rate limited — attempt {attempt}/{max_retries}")

                # Respect the Retry-After header if the API gives us one
                headers         = getattr(e, "headers", {}) or {}
                retry_after     = headers.get("Retry-After") or headers.get("retry-after")
                wait            = float(retry_after) if retry_after else (2 ** attempt) + random.uniform(0.1, 1.0)

                await asyncio.sleep(wait)

            except APIError as e:
                logger.error(f"[{model}] Non-retryable API error: {e}")
                raise e

    async def chat(self):
        """
        Tries each model in the fallback chain.
        Yields chunks from the first model that succeeds.
        """
        for model_name, api_key in self.fallback_chain:
            try:
                async for chunk in self.stream_retry(
                    model       = model_name,
                    api_key     = api_key,
                    messages    = self.messages,
                    temperature = 0.7,
                    max_tokens  = 500,
                ):
                    yield chunk

                return  # success — don't try the next model

            except Exception as e:
                logger.error(f"[{model_name}] Failed, trying next fallback... ({e})")






