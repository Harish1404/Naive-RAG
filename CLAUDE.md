## Project: LLM

### Stack
- FastAPI with async routes
- LiteLLM (Groq primary, Gemini fallback)
- SSE streaming via StreamingResponse
- Python 3.11+

### Conventions
- All LLM calls are async
- No bare except — always catch specific exceptions
- Streaming responses use async generators
- implement this feature inside services/chat.py
- That code should be understand to any beginner 

### Current State
- LiteLLM completion ✅
- Fallback + retry ✅  
- Streaming to FastAPI ✅

### Current sprint

 - Build a simple RAG without langchain or any frameworks i have done some coding but need to do the RAG flow and a descent prompt engineering 

 - for embedding first use a simple all-MiniLM-L6-v2 later we can use good model

 
