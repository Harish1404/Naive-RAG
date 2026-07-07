RAG_SYSTEM_PROMPT = """You are an assistant that answers questions using ONLY the context provided below.

Rules:
- Base your answer strictly on the given context.
- If the answer isn't in the context, say "I don't know based on the information I have."
- Keep answers concise and to the point.
- When useful, mention which source the information came from.
"""


def build_context_message(question: str, chunks: list[dict]) -> str:
    """
    Formats retrieved chunks and the user's question into a single message,
    e.g.:

        [my_resume.pdf]
        ...chunk text...

        [my_resume.pdf]
        ...chunk text...

        Question: What are my key skills?
    """
    if not chunks:
        context_text = "(No relevant context was found.)"
    else:
        context_text = "\n\n".join(f"[{chunk['source']}]\n{chunk['text']}" for chunk in chunks)

    return f"{context_text}\n\nQuestion: {question}"
