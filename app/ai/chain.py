import json                          # needed to parse JSON from Stage 1 output
import logging
import asyncio
import random

from litellm import acompletion, stream_chunk_builder
from litellm.exceptions import RateLimitError, APIError

from app.prompts.chain_prompts import STAGE_1_EXTRACT, STAGE_2_ENRICH, STAGE_3_FORMAT
from app.core.config import settings
from app.prompts.system_prompt import SYSTEM_PROMPT

# logger lets us print debug/error messages to the console with proper labels
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# SECTION 2: Helper — Non-streaming LLM call
# ─────────────────────────────────────────────────────────

async def call_llm_without_streaming(messages: list) -> str:
    """
    Calls the LLM and WAITS for the complete reply before returning.

    Why no streaming here?
    Chaining requires the FULL output of one stage before the next stage can start.
    (e.g. Stage 1 returns JSON — we can't parse half a JSON string)

    Returns the full reply as a plain string.
    """

    gemini_key = settings.gemini_api_key
    groq_key   = settings.groq_api_key

    # Try Gemini first, fall back to Groq if it fails
    fallback_chain = [
        ("gemini/gemini-2.5-flash",   gemini_key),
        ("groq/llama-3.1-8b-instant", groq_key),
    ]

    last_error = None

    for model_name, api_key in fallback_chain:
        if not api_key:
            continue  # skip if this API key is not configured

        try:
            logger.info(f"Calling model (non-streaming): {model_name}")

            # stream=False means we wait for the entire response before continuing
            response = await acompletion(
                model       = model_name,
                api_key     = api_key,
                messages    = messages,
                temperature = 0.3,   # lower temperature = more factual, less creative
                stream      = False
            )

            # Extract the text content from the response object
            return response["choices"][0]["message"]["content"]

        except Exception as e:
            logger.warning(f"[{model_name}] Failed in non-streaming call: {e}. Trying fallback...")
            last_error = e
            continue

    # If we reach here, all models failed
    raise Exception(f"All models failed in non-streaming call. Last error: {last_error}")


# ─────────────────────────────────────────────────────────
# SECTION 3: ChainService  (the new 3-stage pipeline)
# ─────────────────────────────────────────────────────────

class ChainService:
    """
    Runs a 3-stage LLM pipeline on a block of raw text:

      Stage 1 — EXTRACT : Find key concepts in the text (returns JSON)
      Stage 2 — ENRICH  : For each concept, generate a detailed explanation
      Stage 3 — FORMAT  : Combine everything into a clean Markdown report

    Each stage waits for the previous one to finish completely
    because each stage's output becomes the next stage's input.
    """

    @staticmethod
    def clean_json_response(content: str) -> dict:
        """
        The LLM sometimes wraps JSON in markdown fences like:
            ```json
            { ... }
            ```
        This method strips those fences and parses the raw JSON.
        """
        content = content.strip()

        # Remove the opening ``` or ```json line
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]           # remove first line
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]          # remove last line
            content = "\n".join(lines).strip()

        return json.loads(content)          # parse and return as a Python dict

    @classmethod
    async def run_concept_chain(cls, raw_text: str) -> dict:
        """
        Entry point for the full 3-stage chain.

        Parameters:
            raw_text  — the user's input text (a paragraph, article, notes, etc.)

        Returns a dict with:
            success          — True or False
            final_report     — the formatted Markdown output (if successful)
            execution_history — list of what each stage did (useful for debugging)
        """

        logger.info("Starting 3-stage LLM chain...")
        execution_history = []   # we'll append a record after each stage


        # ── STAGE 1: EXTRACT ──────────────────────────────
        # Ask the LLM to read the raw text and pull out key concepts as JSON.
        # Example output:
        #   { "concepts": [{ "name": "Machine Learning", "context": "..." }] }

        logger.info("Stage 1: Extracting concepts from text...")

        messages_stage1 = [
            {"role": "system", "content": STAGE_1_EXTRACT},   # tells the LLM what to do
            {"role": "user",   "content": raw_text}            # the actual input text
        ]

        try:
            raw_json_string  = await call_llm_without_streaming(messages_stage1)
            parsed_json      = cls.clean_json_response(raw_json_string)
            concepts         = parsed_json.get("concepts", [])

            # Save what happened in this stage to history
            execution_history.append({
                "stage":         "extract",
                "raw_output":    raw_json_string,
                "parsed_output": parsed_json,
                "status":        "success"
            })

        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            return {
                "success":           False,
                "error":             f"Stage 1 (Extract) failed: {str(e)}",
                "execution_history": execution_history
            }

        # Safety check — if the LLM found no concepts, we can't continue
        if not concepts:
            return {
                "success":           False,
                "error":             "Stage 1 returned no concepts. Try a more descriptive input.",
                "execution_history": execution_history
            }


        # ── STAGE 2: ENRICH ───────────────────────────────
        # For each concept found in Stage 1, ask the LLM for a deeper explanation.
        # We loop through each concept and make one LLM call per concept.

        logger.info(f"Stage 2: Enriching {len(concepts)} concept(s)...")

        enriched_concepts = []   # we'll build this list as we go

        for index, concept in enumerate(concepts):
            concept_name    = concept.get("name", "Unknown")
            concept_context = concept.get("context", "")

            logger.info(f"  Enriching concept {index + 1}/{len(concepts)}: '{concept_name}'")

            messages_stage2 = [
                {"role": "system", "content": STAGE_2_ENRICH},
                {
                    "role":    "user",
                    "content": f"Concept: {concept_name}\nContext: {concept_context}"
                }
            ]

            try:
                enrichment_text = await call_llm_without_streaming(messages_stage2)

                enriched_concepts.append({
                    "name":       concept_name,
                    "context":    concept_context,
                    "enrichment": enrichment_text    # the detailed explanation
                })

            except Exception as e:
                # If one concept fails to enrich, we don't stop — we add a placeholder and move on
                logger.warning(f"  Failed to enrich '{concept_name}': {e}. Using placeholder.")
                enriched_concepts.append({
                    "name":       concept_name,
                    "context":    concept_context,
                    "enrichment": "Enrichment unavailable due to an API error."
                })

        execution_history.append({
            "stage":             "enrich",
            "enriched_concepts": enriched_concepts,
            "status":            "success"
        })


        # ── STAGE 3: FORMAT ───────────────────────────────
        # Pass all enriched concepts to the LLM and ask it to write a clean Markdown report.

        logger.info("Stage 3: Formatting final Markdown report...")

        # Convert the Python list to a JSON string so the LLM can read it clearly
        enriched_json_string = json.dumps(enriched_concepts, indent=2)

        messages_stage3 = [
            {"role": "system", "content": STAGE_3_FORMAT},
            {"role": "user",   "content": enriched_json_string}
        ]

        try:
            final_report = await call_llm_without_streaming(messages_stage3)

            execution_history.append({
                "stage":        "format",
                "final_report": final_report,
                "status":       "success"
            })

            # All 3 stages done — return the final result
            return {
                "success":           True,
                "final_report":      final_report,
                "execution_history": execution_history
            }

        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            return {
                "success":           False,
                "error":             f"Stage 3 (Format) failed: {str(e)}",
                "execution_history": execution_history
            }



