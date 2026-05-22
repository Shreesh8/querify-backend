"""
services/ai/gemini_client.py

Gemini API client.
Isolated behind this class so swapping to OpenAI is a one-file change.
Uses tenacity for automatic retries on rate-limit / transient errors.
"""

import json
import re
from typing import Any, Dict

import google.generativeai as genai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.exceptions import AIServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiClient:

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def get_operation_spec(self, prompt: str) -> Dict[str, Any]:
        """
        Call Gemini and extract a JSON operation spec.
        The prompt explicitly tells Gemini to return only JSON.
        We strip markdown fences and parse it strictly.
        """
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,          # low temp = deterministic, factual
                    max_output_tokens=512,
                )
            )
            raw_text = response.text
            return self._extract_json(raw_text)

        except Exception as e:
            logger.error("gemini_call_failed", error=str(e))
            raise

    async def get_insights(self, prompt: str) -> str:
        """
        Call Gemini for free-form insight generation.
        Returns plain text — no JSON parsing needed here.
        """
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=1500,
                )
            )
            return response.text
        except Exception as e:
            logger.error("gemini_insights_failed", error=str(e))
            raise AIServiceError(f"Gemini insights call failed: {str(e)}")

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """
        Robust JSON extraction from LLM output.
        Handles markdown fences, leading text, trailing text.
        """
        # Strip ```json ... ``` fences
        cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Find first { ... } block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        raise AIServiceError(
            "Gemini returned non-parseable JSON.",
            detail={"raw_response": text[:500]},
        )
