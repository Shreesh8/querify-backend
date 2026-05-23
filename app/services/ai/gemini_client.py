"""
services/ai/gemini_client.py
Updated to use google-genai SDK (google.generativeai is deprecated).
"""

import json
import re
from typing import Any, Dict

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import AIServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeminiClient:

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def get_operation_spec(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                ),
            )
            return self._extract_json(response.text)
        except Exception as e:
            logger.error("gemini_call_failed", error=str(e))
            raise

    async def get_insights(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=1500,
                ),
            )
            return response.text
        except Exception as e:
            logger.error("gemini_insights_failed", error=str(e))
            raise AIServiceError(f"Gemini insights call failed: {str(e)}")

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
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
