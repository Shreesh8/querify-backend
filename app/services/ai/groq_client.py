"""
services/ai/groq_client.py
Groq-backed client matching GeminiClient's interface
(get_operation_spec, get_insights, _extract_json).
"""
import json
import re
from typing import Any, Dict

from groq import Groq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.exceptions import AIServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)


class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def get_operation_spec(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            return self._extract_json(response.choices[0].message.content)
        except Exception as e:
            logger.error("groq_call_failed", error=str(e))
            raise

    async def get_insights(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("groq_insights_failed", error=str(e))
            raise AIServiceError(f"Groq insights call failed: {str(e)}")

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
            "Groq returned non-parseable JSON.",
            detail={"raw_response": text[:500]},
        )
