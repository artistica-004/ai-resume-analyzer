"""Single Groq client wrapper.

Features:
- JSON mode (response_format={"type": "json_object"})
- Retries (max 3) with exponential backoff on rate limits / transient errors
- 413 'request too large' handling: automatically shrinks max_tokens and retries
- JSON repair fallback (fences, trailing commas, extra text)
- Re-ask with the Pydantic error message when validation fails
- Input length guard (truncation reported back to the caller)
- Timeouts
- logging instead of print; resume contents are NEVER logged
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Optional, Tuple, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# llama-3.3-70b-versatile is no longer available on Groq's free plan.
# Override with GROQ_MODEL in .env / HF Space variables to switch models.
DEFAULT_MODEL = "openai/gpt-oss-20b"

MAX_RETRIES = 3
MIN_MAX_TOKENS = 1500
# Conservative char budget per input so prompt + output fits the context
# and stays friendly to free-tier tokens-per-minute limits (~4 chars/token).
DEFAULT_MAX_INPUT_CHARS = 14_000


class LLMError(Exception):
    """User-facing LLM error. The message is safe to show in the UI."""


def _get_api_key() -> str:
    """Read GROQ_API_KEY from env (.env locally) or Streamlit secrets (HF Spaces)."""
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            import streamlit as st  # imported lazily so core/ works without Streamlit

            key = st.secrets.get("GROQ_API_KEY", "")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - secrets file may simply not exist
            key = ""
    if not key:
        raise LLMError(
            "GROQ_API_KEY is not set. Add it to a .env file locally or as a "
            "Secret in your Hugging Face Space settings."
        )
    return key.strip()


def truncate_text(text: str, max_chars: int = DEFAULT_MAX_INPUT_CHARS) -> Tuple[str, bool]:
    """Cut text at a line boundary below max_chars. Returns (text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars * 0.8:
        cut = cut[:last_nl]
    return cut, True


def repair_json(raw: str) -> dict:
    """Best-effort conversion of a messy model response into a dict."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        s = s[start: end + 1]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    return json.loads(s)


def _error_text(exc: Exception) -> str:
    """Groq's own error message (safe to log: it never contains our prompt text)."""
    try:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                return str(err.get("message", ""))[:300]
        return str(getattr(exc, "message", "") or exc)[:300]
    except Exception:  # noqa: BLE001
        return type(exc).__name__


class LLMClient:
    """Thin, testable wrapper around the Groq SDK."""

    def __init__(self, model: Optional[str] = None, timeout: float = 90.0) -> None:
        from groq import Groq  # local import keeps tests importable without the SDK

        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self._client = Groq(api_key=_get_api_key(), timeout=timeout, max_retries=0)

    def _call(self, messages: List[dict], temperature: float, max_tokens: int) -> str:
        """One API call with retry/backoff for transient errors."""
        import groq

        delay = 2.0
        json_mode = True
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                extra = {}
                if "gpt-oss" in self.model:
                    # Reasoning model: keep hidden "thinking" short so the JSON answer
                    # is not cut off and free-tier token limits last longer.
                    extra["reasoning_effort"] = "low"
                if json_mode:
                    extra["response_format"] = {"type": "json_object"}
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=42,
                    **extra,
                )
                content = resp.choices[0].message.content or ""
                logger.info("LLM call ok (model=%s, attempt=%d, json_mode=%s, chars=%d)",
                            self.model, attempt, json_mode, len(content))
                if not content.strip():
                    # Reasoning used the whole budget: retry with more room
                    max_tokens = min(max_tokens * 2, 8000)
                    logger.warning("Empty response; retrying with max_tokens=%d", max_tokens)
                    continue
                return content

            except groq.AuthenticationError:
                raise LLMError("The GROQ_API_KEY is invalid. Please check it in your .env / Space secrets.")

            except groq.RateLimitError as exc:
                msg = _error_text(exc)
                logger.warning("Rate limited (attempt %d): %s", attempt, msg)
                if "per day" in msg.lower() or "tpd" in msg.lower():
                    raise LLMError(
                        "The Groq free-tier DAILY token limit for this model is used up. Try again "
                        "tomorrow, or set GROQ_MODEL to another model in .env."
                    )
                if attempt == MAX_RETRIES:
                    raise LLMError("The Groq free-tier rate limit was reached. Please wait about a minute and retry.")
                delay = max(delay, 10.0)

            except (groq.APIConnectionError, groq.APITimeoutError):
                logger.warning("Connection/timeout error (attempt %d)", attempt)
                if attempt == MAX_RETRIES:
                    raise LLMError("Could not reach the Groq API. Check your internet connection and retry.")

            except groq.BadRequestError as exc:
                msg = _error_text(exc)
                logger.warning("Bad request (attempt %d): %s", attempt, msg)
                if "json_validate_failed" in str(getattr(exc, "body", "")) or "validate json" in msg.lower():
                    # Strict JSON mode rejected the output: retry without it, with more room.
                    # chat_json() still repairs and Pydantic-validates whatever comes back.
                    json_mode = False
                    max_tokens = min(max_tokens * 2, 8000)
                    continue
                if "model" in msg.lower() and ("not exist" in msg.lower() or "decommissioned" in msg.lower()):
                    raise LLMError(f"The model '{self.model}' is not available. Set GROQ_MODEL to a current Groq model.")
                if attempt == MAX_RETRIES:
                    raise LLMError("The AI model rejected the request. Try a shorter resume/JD and retry.")

            except groq.APIStatusError as exc:
                msg = _error_text(exc)
                logger.warning("API status error %s (attempt %d): %s", exc.status_code, attempt, msg)
                if exc.status_code == 404:
                    raise LLMError(f"The model '{self.model}' was not found. Set GROQ_MODEL to a model "
                                   "listed by scripts/list_models.py.")
                if exc.status_code == 413:
                    if max_tokens > MIN_MAX_TOKENS:
                        max_tokens = max(MIN_MAX_TOKENS, max_tokens // 2)
                        logger.info("Retrying with max_tokens=%d", max_tokens)
                        continue
                    raise LLMError(
                        "Your resume + job description are too large for the Groq free tier in one request. "
                        "Try a shorter JD or another model."
                    )
                if attempt == MAX_RETRIES:
                    raise LLMError(f"The AI service returned an error ({exc.status_code}). Please try again shortly.")

            time.sleep(delay)
            delay *= 2
        raise LLMError("The AI service did not respond correctly after several attempts. Please retry.")

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model_cls: Type[T],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> T:
        """Call the LLM and return a validated Pydantic object.

        If JSON parsing or Pydantic validation fails, the error is sent back to
        the model so it can correct itself (up to MAX_RETRIES times).
        """
        messages: List[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(1, MAX_RETRIES + 1):
            raw = self._call(messages, temperature, max_tokens)
            try:
                data = repair_json(raw)
                return model_cls.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)[:1500]
                logger.warning("Output invalid for %s (attempt %d): %s",
                               model_cls.__name__, attempt, type(exc).__name__)
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw[:6000]},
                    {"role": "user", "content": (
                        "Your previous answer was not valid for the required schema. "
                        f"Error:\n{last_error}\n\nReturn ONLY one corrected JSON object "
                        "that follows the schema exactly. Do not add new facts."
                    )},
                ]
        raise LLMError("The AI returned an unexpected format three times. Please click the button again.")