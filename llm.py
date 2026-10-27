"""
LLM Configuration and Invocation Layer for College Helpdesk.
Provides centralized initialization of Google Gemini Flash via langchain-google-genai,
with environment variable configuration, secure error handling, and model fallback.
"""

import os
import logging
from typing import Any, List, Optional
import dotenv

# Load local environment variables from .env if present
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# Primary and fallback Gemini Flash models supported
DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.5-flash"]

_LLM_CACHE = {}


def get_api_key() -> Optional[str]:
    """
    Safely retrieve Google Gemini API key from environment or Streamlit secrets.
    Never logs or exposes the key value.
    """
    # 1. Check OS environment variable
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2. Check Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    return None


def get_model_name() -> str:
    """
    Get configurable Gemini Flash model identifier from environment or Streamlit secrets.
    """
    model = os.getenv("GEMINI_MODEL")
    if not model:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "GEMINI_MODEL" in st.secrets:
                model = str(st.secrets["GEMINI_MODEL"]).strip()
        except Exception:
            pass

    return model.strip() if model else DEFAULT_MODEL


def clean_response_text(content: Any) -> str:
    """
    Cleanly extract text from ChatGoogleGenerativeAI response.
    Handles raw strings as well as Gemini list of content parts:
    [{'type': 'text', 'text': '...'}].
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            elif hasattr(part, "text"):
                parts.append(part.text)
        return " ".join(parts).strip()

    return str(content).strip()


def get_llm(model_name: Optional[str] = None, temperature: float = 0.0):
    """
    Initialize or retrieve a cached ChatGoogleGenerativeAI instance.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "Google Gemini API key not found. Please configure GOOGLE_API_KEY "
            "in your environment variables, .env file, or Streamlit secrets."
        )

    selected_model = model_name or get_model_name()
    cache_key = (selected_model, temperature)

    if cache_key not in _LLM_CACHE:
        _LLM_CACHE[cache_key] = ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=api_key,
            temperature=temperature,
        )

    return _LLM_CACHE[cache_key]


def invoke_llm_with_fallback(messages: List[Any], temperature: float = 0.0) -> str:
    """
    Invoke Gemini Flash with the configured model and automatic fallback to alternative
    Flash models if a rate limit (429) or high-demand (503) error is encountered.
    """
    primary_model = get_model_name()
    candidate_models = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]

    last_error = None
    for model in candidate_models:
        try:
            llm = get_llm(model_name=model, temperature=temperature)
            response = llm.invoke(messages)
            text = clean_response_text(response.content)
            if text:
                return text
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            # If authentication failure, do not retry other models
            if "unauthenticated" in err_str or "api key not valid" in err_str:
                raise ValueError(
                    "Google Gemini API authentication failed. Please verify your "
                    "GOOGLE_API_KEY in the application secrets or .env file."
                ) from e
            # Log transient model errors and try next candidate
            logger.warning("Error with model %s: %s. Retrying with fallback...", model, e)
            continue

    raise RuntimeError(
        f"Unable to generate response from Gemini Flash after trying available models: {last_error}"
    )
