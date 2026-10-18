import os
import requests
import logging

from .prompts import SYSTEM_PROMPT

# We'll use environment variables directly for simplicity
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-r1:free"

_cached_settings = None

def _get_settings():
    global _cached_settings
    if _cached_settings is None:
        try:
            from .settings import Settings
            _cached_settings = Settings()
        except Exception:
            _cached_settings = False
    return _cached_settings

def generate_answer(question: str, code_chunks: list):
    """Call the OpenRouter chat completion API to get an answer for the question, with given code context."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("OPENROUTER_MODEL")
    if not api_key:
        settings = _get_settings()
        if settings:
            api_key = getattr(settings, "OPENROUTER_API_KEY", None)
            model = getattr(settings, "OPENROUTER_MODEL", model)
    if not api_key:
        raise RuntimeError("OpenRouter API key is not set. Please set OPENROUTER_API_KEY.")
    if not model:
        settings = _get_settings()
        model = getattr(settings, "OPENROUTER_MODEL", None) if settings else None
        model = model or DEFAULT_MODEL
    
    # Include relevant code context in the user message
    if code_chunks:
        context_str = "\n".join([f"File: {chunk['meta'].get('path')} (L{chunk['meta'].get('start_line')}-L{chunk['meta'].get('end_line')}):\n{chunk['text']}" for chunk in code_chunks])
        user_message = f"QUESTION: {question}\n\nRelevant Code:\n{context_str}"
    else:
        user_message = f"QUESTION: {question}\n\n(No relevant code found in the repository.)"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    # Make the API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.2
    }
    logging.info(f"Sending request to OpenRouter model '{model}'...")
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30)
    except Exception as e:
        logging.error(f"OpenRouter API request failed: {e}")
        return "*(Failed to query the language model.)*"
    if response.status_code != 200:
        logging.error(f"OpenRouter API returned error: {response.status_code}, {response.text}")
        return "*(Error from language model API.)*"
    resp_json = response.json()
    try:
        answer = resp_json['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Unexpected response format: {e}")
        return "*(Received an unexpected response from the model.)*"
    return answer.strip()
