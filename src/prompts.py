from pathlib import Path

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt"
_system_prompt_cache = None


def load_system_prompt() -> str:
    global _system_prompt_cache
    if _system_prompt_cache is None:
        try:
            _system_prompt_cache = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise RuntimeError(f"System prompt file not found: {_SYSTEM_PROMPT_PATH}") from exc
    return _system_prompt_cache


SYSTEM_PROMPT = load_system_prompt()    
