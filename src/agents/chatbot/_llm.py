"""Shared LLM call helper — tries Gemini, falls back to Groq on 429."""
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

_MODEL = "gemini-2.5-flash"
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

_last_used_llm: str = "gemini"


def get_last_llm() -> str:
    """Return 'gemini' or 'groq' — whichever LLM answered the most recent llm_call."""
    return _last_used_llm


def llm_call(prompt: str, max_tokens: int = 512) -> str:
    global _last_used_llm
    try:
        result = _client.models.generate_content(model=_MODEL, contents=prompt).text
        _last_used_llm = "gemini"
        return result
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            groq_key = os.environ.get("GROQ_API_KEY")
            if groq_key:
                try:
                    import groq as _groq_lib
                    client = _groq_lib.Groq(api_key=groq_key)
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                    )
                    _last_used_llm = "groq"
                    return resp.choices[0].message.content
                except Exception:
                    pass
        raise
