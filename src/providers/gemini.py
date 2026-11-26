import os
import time
from typing import Union
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_CLIENT = None


def get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment")

    _CLIENT = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=300 * 1000,
            retry_options=types.HttpRetryOptions(
                attempts=5,
                initial_delay=1.0,
                http_status_codes=list((408, 429, 500, 502, 503, 504)),
            ),
        ),
    )
    return _CLIENT


def call_gemini(
    contents: Union[str, list[types.Content]],
    model: str = "gemini-3.5-flash-lite",
    thinking_level: str = "MINIMAL",
    temperature: float = 0.0,
    seed: int | None = None,
    system_instruction: str | None = None,
) -> tuple[str, float]:
    """Returns (response_text, latency_ms)."""
    client = get_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        seed=seed,
        system_instruction=system_instruction,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
    )

    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return response.text or "", latency_ms
