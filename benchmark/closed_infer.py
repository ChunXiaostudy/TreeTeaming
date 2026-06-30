import os
import json
import base64
from typing import List


def _read_image_as_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower().lstrip(".") or "png"
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def _call_openai_chat(model: str, messages: List[dict]) -> str:
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("CLOSEDAPI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CLOSEDAPI_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError(
            "Missing API credentials. Set OPENAI_BASE_URL (or CLOSEDAPI_BASE_URL) "
            "and OPENAI_API_KEY (or CLOSEDAPI_API_KEY)."
        )
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(f"openai_sdk_missing:{type(e).__name__}:{e}") from e
    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content or ""
