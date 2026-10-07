"""Quick Groq connectivity check. Run: python scripts/check_groq.py"""
import os

import groq
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY", "").strip()
model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
print("key loaded:", bool(key), "| length:", len(key), "| starts with gsk_:", key.startswith("gsk_"))
print("model:", model)

try:
    client = groq.Groq(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": 'Reply with the JSON object {"ok": true}'}],
        response_format={"type": "json_object"},
        max_tokens=600,
        **({"reasoning_effort": "low"} if "gpt-oss" in model else {}),
    )
    print("SUCCESS:", resp.choices[0].message.content)
except groq.APIStatusError as exc:
    print("API ERROR status:", exc.status_code)
    print("message:", getattr(exc, "body", None) or exc)
except Exception as exc:  # noqa: BLE001
    print("OTHER ERROR:", type(exc).__name__, exc)