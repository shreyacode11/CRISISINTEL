from utils.timeutil import now_ist, utc_now
import json
from google import genai
from config import Config

_client = genai.Client(api_key=Config.GEMINI_API_KEY)

PROMPT = """You are a crisis help classifier. Classify the user's help request.
Return ONLY valid JSON, no markdown, no extra text.

Schema:
{
  "help_type": "food" | "medical" | "shelter",
  "severity": "red" | "yellow" | "green",
  "reason": "short explanation"
}

Rules:
- red    = life threatening / immediate danger
- yellow = urgent but not immediate danger
- green  = low priority / informational

User message:
"""


def classify_help(message: str):
    resp = _client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=PROMPT + message,
    )
    text = resp.text.strip()

    # Strip code fences if any
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]

    try:
        data = json.loads(text)
    except Exception:
        data = {"help_type": "food", "severity": "yellow", "reason": "fallback"}

    # safety normalisation
    if data.get("help_type") not in ("food", "medical", "shelter"):
        data["help_type"] = "food"
    if data.get("severity") not in ("red", "yellow", "green"):
        data["severity"] = "yellow"

    # tag colour mapping
    data["tag_color"] = {"red": "red", "yellow": "yellow", "green": "green"}[data["severity"]]
    return data