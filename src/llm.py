
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv(override=True)


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY .env mein nahi mili.")
    return genai.Client(api_key=api_key)


def generate_answer(question: str, context: str) -> str:
    if not question.strip():
        raise ValueError("Question khaali hai.")

    if not context.strip():
        return "I could not find the answer in the provided PDF."

    client = get_llm()
    primary = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    fallback = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash").strip()

    prompt = f"""
Answer only from the supplied PDF context.
If the answer is not supported by the context, say it could not be found.
Do not guess or use outside knowledge.

PDF CONTEXT:
{context}

QUESTION:
{question}
"""

    temporary_errors = (429, 500, 503, 504)
    last_error = None

    for model in dict.fromkeys([primary, fallback]):
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                answer = response.text
                return answer.strip() if answer else (
                    "Gemini ne koi text answer return nahi kiya."
                )

            except errors.APIError as exc:
                last_error = exc
                status = getattr(exc, "code", None)

                if status not in temporary_errors:
                    break

                if attempt == 0:
                    time.sleep(2)
                    continue

                break

    raise RuntimeError(
        "Gemini ke dono models se answer nahi mila. "
        f"Last API error: {last_error}"
    )
