
from src.llm import get_llm

print("Testing LLM API...", flush=True)

try:
    llm = get_llm()
    response = llm.invoke("Reply with only: API working")
    print("SUCCESS:", response.content, flush=True)

except Exception as exc:
    print("API ERROR:", repr(exc), flush=True)
