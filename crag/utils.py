import re


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from Qwen3 output."""
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return result.strip()
