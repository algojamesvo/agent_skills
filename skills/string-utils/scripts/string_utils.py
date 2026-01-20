import re
import unicodedata
from typing import Dict, Any, Callable, List

# --- Minimal "tool registry" decorator (keeps scripts close to Claude Skill idea) ---

_TOOL_REGISTRY: List[Dict[str, Any]] = []

def tool(name: str, description: str, parameters: Dict[str, Any]):
    """
    Decorator to mark a function as an OpenAI-callable tool.
    `parameters` must be a JSON Schema object for function arguments.
    """
    def wrap(fn: Callable[..., Any]):
        _TOOL_REGISTRY.append({
            "name": name,
            "description": description,
            "parameters": parameters,
            "fn": fn,
        })
        return fn
    return wrap

def get_registered_tools():
    return _TOOL_REGISTRY

# --- Actual executable code ---

@tool(
    name="slugify",
    description="Convert text into a URL-friendly slug (lowercase, hyphens, ASCII).",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Input text to slugify."},
            "max_length": {"type": "integer", "description": "Optional max length of slug.", "default": 80},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
)
def slugify(text: str, max_length: int = 80) -> Dict[str, Any]:
    # Normalize unicode -> ASCII
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()

    # Replace non-alnum with hyphen
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")

    if max_length and len(text) > max_length:
        text = text[:max_length].rstrip("-")

    return {"slug": text}

@tool(
    name="count_words",
    description="Count words and characters in a text.",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Input text."}
        },
        "required": ["text"],
        "additionalProperties": False,
    },
)
def count_words(text: str) -> Dict[str, Any]:
    words = re.findall(r"\S+", text.strip())
    return {"words": len(words), "chars": len(text)}
