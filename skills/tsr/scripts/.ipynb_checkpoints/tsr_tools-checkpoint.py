# skills/tsr/scripts/tsr_tools.py
"""
TSR tools (Claude-skill-style scripts) exposed as OpenAI-callable function tools.

This module follows the same "tool registry" pattern as string-utils:
- Use @tool(...) decorator to register tools with name/description/parameters
- Expose get_registered_tools() for the Skill Loader/Adapter

Tools provided:
- tsr_extract(image_path, backend, output_format, max_tokens)
- otsl_to_html(otsl_text)
- save_html(html, out_path)

Backends (defaults align with your test scripts):
- backend="tsr"        -> base_url http://vllm-vllm-vlm-tsr-1:8000/v1, model agilesoda/vlm-tsr
- backend="perception" -> base_url http://vllm-vllm-vlm-perception-1:8000/v1, model nanonets/Nanonets-OCR2-3B
"""

from __future__ import annotations

import os
import re
import json
import base64
from typing import Any, Dict, Callable, List, Optional


FORCE_TSR_BACKEND = os.getenv("TSR_FORCE_BACKEND", "").strip().lower()


from openai import OpenAI

# Optional: use your existing conversion util if available in PYTHONPATH / skill bundle
import importlib.util
import os

def _load_local_otsl_utils():
    here = os.path.dirname(__file__)
    path = os.path.join(here, "otsl_utils.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("tsr_otsl_utils", path)
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    _otsl_mod = _load_local_otsl_utils()
    convert_otsl_to_html = getattr(_otsl_mod, "convert_otsl_to_html", None) if _otsl_mod else None
    OTSL_FCEL = getattr(_otsl_mod, "OTSL_FCEL", "<fcel>") if _otsl_mod else "<fcel>"
    OTSL_ECEL = getattr(_otsl_mod, "OTSL_ECEL", "<ecel>") if _otsl_mod else "<ecel>"
    OTSL_NL = getattr(_otsl_mod, "OTSL_NL", "<nl>") if _otsl_mod else "<nl>"
except Exception:
    convert_otsl_to_html = None  # type: ignore
    OTSL_FCEL, OTSL_ECEL, OTSL_NL = "<fcel>", "<ecel>", "<nl>"



# ---------------------------
# Tool registry (same pattern as string-utils)
# ---------------------------

_TOOL_REGISTRY: List[Dict[str, Any]] = []


def tool(name: str, description: str, parameters: Dict[str, Any]):
    """
    Decorator to mark a function as a tool.
    parameters must be a JSON Schema object for tool arguments.
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


# ---------------------------
# Helpers
# ---------------------------

DEFAULT_TSR_BASE_URL = os.getenv("TSR_BASE_URL", "http://vllm-vllm-vlm-tsr-1:8000/v1")
DEFAULT_TSR_MODEL = os.getenv("TSR_MODEL", "agilesoda/vlm-tsr")

DEFAULT_PERCEPTION_BASE_URL = os.getenv("PERCEPTION_BASE_URL", "http://vllm-vllm-vlm-perception-1:8000/v1")
DEFAULT_PERCEPTION_MODEL = os.getenv("PERCEPTION_MODEL", "nanonets/Nanonets-OCR2-3B")

DEFAULT_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

SYSTEM_PROMPT_OTSL = os.getenv(
    "TSR_SYSTEM_PROMPT",
    "You are an AI specialized in recognizing and extracting table from images. "
    "Your mission is to analyze the table image and generate the result in OTSL format "
    "using specified tags. Output only the results without any other words and explanation."
)

DEFAULT_IMAGE_MIME = os.getenv("TSR_IMAGE_MIME", "image/png")


import html as _html

# Detect OTSL tags in both escaped and unescaped forms, case-insensitive.
# Matches: <fcel>  </fcel> doesn't exist, but we only need open tags.
# Also matches: &lt;fcel&gt;
_OTSL_TAG_RE = re.compile(r"(?:<|&lt;)\s*(fcel|ecel|nl|lcel|ucel|xcel)\s*(?:>|&gt;)", re.IGNORECASE)

def _strip_pre_wrappers(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\s*<pre>\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*</pre>\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()

def _normalize_otsl(raw: str) -> str:
    """
    Normalize TSR backend output to the exact tag format expected by otsl_utils:
    - remove <pre> wrapper
    - unescape HTML entities: &lt;fcel&gt; -> <fcel>
    - keep lowercase tag tokens like <fcel>, <nl>, ...
    """
    s = _strip_pre_wrappers(raw)
    s = _html.unescape(s)  # IMPORTANT: &lt;fcel&gt; -> <fcel>
    return s.strip()



def _encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _looks_like_otsl(text: str) -> bool:
    """
    Robust OTSL detection:
    - supports <fcel>...<nl>... style
    - supports HTML-escaped &lt;fcel&gt;...&lt;nl&gt;...
    - supports case-insensitive tags
    """
    if not text:
        return False
    return _OTSL_TAG_RE.search(text) is not None



def _call_vlm_extract(
    *,
    base_url: str,
    model: str,
    image_b64: str,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    prompt_text: str = "Extract table from this image."
) -> str:
    """
    Calls an OpenAI-compatible /v1/chat/completions endpoint (e.g. vLLM).
    """
    client = OpenAI(base_url=base_url, api_key=DEFAULT_API_KEY)

    result = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_OTSL},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{DEFAULT_IMAGE_MIME};base64,{image_b64}"},
                    },
                ],
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return result.choices[0].message.content or ""


# ---------------------------
# Tools
# ---------------------------

@tool(
    name="tsr_extract",
    description=(
        "Extract table structure from a table image using TSR/perception backends. "
        "Returns raw model output plus detected format and (when possible) both OTSL and HTML."
    ),
    parameters={
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "Local path to the input table image (png/jpg)."},
            "backend": {
                "type": "string",
                "description": "Which backend to use: 'tsr' (dedicated TSR model) or 'perception' (OCR/perception model).",
                "enum": ["tsr", "perception"],
                "default": "tsr",
            },
            "output_format": {
                "type": "string",
                "description": "Desired output: 'auto' (prefer both), 'otsl', or 'html'.",
                "enum": ["auto", "otsl", "html"],
                "default": "auto",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max tokens for the VLM output (increase if large tables are truncated).",
                "default": 8192,
            },
            "prompt_text": {
                "type": "string",
                "description": "User prompt text sent alongside the image.",
                "default": "Extract table from this image.",
            },
        },
        "required": ["image_path"],
        "additionalProperties": False,
    },
)
def tsr_extract(
    image_path: str,
    backend: str = "tsr",
    output_format: str = "auto",
    max_tokens: int = 8192,
    prompt_text: str = "Extract table from this image.",
) -> Dict[str, Any]:
    # Force backend for evaluation if TSR_FORCE_BACKEND is set
    if FORCE_TSR_BACKEND in ("tsr", "perception"):
        backend = FORCE_TSR_BACKEND

    # print("#"*32)
    # print(f"backend: {backend}")
    # print("#"*32)
    
    if not os.path.exists(image_path):
        return {"error": f"image_path not found: {image_path}"}

    if backend == "tsr":
        base_url, model = DEFAULT_TSR_BASE_URL, DEFAULT_TSR_MODEL
    elif backend == "perception":
        base_url, model = DEFAULT_PERCEPTION_BASE_URL, DEFAULT_PERCEPTION_MODEL
    else:
        return {"error": f"Invalid backend: {backend}"}

    try:
        # print("#"*32)
        # print(f"call base_url: {base_url}")
        # print(f"call model: {model}")
        # print("#"*32)
    
        
        img_b64 = _encode_image_to_base64(image_path)
        raw = _call_vlm_extract(
            base_url=base_url,
            model=model,
            image_b64=img_b64,
            temperature=0.0,
            max_tokens=max_tokens,
            prompt_text=prompt_text,
        )
    except Exception as e:
        return {"error": str(e), "backend_used": backend, "model": model, "base_url": base_url}

    raw = (raw or "").strip()

    detected_format = "otsl" if _looks_like_otsl(raw) else "html"

    otsl_text = ""
    html_text = ""

    if detected_format == "otsl":
        # Normalize TSR OTSL output to the exact tag format expected by otsl_utils:
        # - strip <pre> wrappers
        # - html.unescape: &lt;fcel&gt; -> <fcel>
        otsl_text = _normalize_otsl(raw)

        # If user wants HTML and we detected OTSL, try to convert (if converter exists)
        if output_format in ("auto", "html"):
            if convert_otsl_to_html is None:
                html_text = ""
            else:
                try:
                    html_text = convert_otsl_to_html(otsl_text)  # type: ignore[misc]
                except Exception as e:
                    return {
                        "error": f"OTSL->HTML conversion failed: {e}",
                        "raw_text": raw,
                        "detected_format": detected_format,
                        "otsl": otsl_text,
                        "html": "",
                        "backend_used": backend,
                        "model": model,
                        "base_url": base_url,
                    }
    else:
        # Treat as HTML/plain output
        html_text = raw

    # Enforce output_format selection
    if output_format == "otsl":
        html_text = ""
    elif output_format == "html":
        otsl_text = ""

    return {
        "raw_text": raw,
        "detected_format": detected_format,
        "otsl": otsl_text,
        "html": html_text,
        "backend_used": backend,
        "model": model,
        "base_url": base_url,
    }



@tool(
    name="otsl_to_html",
    description="Convert OTSL text (tags like <fcel>, <ecel>, <nl>) into HTML table.",
    parameters={
        "type": "object",
        "properties": {
            "otsl_text": {"type": "string", "description": "OTSL text to convert."}
        },
        "required": ["otsl_text"],
        "additionalProperties": False,
    },
)
def otsl_to_html(otsl_text: str) -> Dict[str, Any]:
    if not otsl_text:
        return {"error": "otsl_text is empty", "html": ""}

    if convert_otsl_to_html is None:
        return {
            "error": "convert_otsl_to_html not available. Bundle/import otsl_utils in this skill to enable conversion.",
            "html": "",
        }

    try:
        html = convert_otsl_to_html(otsl_text)  # type: ignore[misc]
        return {"html": html}
    except Exception as e:
        return {"error": str(e), "html": ""}


@tool(
    name="save_html",
    description="Save an HTML string to a file path.",
    parameters={
        "type": "object",
        "properties": {
            "html": {"type": "string", "description": "HTML content to save."},
            "out_path": {"type": "string", "description": "Output file path, e.g. outputs/table.html."},
        },
        "required": ["html", "out_path"],
        "additionalProperties": False,
    },
)
def save_html(html: str, out_path: str) -> Dict[str, Any]:
    if not out_path:
        return {"error": "out_path is empty"}

    # Create parent dirs if needed
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html or "")
        return {"out_path": out_path}
    except Exception as e:
        return {"error": str(e), "out_path": out_path}
