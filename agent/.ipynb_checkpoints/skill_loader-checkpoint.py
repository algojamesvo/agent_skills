# agent/skill_loader.py
import os
import re
import importlib.util
from typing import Dict, Any, List, Tuple, Optional


def _parse_frontmatter(skill_md_text: str) -> Dict[str, str]:
    """
    Minimal YAML-frontmatter parser for:
      ---
      name: ...
      description: ...
      ---
    """
    m = re.search(r"^---\s*(.*?)\s*---\s*", skill_md_text, re.DOTALL | re.MULTILINE)
    if not m:
        raise ValueError("Missing YAML frontmatter in SKILL.md")
    block = m.group(1)

    out: Dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")

    if "name" not in out or "description" not in out:
        raise ValueError("Frontmatter must contain name and description")
    return out


def _load_python_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_valid_skill_dir(skill_dir: str) -> bool:
    """A folder is considered a skill folder if it contains SKILL.md."""
    if not os.path.isdir(skill_dir):
        return False
    if os.path.basename(skill_dir).startswith("."):
        return False
    return os.path.exists(os.path.join(skill_dir, "SKILL.md"))


def _to_responses_function_tool_schema(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an internal tool definition to Responses API tool schema.

    IMPORTANT:
    Responses API expects function tools as:
      {
        "type": "function",
        "name": "...",
        "description": "...",
        "parameters": {...},
        "strict": true
      }

    NOT the ChatCompletions style:
      {"type":"function","function":{"name":...}}
    """
    name = tool_def["name"]
    desc = tool_def.get("description", "")
    params = tool_def.get("parameters", {"type": "object", "properties": {}, "additionalProperties": False})

    schema = {
        "type": "function",
        "name": name,
        "description": desc,
        "parameters": params,
        # strict=True encourages the model to follow your JSON schema tightly
        "strict": True,
    }
    return schema


def load_skill(skill_dir: str) -> Dict[str, Any]:
    """
    Load a Claude-style skill folder and convert it into:
      - meta: {name, description}
      - tools: list of Responses API tool schemas
      - executors: { tool_name: python_callable }

    Conventions:
      - scripts/*.py may expose get_registered_tools(), returning a list of dicts:
          {"name": str, "description": str, "parameters": json_schema, "fn": callable}
    """
    skill_md_path = os.path.join(skill_dir, "SKILL.md")
    if not os.path.exists(skill_md_path):
        raise FileNotFoundError(f"Missing SKILL.md: {skill_md_path}")

    with open(skill_md_path, "r", encoding="utf-8") as f:
        skill_md_text = f.read()

    meta = _parse_frontmatter(skill_md_text)

    scripts_dir = os.path.join(skill_dir, "scripts")
    executors: Dict[str, Any] = {}
    tools_schema: List[Dict[str, Any]] = []

    if os.path.isdir(scripts_dir):
        for filename in os.listdir(scripts_dir):
            if filename.startswith("."):
                continue
            if not filename.endswith(".py"):
                continue

            path = os.path.join(scripts_dir, filename)
            mod = _load_python_module_from_path(f"{meta['name']}.{filename[:-3]}", path)

            if hasattr(mod, "get_registered_tools"):
                registered = mod.get_registered_tools()
                if not isinstance(registered, list):
                    raise TypeError(f"{path}: get_registered_tools() must return a list")

                for t in registered:
                    tool_name = t["name"]
                    fn = t["fn"]

                    # tool_name collision protection (optional but useful)
                    if tool_name in executors:
                        raise ValueError(
                            f"Tool name collision: '{tool_name}' already defined; "
                            f"conflict in skill_dir={skill_dir}"
                        )

                    executors[tool_name] = fn
                    tools_schema.append(_to_responses_function_tool_schema({
                        "name": tool_name,
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}, "additionalProperties": False}),
                    }))

    return {
        "meta": meta,
        "tools": tools_schema,
        "executors": executors,
        "skill_md": skill_md_text,
        "skill_dir": skill_dir,
    }


def load_all_skills(skills_root: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Discover and load all valid skill folders under skills_root.

    Fix #1: Skip non-skill folders like .ipynb_checkpoints by requiring SKILL.md.
    Fix #2: Produce Responses API tools schema for function tools.
    """
    tools: List[Dict[str, Any]] = []
    executors: Dict[str, Any] = {}
    skills_meta: List[Dict[str, str]] = []
    skills_full: List[Dict[str, Any]] = []

    if not os.path.isdir(skills_root):
        raise FileNotFoundError(f"skills_root not found or not a directory: {skills_root}")

    for name in os.listdir(skills_root):
        if name.startswith("."):
            continue

        skill_path = os.path.join(skills_root, name)
        if not _is_valid_skill_dir(skill_path):
            continue

        skill = load_skill(skill_path)
        skills_full.append(skill)
        skills_meta.append(skill["meta"])
        tools.extend(skill["tools"])
        executors.update(skill["executors"])

    runtime = {
        "executors": executors,
        "skills_meta": skills_meta,
        "skills_full": skills_full,  # includes full SKILL.md text if you need on-demand body loading
    }
    return tools, runtime


# --- Optional helper (recommended) ---
def append_response_output_items(input_list: List[Dict[str, Any]], resp: Any) -> None:
    """
    Recommended pattern for reasoning models:
    After each client.responses.create(...), append resp.output back into the next input.
    This preserves reasoning/trace items that some models expect for subsequent turns.

    Usage in run_agent.py:
        resp = client.responses.create(...)
        append_response_output_items(input_list, resp)

    Then append your function_call_output items to input_list and call again.
    """
    # resp.output in the Python SDK is typically a list of objects; we need dict form.
    # The SDK objects often have model_dump(). Fall back to __dict__ if needed.
    out_items = getattr(resp, "output", None)
    if not out_items:
        return

    for item in out_items:
        if hasattr(item, "model_dump"):
            input_list.append(item.model_dump())
        elif isinstance(item, dict):
            input_list.append(item)
        else:
            # best-effort fallback
            input_list.append(getattr(item, "__dict__", {"type": str(getattr(item, "type", "unknown"))}))
