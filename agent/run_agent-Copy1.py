import os
import json
import re
from openai import OpenAI

from skill_loader import load_all_skills, append_response_output_items

MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
SKILLS_ROOT = os.path.join(os.path.dirname(__file__), "..", "skills")


def _strip_frontmatter(md: str) -> str:
    """Remove YAML frontmatter from SKILL.md and return only the body."""
    m = re.search(r"^---\s*.*?\s*---\s*", md, re.DOTALL | re.MULTILINE)
    if not m:
        return md.strip()
    return md[m.end():].strip()


def _extract_reference_paths(skill_body: str) -> list[str]:
    """
    Extract references paths mentioned in SKILL.md body.
    Supported patterns:
      - references/foo.md
      - (references/foo.md) in markdown links
      - [text](references/foo.md)
    Returns unique paths preserving order.
    """
    # Find "references/....md" anywhere
    paths = re.findall(r"(references/[A-Za-z0-9_\-./]+\.md)", skill_body)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _safe_read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _get_item_type(item):
    if isinstance(item, dict):
        return item.get("type")
    return getattr(item, "type", None)


def _get_call_name(call):
    if isinstance(call, dict):
        return call.get("name")
    return getattr(call, "name", None)


def _get_call_arguments(call):
    if isinstance(call, dict):
        return call.get("arguments")
    return getattr(call, "arguments", None)


def _get_call_id(call):
    if isinstance(call, dict):
        return call.get("call_id")
    return getattr(call, "call_id", None)


def run():
    client = OpenAI(
        base_url="http://vllm-vllm-llm-120b-1:8000/v1",
        api_key="EMPTY"
    )

    tools, runtime = load_all_skills(SKILLS_ROOT)
    executors = runtime["executors"]

    # --- Build tool -> skill mapping, skill_name -> skill_body mapping, skill_name -> skill_dir mapping ---
    tool_to_skill = {}          # "slugify" -> "string-utils"
    skill_name_to_body = {}     # "string-utils" -> "<SKILL.md body>"
    skill_name_to_dir = {}      # "string-utils" -> ".../skills/string-utils"

    for skill in runtime.get("skills_full", []):
        meta = skill.get("meta", {})
        sname = meta.get("name")
        if not sname:
            continue

        skill_dir = skill.get("skill_dir")
        if skill_dir:
            skill_name_to_dir[sname] = skill_dir

        skill_md = skill.get("skill_md", "")
        skill_name_to_body[sname] = _strip_frontmatter(skill_md)

        for t in skill.get("tools", []):
            tname = t.get("name")  # Responses tool schema uses top-level "name"
            if tname:
                tool_to_skill[tname] = sname

    # --- Metadata hint (level-1 disclosure) ---
    skills_meta = runtime["skills_meta"]
    skills_hint = "\n".join([f"- {m['name']}: {m['description']}" for m in skills_meta])

    instructions = f"""
You are an agent that can call tools to get deterministic results.
Available skill metadata:
{skills_hint}

Rules:
- If the user request matches a tool, call it.
- After tool outputs are provided, produce the final answer to the user.
""".strip()

    user_text = "Slugify this title and also count words: 'TSR Agent: Input → Agent → Structured Tables'"
    input_list = [{"role": "user", "content": user_text}]

    loaded_skills = set()                 # skills whose SKILL.md body injected
    loaded_references = set()             # (skill_name, ref_rel_path) injected

    while True:
        resp = client.responses.create(
            model=MODEL,
            instructions=instructions,
            tools=tools,
            input=input_list,
        )

        # Keep model's output items (reasoning / tool calls) in context
        append_response_output_items(input_list, resp)

        out_items = getattr(resp, "output", []) or []
        tool_calls = [it for it in out_items if _get_item_type(it) == "function_call"]

        if not tool_calls:
            break

        # 1) Load SKILL.md body on-demand when a tool call belongs to that skill
        newly_loaded_skills = []
        for call in tool_calls:
            tname = _get_call_name(call)
            if not tname:
                continue

            sname = tool_to_skill.get(tname)
            if not sname or sname in loaded_skills:
                continue

            body = skill_name_to_body.get(sname, "").strip()
            if body:
                input_list.append({
                    "role": "system",
                    "content": f"[Loaded SKILL.md body for skill '{sname}']\n\n{body}"
                })
                loaded_skills.add(sname)
                newly_loaded_skills.append(sname)

        # 2) References on-demand:
        # Only load references IF the SKILL.md body explicitly mentions references/<file>.md
        # and only after that skill body has been loaded (now or earlier).
        skills_to_check_refs = set(newly_loaded_skills)

        # (Optional) also check skills already loaded, because tool calls might continue later.
        # This makes it robust if you update SKILL.md and reuse the same process.
        for call in tool_calls:
            tname = _get_call_name(call)
            if not tname:
                continue
            sname = tool_to_skill.get(tname)
            if sname and sname in loaded_skills:
                skills_to_check_refs.add(sname)

        for sname in skills_to_check_refs:
            body = skill_name_to_body.get(sname, "")
            if not body:
                continue

            ref_paths = _extract_reference_paths(body)
            if not ref_paths:
                continue

            skill_dir = skill_name_to_dir.get(sname)
            if not skill_dir:
                continue

            for rel_ref in ref_paths:
                key = (sname, rel_ref)
                if key in loaded_references:
                    continue

                abs_ref = os.path.join(skill_dir, rel_ref)
                if not os.path.exists(abs_ref):
                    # If the body references a missing file, inject a warning (optional)
                    input_list.append({
                        "role": "system",
                        "content": f"[Reference missing for skill '{sname}'] {rel_ref} not found at {abs_ref}"
                    })
                    loaded_references.add(key)
                    continue

                ref_text = _safe_read_text(abs_ref).strip()
                if ref_text:
                    input_list.append({
                        "role": "system",
                        "content": f"[Loaded reference for skill '{sname}': {rel_ref}]\n\n{ref_text}"
                    })
                loaded_references.add(key)

        # 3) Execute tool calls and append outputs
        for call in tool_calls:
            name = _get_call_name(call)
            args = _get_call_arguments(call)
            call_id = _get_call_id(call)

            if not name or not call_id:
                input_list.append({
                    "type": "function_call_output",
                    "call_id": call_id or "missing_call_id",
                    "output": json.dumps({"error": "Malformed function_call item"}, ensure_ascii=False),
                })
                continue

            if name not in executors:
                output = {"error": f"Tool not found: {name}"}
            else:
                try:
                    if isinstance(args, str):
                        args_obj = json.loads(args) if args else {}
                    elif isinstance(args, dict):
                        args_obj = args
                    else:
                        args_obj = {}

                    output = executors[name](**args_obj)
                except Exception as e:
                    output = {"error": str(e)}

            input_list.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(output, ensure_ascii=False),
            })

    print(resp.output_text)


if __name__ == "__main__":
    run()
