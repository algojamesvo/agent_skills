import os
import json
import re
import argparse
from typing import List, Dict, Any, Optional
from openai import OpenAI

from skill_loader import load_all_skills, append_response_output_items

MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
SKILLS_ROOT = os.path.join(os.path.dirname(__file__), "..", "skills")

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
PROFILE_PATH = os.getenv("AGENT_PROFILE_PATH", os.path.join(PROMPTS_DIR, "profile.md"))
GOALS_PATH = os.getenv("AGENT_GOALS_PATH", os.path.join(PROMPTS_DIR, "goals.md"))
INSTRUCTIONS_PATH = os.getenv("AGENT_INSTRUCTIONS_PATH", os.path.join(PROMPTS_DIR, "instructions.md"))

DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://vllm-vllm-llm-120b-1:8000/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")


def _safe_read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _strip_frontmatter(md: str) -> str:
    """Remove YAML frontmatter from SKILL.md and return only the body."""
    m = re.search(r"^---\s*.*?\s*---\s*", md, re.DOTALL | re.MULTILINE)
    if not m:
        return md.strip()
    return md[m.end():].strip()


def _extract_reference_paths(skill_body: str) -> List[str]:
    """
    Extract references paths mentioned in SKILL.md body.
    Matches references/<something>.md anywhere.
    Returns unique paths preserving order.
    """
    paths = re.findall(r"(references/[A-Za-z0-9_\-./]+\.md)", skill_body)
    seen = set()
    out: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


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


def _load_agent_prompts(skills_hint: str) -> str:
    """
    Compose the final 'instructions' string from:
      - profile.md
      - goals.md
      - instructions.md (templated with {skills_hint})
    """
    profile = _safe_read_text(PROFILE_PATH).strip()
    goals = _safe_read_text(GOALS_PATH).strip()
    inst_tpl = _safe_read_text(INSTRUCTIONS_PATH).strip()
    inst = inst_tpl.format(skills_hint=skills_hint)

    # One combined instructions string
    combined = "\n\n".join([
        profile,
        goals,
        inst,
    ]).strip()
    return combined


def _load_jsonl_inputs(path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL where each line is either:
      {"input": "..."}  (recommended)
    or a raw JSON string value.
    """
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                cases.append({"input": obj})
            elif isinstance(obj, dict) and "input" in obj:
                cases.append(obj)
            else:
                raise ValueError(f"Invalid JSONL line {i}: expected string or object with 'input'")
    return cases


def run_single_case(
    *,
    client: OpenAI,
    tools: List[Dict[str, Any]],
    runtime: Dict[str, Any],
    instructions: str,
    user_text: str,
) -> str:
    """
    Run one agent loop for a single user_text.
    Implements:
      - append resp.output into input_list (reasoning-safe)
      - load SKILL.md body on tool-call (on-demand)
      - load references on-demand if mentioned in SKILL.md body
      - execute function tools locally
    Returns final resp.output_text.
    """
    executors = runtime["executors"]

    # --- Build mappings ---
    tool_to_skill: Dict[str, str] = {}
    skill_name_to_body: Dict[str, str] = {}
    skill_name_to_dir: Dict[str, str] = {}

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
            tname = t.get("name")
            if tname:
                tool_to_skill[tname] = sname

    input_list: List[Dict[str, Any]] = [{"role": "user", "content": user_text}]
    loaded_skills = set()
    loaded_references = set()

    while True:
        resp = client.responses.create(
            model=MODEL,
            instructions=instructions,
            tools=tools,
            input=input_list,
        )

        # Keep model output items in context for next turn
        append_response_output_items(input_list, resp)

        out_items = getattr(resp, "output", []) or []
        tool_calls = [it for it in out_items if _get_item_type(it) == "function_call"]

        if not tool_calls:
            break

        # 1) Load SKILL.md body on-demand (once per skill)
        newly_loaded_skills: List[str] = []
        for call in tool_calls:
            tname = _get_call_name(call)
            if not tname:
                continue

            sname = tool_to_skill.get(tname)
            if not sname or sname in loaded_skills:
                continue

            body = (skill_name_to_body.get(sname) or "").strip()
            if body:
                input_list.append({
                    "role": "system",
                    "content": f"[Loaded SKILL.md body for skill '{sname}']\n\n{body}",
                })
                loaded_skills.add(sname)
                newly_loaded_skills.append(sname)

        # 2) References on-demand:
        # Load only references explicitly mentioned in skill body, only after skill body is loaded.
        skills_to_check_refs = set(newly_loaded_skills)
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
                    input_list.append({
                        "role": "system",
                        "content": f"[Reference missing for skill '{sname}'] {rel_ref} not found at {abs_ref}",
                    })
                    loaded_references.add(key)
                    continue

                ref_text = _safe_read_text(abs_ref).strip()
                if ref_text:
                    input_list.append({
                        "role": "system",
                        "content": f"[Loaded reference for skill '{sname}': {rel_ref}]\n\n{ref_text}",
                    })
                loaded_references.add(key)

        # 3) Execute tool calls locally and append outputs
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

    return resp.output_text


def main():
    parser = argparse.ArgumentParser(description="Run OpenAI-style tool-using agent with Claude-skill folders.")
    parser.add_argument("--input", type=str, default=None, help="Single input string to run.")
    parser.add_argument("--input_file", type=str, default=None, help="JSONL file with lines like {'input': '...'}")
    args = parser.parse_args()

    # Load skills
    tools, runtime = load_all_skills(SKILLS_ROOT)

    # Build metadata hint (level-1 disclosure)
    skills_meta = runtime["skills_meta"]
    skills_hint = "\n".join([f"- {m['name']}: {m['description']}" for m in skills_meta])

    # Compose agent instructions from prompt files
    instructions = _load_agent_prompts(skills_hint)

    # Build client
    client = OpenAI(base_url=DEFAULT_BASE_URL, api_key=DEFAULT_API_KEY)

    # Build cases
    cases: List[Dict[str, Any]] = []
    if args.input:
        cases = [{"input": args.input}]
    elif args.input_file:
        cases = _load_jsonl_inputs(args.input_file)
    else:
        # Default demo cases if nothing provided
        cases = [
            {"input": "Slugify this title and also count words: 'TSR Agent: Input → Agent → Structured Tables'"},
        ]

    # Run cases sequentially
    for idx, case in enumerate(cases, start=1):
        user_text = case["input"]
        title = case.get("id") or case.get("title") or f"case_{idx}"

        print("=" * 80)
        print(f"[{idx}/{len(cases)}] {title}")
        print("- Input:")
        print(user_text)

        out = run_single_case(
            client=client,
            tools=tools,
            runtime=runtime,
            instructions=instructions,
            user_text=user_text,
        )

        print("- Output:")
        print(out.strip() if out else out)


if __name__ == "__main__":
    main()
