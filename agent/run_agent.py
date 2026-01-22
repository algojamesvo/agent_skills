# agent/run_agent.py
import os
import json
import re
import argparse
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
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


# ---------------------------
# Utilities
# ---------------------------

def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _truncate(s: Any, n: int) -> str:
    if s is None:
        return ""
    try:
        s = json.dumps(s, ensure_ascii=False)
    except Exception:
        s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"... [truncated {len(s) - n} chars]"


def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def _log(enabled: bool, msg: str):
    if enabled:
        print(f"{_now_ts()} {msg}")


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


def _stable_json(obj: Any) -> str:
    """Stable JSON string for repeat-detection signatures."""
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(obj)


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
    return "\n\n".join([profile, goals, inst]).strip()


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


def _open_trace_file(path: Optional[str]):
    if not path:
        return None
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def _trace_write(fp, event: Dict[str, Any]):
    if not fp:
        return
    fp.write(json.dumps(event, ensure_ascii=False) + "\n")
    fp.flush()


def _parse_args_obj(args: Any) -> Dict[str, Any]:
    if isinstance(args, str):
        return json.loads(args) if args else {}
    if isinstance(args, dict):
        return args
    return {}


# ---------------------------
# Core runner (single case)
# ---------------------------

def run_single_case(
    *,
    client: OpenAI,
    tools: List[Dict[str, Any]],
    runtime: Dict[str, Any],
    instructions: str,
    user_text: str,
    progress: bool,
    debug: bool,
    trace: bool,
    max_log_chars: int,
    max_loops: int,
    max_tool_retries: int,
    repeat_limit: int,
    trace_fp=None,
    case_id: str = "case",
) -> str:
    """
    Run one agent loop for a single user_text.
    Adds:
      - progress/debug/trace logging
      - optional JSONL trace file
      - safety guards: max_loops, repeat_limit
      - tool retry policy: exceptions or output dict containing {"error": ...}
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

    loop_idx = 0
    end_reason = "unknown"

    # Repeat detection: signature = f"{tool}:{stable_json(args_obj)}" -> count
    repeat_counts: Dict[str, int] = {}

    _log(progress, f"[{case_id}] START")
    _trace_write(trace_fp, {
        "type": "case_start",
        "case_id": case_id,
        "model": MODEL,
        "base_url": DEFAULT_BASE_URL,
        "user_text": user_text,
        "max_loops": max_loops,
        "max_tool_retries": max_tool_retries,
        "repeat_limit": repeat_limit,
    })

    resp = None

    while True:
        loop_idx += 1

        if loop_idx > max_loops:
            end_reason = "max_loops"
            _log(True, f"[{case_id}] [loop {loop_idx}] reached max_loops={max_loops} -> stopping")
            _trace_write(trace_fp, {
                "type": "loop_stop",
                "case_id": case_id,
                "loop": loop_idx,
                "end_reason": end_reason,
            })
            # Give a small in-context signal to finish best-effort
            input_list.append({
                "role": "system",
                "content": f"[Agent stopped] Reached max_loops={max_loops}. Provide best-effort final answer with current info.",
            })
            # One last model call to produce final answer (optional)
            resp = client.responses.create(
                model=MODEL,
                instructions=instructions,
                tools=tools,
                input=input_list,
            )
            break

        _log(progress, f"[{case_id}] [loop {loop_idx}] calling model...")
        _trace_write(trace_fp, {"type": "loop_start", "case_id": case_id, "loop": loop_idx})

        resp = client.responses.create(
            model=MODEL,
            instructions=instructions,
            tools=tools,
            input=input_list,
        )

        # Keep model output items in context for next turn
        append_response_output_items(input_list, resp)

        out_items = getattr(resp, "output", []) or []

        # Trace: show output item snapshots
        if trace:
            for it in out_items:
                it_type = _get_item_type(it)
                if isinstance(it, dict):
                    snapshot = {
                        "type": it_type,
                        "name": it.get("name"),
                        "call_id": it.get("call_id"),
                        "arguments": it.get("arguments"),
                        "content": it.get("content"),
                        "text": it.get("text"),
                    }
                else:
                    snapshot = {
                        "type": it_type,
                        "name": getattr(it, "name", None),
                        "call_id": getattr(it, "call_id", None),
                        "arguments": getattr(it, "arguments", None),
                        "content": getattr(it, "content", None),
                        "text": getattr(it, "text", None),
                    }
                _log(True, f"[{case_id}] [trace] item={_truncate(snapshot, max_log_chars)}")

        _trace_write(trace_fp, {
            "type": "model_output",
            "case_id": case_id,
            "loop": loop_idx,
            "num_items": len(out_items),
            "item_types": [(_get_item_type(it) or "unknown") for it in out_items],
        })

        tool_calls = [it for it in out_items if _get_item_type(it) == "function_call"]
        _log(progress, f"[{case_id}] [loop {loop_idx}] tool_calls={len(tool_calls)}")

        if not tool_calls:
            end_reason = "no_tool_calls"
            _log(progress, f"[{case_id}] [loop {loop_idx}] no tool calls -> finishing")
            _trace_write(trace_fp, {
                "type": "loop_end",
                "case_id": case_id,
                "loop": loop_idx,
                "ended": True,
                "end_reason": end_reason,
            })
            break

        # Log tool calls
        for call in tool_calls:
            tname = _get_call_name(call)
            args = _get_call_arguments(call)
            _log(debug, f"[{case_id}] [call] {tname} args={_truncate(args, max_log_chars)}")
            _trace_write(trace_fp, {
                "type": "tool_call",
                "case_id": case_id,
                "loop": loop_idx,
                "name": tname,
                "arguments": args,
                "call_id": _get_call_id(call),
            })

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
                _log(progress, f"[{case_id}] [load] SKILL.md body: {sname}")
                _trace_write(trace_fp, {
                    "type": "load_skill_body",
                    "case_id": case_id,
                    "loop": loop_idx,
                    "skill": sname,
                })

        # 2) References on-demand
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
                    _log(progress, f"[{case_id}] [warn] missing reference: {sname}/{rel_ref}")
                    _trace_write(trace_fp, {
                        "type": "missing_reference",
                        "case_id": case_id,
                        "loop": loop_idx,
                        "skill": sname,
                        "ref": rel_ref,
                        "abs_path": abs_ref,
                    })
                    continue

                ref_text = _safe_read_text(abs_ref).strip()
                if ref_text:
                    input_list.append({
                        "role": "system",
                        "content": f"[Loaded reference for skill '{sname}': {rel_ref}]\n\n{ref_text}",
                    })
                loaded_references.add(key)
                _log(progress, f"[{case_id}] [load] reference: {sname}/{rel_ref}")
                _trace_write(trace_fp, {
                    "type": "load_reference",
                    "case_id": case_id,
                    "loop": loop_idx,
                    "skill": sname,
                    "ref": rel_ref,
                    "abs_path": abs_ref,
                })

        # 3) Execute tool calls
        stopped_by_repeat = False

        for call in tool_calls:
            name = _get_call_name(call)
            args = _get_call_arguments(call)
            call_id = _get_call_id(call)

            if not name or not call_id:
                err = {"error": "Malformed function_call item", "name": name, "call_id": call_id}
                input_list.append({
                    "type": "function_call_output",
                    "call_id": call_id or "missing_call_id",
                    "output": json.dumps(err, ensure_ascii=False),
                })
                _log(progress, f"[{case_id}] [error] malformed function_call item")
                _trace_write(trace_fp, {
                    "type": "tool_output",
                    "case_id": case_id,
                    "loop": loop_idx,
                    "name": name,
                    "call_id": call_id,
                    "output": err,
                })
                continue

            if name not in executors:
                output = {"error": f"Tool not found: {name}"}
                input_list.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output, ensure_ascii=False),
                })
                _log(debug, f"[{case_id}] [out]  {name} -> {_truncate(output, max_log_chars)}")
                _trace_write(trace_fp, {
                    "type": "tool_output",
                    "case_id": case_id,
                    "loop": loop_idx,
                    "name": name,
                    "call_id": call_id,
                    "output": output,
                })
                continue

            args_obj = _parse_args_obj(args)

            # Repeat guard
            sig = f"{name}:{_stable_json(args_obj)}"
            repeat_counts[sig] = repeat_counts.get(sig, 0) + 1
            if repeat_counts[sig] > repeat_limit:
                end_reason = "repeated_calls"
                stopped_by_repeat = True
                _log(True, f"[{case_id}] [stop] repeated_calls: {name} same args count={repeat_counts[sig]} > repeat_limit={repeat_limit}")
                _trace_write(trace_fp, {
                    "type": "loop_stop",
                    "case_id": case_id,
                    "loop": loop_idx,
                    "end_reason": end_reason,
                    "tool": name,
                    "args_obj": args_obj,
                    "count": repeat_counts[sig],
                    "repeat_limit": repeat_limit,
                })
                # Signal model to finish best-effort
                input_list.append({
                    "role": "system",
                    "content": (
                        f"[Agent stopped] Detected repeated tool calls for '{name}' with identical arguments "
                        f"(>{repeat_limit} times). Provide best-effort final answer."
                    ),
                })
                break

            # Retry policy
            output: Any = None
            last_err: Optional[str] = None

            for attempt in range(max_tool_retries + 1):
                try:
                    _log(debug, f"[{case_id}] [exec] {name} attempt={attempt+1}/{max_tool_retries+1} args_obj={_truncate(args_obj, max_log_chars)}")
                    out0 = executors[name](**args_obj)
                    output = out0

                    # tool returned structured error?
                    if isinstance(out0, dict) and out0.get("error"):
                        last_err = str(out0.get("error"))
                        if attempt < max_tool_retries:
                            sleep_s = 0.5 * (2 ** attempt)
                            _log(debug, f"[{case_id}] [retry] {name} tool_error={last_err!r} sleep={sleep_s:.1f}s")
                            time.sleep(sleep_s)
                            continue
                    break
                except Exception as e:
                    last_err = str(e)
                    if attempt < max_tool_retries:
                        sleep_s = 0.5 * (2 ** attempt)
                        _log(debug, f"[{case_id}] [retry] {name} exception={last_err!r} sleep={sleep_s:.1f}s")
                        time.sleep(sleep_s)
                        continue
                    output = {"error": last_err}

            input_list.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(output, ensure_ascii=False),
            })

            _log(debug, f"[{case_id}] [out]  {name} -> {_truncate(output, max_log_chars)}")

            if name == "save_html" and isinstance(output, dict) and output.get("out_path"):
                _log(True, f"[{case_id}] [artifact] saved HTML -> {output.get('out_path')}")

            _trace_write(trace_fp, {
                "type": "tool_output",
                "case_id": case_id,
                "loop": loop_idx,
                "name": name,
                "call_id": call_id,
                "output": output,
            })

        _trace_write(trace_fp, {
            "type": "loop_end",
            "case_id": case_id,
            "loop": loop_idx,
            "ended": False,
            "end_reason": end_reason,
        })

        if stopped_by_repeat:
            # One last model call to produce final answer (optional)
            resp = client.responses.create(
                model=MODEL,
                instructions=instructions,
                tools=tools,
                input=input_list,
            )
            break

    final_text = resp.output_text if resp is not None else ""
    _trace_write(trace_fp, {
        "type": "case_end",
        "case_id": case_id,
        "final_output_text": final_text,
        "loops": loop_idx,
        "end_reason": end_reason,
        "loaded_skills": sorted(list(loaded_skills)),
        "loaded_references": sorted([f"{a}:{b}" for (a, b) in loaded_references]),
    })
    _log(progress, f"[{case_id}] DONE (loops={loop_idx}) end_reason={end_reason}")
    return final_text


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Run OpenAI-style tool-using agent with Claude-skill folders.")
    parser.add_argument("--input", type=str, default=None, help="Single input string to run.")
    parser.add_argument("--input_file", type=str, default=None, help="JSONL file with lines like {'input': '...'}")

    # Observability flags
    parser.add_argument("--progress", action="store_true", help="Print progress logs.")
    parser.add_argument("--debug", action="store_true", help="Print tool args/outputs and loading events.")
    parser.add_argument("--trace", action="store_true", help="Print model output items (may be verbose).")
    parser.add_argument("--trace_file", type=str, default=None, help="Write JSONL trace logs to this file.")
    parser.add_argument("--max_log_chars", type=int, default=800, help="Truncate long logs.")

    # Safety / stability
    parser.add_argument("--max_loops", type=int, default=int(os.getenv("AGENT_MAX_LOOPS", "8")), help="Max agent loops per case.")
    parser.add_argument("--max_tool_retries", type=int, default=int(os.getenv("AGENT_MAX_TOOL_RETRIES", "1")), help="Retries per tool call (error/exception).")
    parser.add_argument("--repeat_limit", type=int, default=int(os.getenv("AGENT_REPEAT_LIMIT", "3")), help="Stop if same tool+args repeats more than this.")

    args = parser.parse_args()

    # Default behavior: show progress if debug/trace is enabled
    progress = bool(args.progress or args.debug or args.trace)
    debug = bool(args.debug)
    trace = bool(args.trace)
    max_log_chars = int(args.max_log_chars)

    max_loops = int(args.max_loops)
    max_tool_retries = int(args.max_tool_retries)
    repeat_limit = int(args.repeat_limit)

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
        cases = [
            {"id": "demo_string_utils", "input": "Slugify this title and also count words: 'TSR Agent: Input → Agent → Structured Tables'"},
        ]

    trace_fp = _open_trace_file(args.trace_file)

    try:
        for idx, case in enumerate(cases, start=1):
            user_text = case["input"]
            case_id = case.get("id") or case.get("title") or f"case_{idx}"

            print("=" * 80)
            print(f"[{idx}/{len(cases)}] {case_id}")
            print("- Input:")
            print(user_text)

            out = run_single_case(
                client=client,
                tools=tools,
                runtime=runtime,
                instructions=instructions,
                user_text=user_text,
                progress=progress,
                debug=debug,
                trace=trace,
                max_log_chars=max_log_chars,
                max_loops=max_loops,
                max_tool_retries=max_tool_retries,
                repeat_limit=repeat_limit,
                trace_fp=trace_fp,
                case_id=case_id,
            )

            print("- Output:")
            print(out.strip() if out else out)
    finally:
        if trace_fp:
            trace_fp.close()


if __name__ == "__main__":
    main()
