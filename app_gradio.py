# app_gradio.py
import os
import re
import io
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr

import gradio as gr
from openai import OpenAI


# ---------------------------
# Local module guard + sys.path bootstrap
# ---------------------------
# Problem:
# - agent/run_agent.py imports `skill_loader` as a TOP-LEVEL module:
#     from skill_loader import ...
#   so we MUST have ./agent on sys.path (so skill_loader.py resolves).
#
# - We also want to avoid colliding with external pip packages named `agent`,
#   so we pin imports to this repo's local ./agent directory.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_AGENT_DIR = os.path.join(REPO_ROOT, "agent")

if not os.path.isdir(LOCAL_AGENT_DIR):
    raise RuntimeError(
        "Local `agent/` directory not found.\n"
        "This app requires the repository layout with ./agent.\n"
        "Please run from the repository root."
    )

# 1) Ensure ./agent is first so `import skill_loader` inside run_agent.py works
if LOCAL_AGENT_DIR not in sys.path:
    sys.path.insert(0, LOCAL_AGENT_DIR)

# 2) Ensure repo root is also available so `import agent.*` works as expected
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------
# Imports from local agent module
# ---------------------------
from agent.run_agent import (  # noqa: E402
    run_single_case,
    _load_agent_prompts,
    SKILLS_ROOT,
    DEFAULT_BASE_URL,
    DEFAULT_API_KEY,
)
from agent.skill_loader import load_all_skills  # noqa: E402


@dataclass
class AgentRuntime:
    client: OpenAI
    tools: List[Dict[str, Any]]
    runtime: Dict[str, Any]
    instructions: str


_AGENT: Optional[AgentRuntime] = None


def get_agent() -> AgentRuntime:
    """
    Initialize once and reuse:
      - load skills/tools
      - build instructions (skills hint + prompt files)
      - create OpenAI client
    """
    global _AGENT
    if _AGENT is not None:
        return _AGENT

    tools, runtime = load_all_skills(SKILLS_ROOT)

    # Build metadata hint (level-1 disclosure)
    skills_meta = runtime.get("skills_meta", [])
    skills_hint = "\n".join([f"- {m['name']}: {m['description']}" for m in skills_meta])

    # Compose agent instructions from prompt files
    instructions = _load_agent_prompts(skills_hint)

    # Build client
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY),
    )

    _AGENT = AgentRuntime(client=client, tools=tools, runtime=runtime, instructions=instructions)
    return _AGENT


def extract_image_path(text: str) -> Optional[str]:
    """
    Best-effort extraction of a local image path from user instructions.
    Supports:
      - image_path='./x.png' or "./x.jpg"
      - ./x.png
      - x.png (relative)  [optional]
    """
    if not text:
        return None

    # 1) image_path='...'
    m = re.search(
        r"image_path\s*=\s*['\"]([^'\"]+\.(?:png|jpg|jpeg|webp))['\"]",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)

    # 2) ./xxx.(png|jpg|jpeg|webp)
    m = re.search(r"(\./[^\s'\"<>]+\.(?:png|jpg|jpeg|webp))", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # 3) fallback: bare filename like sample.png (avoid catching URLs)
    m = re.search(r"(?<!/)(?<!\w)([^\s'\"<>]+\.(?:png|jpg|jpeg|webp))", text, re.IGNORECASE)
    if m:
        return m.group(1)

    return None


def _resolve_path(p: str) -> str:
    """Resolve to absolute path relative to repo root if not already absolute."""
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(REPO_ROOT, p))


def respond(
    message: str,
    history: List[Tuple[str, str]],
    progress: bool,
    debug: bool,
    trace: bool,
    max_log_chars: int,
    trace_file: str,
) -> Tuple[List[Tuple[str, str]], str, Optional[str], str]:
    """
    Run one agent request and append to chat history.

    Returns:
      - updated history
      - cleared textbox content
      - image filepath (or None)
      - captured logs (stdout+stderr)
    """
    if not message or not message.strip():
        return history, "", None, ""

    agent = get_agent()
    user_text = message.strip()

    # Image (auto-detect from user instruction)
    img_path = extract_image_path(user_text)
    image_for_ui: Optional[str] = None
    if img_path:
        abs_img = _resolve_path(img_path)
        if os.path.exists(abs_img):
            image_for_ui = abs_img

    # Trace file (optional JSONL)
    trace_fp = None
    trace_file = (trace_file or "").strip()
    if trace_file:
        trace_file_abs = _resolve_path(trace_file)
        parent = os.path.dirname(trace_file_abs)
        if parent:
            os.makedirs(parent, exist_ok=True)
        trace_fp = open(trace_file_abs, "a", encoding="utf-8")

    # Capture terminal-style logs from stdout/stderr
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            out_text = run_single_case(
                client=agent.client,
                tools=agent.tools,
                runtime=agent.runtime,
                instructions=agent.instructions,
                user_text=user_text,
                progress=bool(progress),
                debug=bool(debug),
                trace=bool(trace),
                max_log_chars=int(max_log_chars),
                trace_fp=trace_fp,
                case_id="gradio_chat",
            )
    finally:
        if trace_fp:
            trace_fp.close()

    logs = (stdout_buf.getvalue() or "") + (stderr_buf.getvalue() or "")
    out_text = (out_text or "").strip()
    history = history + [(user_text, out_text)]

    # Clear textbox after send
    return history, "", image_for_ui, logs


def reset_all() -> Tuple[List[Tuple[str, str]], Optional[str], str]:
    """Reset chat + image + logs."""
    return [], None, ""


EXAMPLES = [
    "extrac table on image ./sample_table.png. Use backend='tsr'. Return HTML only.",
    "Run TSR on image ./sample_table.png. Use backend='tsr'. Return HTML only.",
    "Extract OTSL from ./sample_table.png using backend='tsr'. Return OTSL only.",
    "Extract table from ./sample_table.png using backend='tsr'. Convert to HTML and save to outputs/sample_table.html.",
    "Call tsr_extract with image_path='./sample_table.png', backend='tsr', output_format='html'. Then call save_html to write outputs/sample_table.html.",
    "Extract table from ./sample_table.png using backend='perception'. Convert to HTML and save to outputs/sample_table.html.",
    "Extract table from ./sample_table.png using backend='tsr'. Convert to HTML and save to outputs/sample_table.html.",
]

CUSTOM_CSS = """
/* Terminal-like logs */
.terminal textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    background: #0d1117;
    color: #d1d5db;
}
.gradio-container { max-width: 2000px !important; }
"""

with gr.Blocks(title="Agent Skills - TSR", fill_height=True, css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
    gr.Markdown("""# Agent Skills - TSR """)

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="Chat", height=520, show_copy_button=True)

            msg = gr.Textbox(
                label="Message",
                placeholder="Type an instruction like: Extract table from ./sample_table.png ...",
                lines=3,
            )
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Reset")

            gr.Examples(
                examples=EXAMPLES,
                inputs=msg,
                label="Examples (click to insert)",
            )

        with gr.Column(scale=3):
            gr.Markdown("### Input Image (auto-detected)")
            image_view = gr.Image(label="Image", type="filepath", height=240)

            gr.Markdown("### Run Options")
            progress = gr.Checkbox(value=True, label="progress logs")
            debug = gr.Checkbox(value=False, label="debug (tool args/outputs)")
            trace = gr.Checkbox(value=False, label="trace (model output items)")
            max_log_chars = gr.Slider(
                minimum=200,
                maximum=5000,
                step=100,
                value=800,
                label="max_log_chars",
            )
            trace_file = gr.Textbox(
                label="trace_file (optional JSONL path)",
                placeholder="e.g., outputs/trace.jsonl",
                value="",
            )
            gr.Markdown(
                """
**Tip:** Match CLI flags:
- `--debug` → **debug**
- `--trace` → **trace**
- `--progress` → **progress logs**
"""
            )

    logs = gr.Textbox(
        label="Agent Logs (terminal)",
        lines=14,
        interactive=False,
        elem_classes=["terminal"],
        placeholder="Logs will appear here after you press Send...",
    )

    send.click(
        fn=respond,
        inputs=[msg, chatbot, progress, debug, trace, max_log_chars, trace_file],
        outputs=[chatbot, msg, image_view, logs],
    )
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, progress, debug, trace, max_log_chars, trace_file],
        outputs=[chatbot, msg, image_view, logs],
    )
    clear.click(
        fn=reset_all,
        inputs=None,
        outputs=[chatbot, image_view, logs],
    )

if __name__ == "__main__":
    # queue=True helps with concurrency; you can remove if not needed
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
