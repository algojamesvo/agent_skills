---
name: tsr
description: Table Structure Recognition (TSR) skill to extract structured tables from table images. Use when the user provides a table image and asks to convert it into OTSL and/or HTML, validate table structure, or troubleshoot TSR output. Supports two backends: (1) dedicated TSR VLM at vllm-vlm-tsr-1 (model agilesoda/vlm-tsr) and (2) perception VLM at vllm-vlm-perception-1 (model nanonets/Nanonets-OCR2-3B) which may return HTML directly or OTSL-like tags that need conversion.
---

# TSR Skill

## Workflow

### 1) Extract structured table from an image
- When the user provides a table image and asks for structured output, call:
  - `tsr_extract(image_path, backend, output_format)`
- **backend selection**
  - Use `backend="tsr"` for the dedicated TSR endpoint (most reliable for structure).
  - Use `backend="perception"` for the perception endpoint (may return HTML directly).
- **output_format**
  - Use `output_format="html"` when the user wants HTML output.
  - Use `output_format="otsl"` when the user wants OTSL output.
  - Use `output_format="auto"` to return both when possible (recommended default).

### 2) Detect whether the model output is OTSL or HTML
- If the model output contains OTSL tags like `FCEL`, `ECEL`, or `NL`, treat it as OTSL and convert it.
- Otherwise, treat it as HTML (some perception models may return HTML directly).

### 3) Convert OTSL → HTML
- If you have OTSL text and the user wants HTML, call:
  - `otsl_to_html(otsl_text)`
- If conversion fails, return:
  - the raw OTSL
  - the error message
  - a short suggested fix (e.g., retry with `backend="tsr"` or re-run with stricter prompt from references).

### 4) Save HTML output to a file (optional)
- If the user wants an output file, call:
  - `save_html(html, out_path)`
- Return the file path and a short confirmation.

---

## Output Contract (what tools should return)

Prefer returning stable JSON objects:

- `tsr_extract(...)` should return:
  - `raw_text`: raw model output
  - `detected_format`: `"otsl"` or `"html"`
  - `otsl`: OTSL string when available (else empty)
  - `html`: HTML string when available (else empty)
  - `backend_used`: `"tsr"` or `"perception"`
  - `model`: model name used

- `otsl_to_html(...)` should return:
  - `html`

- `save_html(...)` should return:
  - `out_path`

---

## Prompting Rules

- Use deterministic settings:
  - `temperature=0`
  - keep `max_tokens` high enough for large tables (e.g., 8192+)
- Use a TSR-focused system prompt that requests OTSL tags and structured output.

See:
- `references/prompts.md` for recommended system prompts and examples
- `references/otsl_format.md` for OTSL tags and formatting rules

---

## Debugging & Recovery

If output looks wrong (missing structure, merged cells, broken HTML):

1) Retry with `backend="tsr"` if using perception.
2) Increase `max_tokens` if output is truncated.
3) If HTML is malformed, convert from OTSL again and validate.
4) If model returns non-table content, confirm the input image is a table crop (not the full page).

---

## Notes

- Keep SKILL.md concise. Put long prompt templates and tag specifications into references files.
- Prefer tool calls (`tsr_extract`, `otsl_to_html`, `save_html`) over manual parsing for reliability.
