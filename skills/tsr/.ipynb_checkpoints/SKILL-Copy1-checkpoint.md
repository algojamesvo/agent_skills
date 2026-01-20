---
name: tsr
description: Table Structure Recognition (TSR) from table images. Use when user provides a table image and asks to extract table structure as OTSL/HTML, convert OTSL to HTML, or validate TSR output. Supports vLLM endpoints (vlm-tsr, vlm-perception).
---

# TSR Skill

## Workflow
1) If user gives a table image and wants structured output:
   - Call `tsr_extract(image_path, backend=..., output_format=...)`
2) If output looks like OTSL, convert to HTML:
   - Call `otsl_to_html(otsl)`
3) If user wants HTML file output:
   - Call `save_html(html, out_path)`

## Backend selection
- Use `backend="tsr"` for dedicated TSR model endpoint.
- Use `backend="perception"` for perception OCR model; output may already be HTML, auto-detect.

## References
- Prompt standards and examples: references/prompts.md
- OTSL tags and rules: references/otsl_format.md
