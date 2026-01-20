# TSR Prompt Templates

This file contains recommended system and user prompts for Table Structure Recognition (TSR).

These prompts are designed to produce stable OTSL outputs compatible with the conversion utilities.

---

## Recommended System Prompt (OTSL-focused)

Use this system prompt for dedicated TSR models:

```
You are an AI specialized in recognizing and extracting table from images. 
Your mission is to analyze the table image and generate the result in OTSL format 
using specified tags. Output only the results without any other words and explanation.
```

Guidelines:

- Always request **OTSL format only**
- No preamble, no explanation
- Deterministic settings: `temperature=0`

---

## User Prompt Template

Minimal, reliable user prompt:

```
Extract table from this image.
```

---

## Alternative Detailed User Prompt

If tables are complex (merged cells, multi-line headers):

```
Extract the table structure from the given image using OTSL format.

Requirements:
- Preserve row and column relationships
- Respect merged cells
- Keep cell text exactly as shown
- Do not summarize or paraphrase
- Output OTSL only
```

---

## Prompt for Perception Backend

When using the perception model, which may return HTML directly:

```
Analyze the table in this image and return the table structure.

If possible, return OTSL format.
If you cannot produce OTSL, return clean HTML table markup only.
```

---

## Parameters to Use

Recommended generation parameters:

| Parameter     | Value |
|----------------|-------|
| temperature    | 0.0   |
| max_tokens     | 8192+ |
| top_p          | 1.0   |
| frequency_penalty | 0 |
| presence_penalty  | 0 |

---

## Troubleshooting Prompts

If output is truncated:

```
Extract the table in OTSL format. The table is large. Continue until complete.
```

If structure is wrong:

```
Extract the table with strict attention to row/column alignment and merged cells.
Use OTSL tags precisely.
```

---

## Validation Tips

- If the output contains tags like `FCEL`, `ECEL`, or `NL`, treat it as OTSL.
- Otherwise assume the model returned HTML directly.
- For malformed HTML, re-run extraction with `backend="tsr"`.

---

End of prompts reference.
