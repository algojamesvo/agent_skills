---
name: string-utils
description: Utilities for text normalization and counting. Use when user asks to slugify text, normalize whitespace, or count words/characters.
---

# String Utils Skill

## Workflow
- If user asks to "slugify" or "make URL-friendly": call tool `slugify`.
- If user asks to "count words/chars": call tool `count_words`.
- Follow formatting rules in references if needed: `references/rules.md`.

## Notes
- Prefer tool calls for deterministic outputs.
