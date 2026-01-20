# OTSL Format Specification

This file describes the OTSL (Open Table Structure Language) format used for TSR outputs.

OTSL is a lightweight, tag-based representation of tables designed to be easy to parse and convert into HTML.

---

## Core Tags

The most common tags used in OTSL:

| Tag  | Meaning |
|------|--------|
| FCEL | Start of a table cell (First Cell) |
| ECEL | End of a table cell |
| NL   | New line (row separator) |

---

## Basic Structure

An OTSL table is represented as a sequence of cells separated by NL markers.

Example:

```
FCEL A1 ECEL FCEL B1 ECEL NL
FCEL A2 ECEL FCEL B2 ECEL
```

This corresponds to the table:

| A1 | B1 |
|----|----|
| A2 | B2 |

---

## Merged Cells

Merged cells are represented using attributes within FCEL tags.

Typical attributes:

- rowspan
- colspan

Example:

```
FCEL rowspan=2 colspan=2 Header ECEL NL
FCEL Cell1 ECEL FCEL Cell2 ECEL
```

---

## Text Rules

- Text inside cells should be kept **exactly as shown in the image**
- Do not modify punctuation or spacing
- Do not translate or summarize
- Multi-line cell text should be preserved

---

## Conversion Expectations

When converting OTSL to HTML:

- Each FCEL/ECEL pair becomes `<td>...</td>`
- NL becomes `</tr><tr>`
- rowspan/colspan map directly to HTML attributes

Example conversion:

OTSL:

```
FCEL A ECEL FCEL B ECEL NL
FCEL C ECEL FCEL D ECEL
```

HTML:

```html
<table>
<tr><td>A</td><td>B</td></tr>
<tr><td>C</td><td>D</td></tr>
</table>
```

---

## Error Patterns

Common issues to watch for:

- Missing ECEL tags
- Extra NL at end of output
- Unbalanced rowspan/colspan
- Truncated output due to max_tokens

If any of these occur:

- Retry with higher max_tokens
- Use backend="tsr" instead of perception
- Re-run conversion using `otsl_to_html`

---

## Detection Heuristics

To determine whether text is OTSL:

Check for presence of any of:

- `FCEL`
- `ECEL`
- `NL`

If none are present, assume the output is HTML or plain text.

---

## Best Practices

- Always request **OTSL format** from TSR models
- Use deterministic settings (temperature=0)
- Validate structure before saving final HTML

---

End of OTSL format reference.
