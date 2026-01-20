# TSR Examples (Image → OTSL → HTML)

This file provides **illustrative examples** of how a table image can be represented in **OTSL** and then converted into **HTML**.

> Notes
> - These are **synthetic examples** (not extracted from a real image) to demonstrate formatting.
> - If your perception backend returns HTML directly, you can skip OTSL conversion.
> - OTSL detection heuristic: if the output contains any of `FCEL`, `ECEL`, `NL`, treat it as OTSL.

---

## Example 1: Simple 2×2 table

### Image (concept)
A small table with 2 columns and 2 rows:

| Product | Price |
|--------|-------|
| Apple  | 10    |
| Banana | 12    |

### OTSL
```
FCEL Product ECEL FCEL Price ECEL NL
FCEL Apple ECEL FCEL 10 ECEL NL
FCEL Banana ECEL FCEL 12 ECEL
```

### HTML (expected)
```html
<table>
  <tr><td>Product</td><td>Price</td></tr>
  <tr><td>Apple</td><td>10</td></tr>
  <tr><td>Banana</td><td>12</td></tr>
</table>
```

---

## Example 2: Header with colspan (merged cell)

### Image (concept)
Header spans 2 columns:

| 2026 Sales |
| Q1 | Q2 |
| 10 | 12 |

### OTSL (with colspan)
```
FCEL colspan=2 2026 Sales ECEL NL
FCEL Q1 ECEL FCEL Q2 ECEL NL
FCEL 10 ECEL FCEL 12 ECEL
```

### HTML (expected)
```html
<table>
  <tr><td colspan="2">2026 Sales</td></tr>
  <tr><td>Q1</td><td>Q2</td></tr>
  <tr><td>10</td><td>12</td></tr>
</table>
```

---

## Example 3: Row header with rowspan

### Image (concept)
A row header spans 2 rows:

| Region | Revenue |
| APAC   | 100     |
|        | 120     |

Interpretation: “APAC” is a row header that applies to two revenue rows.

### OTSL (with rowspan)
```
FCEL Region ECEL FCEL Revenue ECEL NL
FCEL rowspan=2 APAC ECEL FCEL 100 ECEL NL
FCEL 120 ECEL
```

### HTML (expected)
```html
<table>
  <tr><td>Region</td><td>Revenue</td></tr>
  <tr><td rowspan="2">APAC</td><td>100</td></tr>
  <tr><td>120</td></tr>
</table>
```

---

## Example 4: Multi-line cell text

### Image (concept)
A cell contains line breaks (e.g., address):

| Name | Address |
| John | 12 Main St\nBangkok |
| Ana  | 99 Lake Rd\nHanoi   |

### OTSL (newline preserved as literal \n in text)
```
FCEL Name ECEL FCEL Address ECEL NL
FCEL John ECEL FCEL 12 Main St\nBangkok ECEL NL
FCEL Ana ECEL FCEL 99 Lake Rd\nHanoi ECEL
```

### HTML (expected)
```html
<table>
  <tr><td>Name</td><td>Address</td></tr>
  <tr><td>John</td><td>12 Main St\nBangkok</td></tr>
  <tr><td>Ana</td><td>99 Lake Rd\nHanoi</td></tr>
</table>
```

> Depending on your converter, you may want to post-process `\n` into `<br/>`.

---

## Example 5: Perception backend returns HTML directly

### Image (concept)
Some OCR/perception models output HTML table markup without OTSL tags.

### Raw output (HTML)
```html
<table>
  <tr><th>Item</th><th>Qty</th></tr>
  <tr><td>Pen</td><td>3</td></tr>
</table>
```

### Handling
- `detected_format="html"`
- No conversion needed; you can save as-is using `save_html(html, out_path)`

---

## Quick Checklist (for debugging)

- If OTSL conversion fails:
  - Re-run extraction with `backend="tsr"`
  - Increase `max_tokens`
  - Confirm the image is a tight crop of the table region
- If HTML is malformed:
  - Prefer converting from OTSL again (if available)
  - Validate that `rowspan/colspan` attributes are balanced

---

End of examples.
