import re
import itertools
import html
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, Union
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

class TableCell(BaseModel):
    """TableCell."""
    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False

    @model_validator(mode="before")
    @classmethod
    def from_dict_format(cls, data: Any) -> Any:
        """from_dict_format."""
        if isinstance(data, Dict):
            # Check if this is a native BoundingBox or a bbox from docling-ibm-models
            if (
                # "bbox" not in data
                # or data["bbox"] is None
                # or isinstance(data["bbox"], BoundingBox)
                "text"
                in data
            ):
                return data
            text = data["bbox"].get("token", "")
            if not len(text):
                text_cells = data.pop("text_cell_bboxes", None)
                if text_cells:
                    for el in text_cells:
                        text += el["token"] + " "

                text = text.strip()
            data["text"] = text

        return data


class TableData(BaseModel):  # TBD
    """BaseTableData."""

    table_cells: List[TableCell] = []
    num_rows: int = 0
    num_cols: int = 0

    @computed_field  # type: ignore
    @property
    def grid(
        self,
    ) -> List[List[TableCell]]:
        """grid."""
        # Initialise empty table data grid (only empty cells)
        table_data = [
            [
                TableCell(
                    text="",
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]

        # Overwrite cells in table data for which there is actual cell content.
        for cell in self.table_cells:
            for i in range(
                min(cell.start_row_offset_idx, self.num_rows),
                min(cell.end_row_offset_idx, self.num_rows),
            ):
                for j in range(
                    min(cell.start_col_offset_idx, self.num_cols),
                    min(cell.end_col_offset_idx, self.num_cols),
                ):
                    table_data[i][j] = cell

        return table_data

"""
OTSL
"""
OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"

OTSL_NTABLE_START = "<ntable>"
OTSL_NTABLE_END = "</ntable>"

def otsl_extract_tokens_and_text(s: str):
    # First, protect <ntable>...</ntable> blocks with placeholders
    # to prevent them from being tokenized as regular text
    NTABLE_PLACEHOLDER_PREFIX = "__NTABLE_PLACEHOLDER_"
    ntable_placeholders = {}
    placeholder_counter = 0
    
    # Find and replace <ntable>...</ntable> blocks with placeholders
    # Use a function to handle replacement properly
    def replace_ntable(match):
        nonlocal placeholder_counter
        placeholder = f"{NTABLE_PLACEHOLDER_PREFIX}{placeholder_counter}__"
        ntable_placeholders[placeholder] = match.group(0)  # Store the full <ntable>...</ntable> block
        placeholder_counter += 1
        return placeholder
    
    # Pattern to match <ntable>...</ntable> blocks (non-greedy, matches each block separately)
    ntable_pattern = r'<ntable>(.*?)</ntable>'
    # Replace all occurrences - re.sub handles all matches automatically
    protected_s = re.sub(ntable_pattern, replace_ntable, s, flags=re.DOTALL)
    
    # Pattern to match anything enclosed by < >
    # (including the angle brackets themselves)
    # pattern = r"(<[^>]+>)"
    pattern = r"(" + r"|".join([OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]) + r")"
    # Find all tokens (e.g. "<otsl>", "<loc_140>", etc.)
    tokens = re.findall(pattern, protected_s)
    # Remove any tokens that start with "<loc_"
    tokens = [token for token in tokens]
    # Split the string by those tokens to get the in-between text
    text_parts = re.split(pattern, protected_s)
    text_parts = [token for token in text_parts]
    # Remove any empty or purely whitespace strings from text_parts
    text_parts = [part for part in text_parts if part.strip()]
    
    # Restore <ntable> placeholders in text_parts BEFORE returning
    # Need to restore ALL placeholders in each text part (in reverse order to preserve indices)
    for i, part in enumerate(text_parts):
        # Restore all placeholders in this part (in reverse order to avoid index issues)
        restored_part = part
        for placeholder in sorted(ntable_placeholders.keys(), reverse=True):
            if placeholder in restored_part:
                restored_part = restored_part.replace(placeholder, ntable_placeholders[placeholder])
        text_parts[i] = restored_part

    return tokens, text_parts

def otsl_parse_texts(texts, tokens):
    split_word = OTSL_NL
    split_row_tokens = [
        list(y)
        for x, y in itertools.groupby(tokens, lambda z: z == split_word)
        if not x
    ]
    table_cells = []
    r_idx = 0
    c_idx = 0

    # 检查并补充矩阵以使其完整
    if split_row_tokens:
        # 找到最大列数
        max_cols = max(len(row) for row in split_row_tokens)
        
        # 补充每一行使其达到最大列数
        for row_idx, row in enumerate(split_row_tokens):
            while len(row) < max_cols:
                row.append(OTSL_ECEL)
        
        # 在texts中也需要相应补充<ecel>
        # 重新构建texts以包含补充的<ecel>
        new_texts = []
        text_idx = 0
        
        for row_idx, row in enumerate(split_row_tokens):
            for col_idx, token in enumerate(row):
                new_texts.append(token)
                # 如果这个token在原始texts中有对应的文本内容，添加它
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    # 检查下一个是否是文本内容（不是token）
                    if (text_idx < len(texts) and 
                        texts[text_idx] not in [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]):
                        new_texts.append(texts[text_idx])
                        text_idx += 1

            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1
        
        texts = new_texts

    def count_right(tokens, c_idx, r_idx, which_tokens):
        span = 0
        c_idx_iter = c_idx
        while tokens[r_idx][c_idx_iter] in which_tokens:
            c_idx_iter += 1
            span += 1
            if c_idx_iter >= len(tokens[r_idx]):
                return span
        return span

    def count_down(tokens, c_idx, r_idx, which_tokens):
        span = 0
        r_idx_iter = r_idx
        while tokens[r_idx_iter][c_idx] in which_tokens:
            r_idx_iter += 1
            span += 1
            if r_idx_iter >= len(tokens):
                return span
        return span

    for i, text in enumerate(texts):
        cell_text = ""
        if text in [
            OTSL_FCEL,
            OTSL_ECEL,
        ]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != OTSL_ECEL and (texts[i + 1] not in [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]):
                cell_text = texts[i + 1]
                right_offset = 2

            # Check next element(s) for lcel / ucel / xcel,
            # set properly row_span, col_span
            next_right_cell = ""
            if i + right_offset < len(texts):
                next_right_cell = texts[i + right_offset]

            next_bottom_cell = ""
            if r_idx + 1 < len(split_row_tokens):
                if c_idx < len(split_row_tokens[r_idx + 1]):
                    next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]

            if next_right_cell in [
                OTSL_LCEL,
                OTSL_XCEL,
            ]:
                # we have horisontal spanning cell or 2d spanning cell
                col_span += count_right(
                    split_row_tokens,
                    c_idx + 1,
                    r_idx,
                    [OTSL_LCEL, OTSL_XCEL],
                )
            if next_bottom_cell in [
                OTSL_UCEL,
                OTSL_XCEL,
            ]:
                # we have a vertical spanning cell or 2d spanning cell
                row_span += count_down(
                    split_row_tokens,
                    c_idx,
                    r_idx + 1,
                    [OTSL_UCEL, OTSL_XCEL],
                )

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                )
            )
        if text in [
            OTSL_FCEL,
            OTSL_ECEL,
            OTSL_LCEL,
            OTSL_UCEL,
            OTSL_XCEL,
        ]:
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0
    return table_cells, split_row_tokens


def process_nested_tables_in_text(cell_text: str) -> str:
    """
    Process cell text and convert any <ntable>...</ntable> blocks to HTML tables.
    Also handles placeholders that may have been missed during restoration.
    Returns the processed text with nested tables converted to HTML.
    """
    if not cell_text:
        return cell_text
    
    # First, check if there are any placeholders that need to be restored
    # Pattern to match placeholders like __NTABLE_PLACEHOLDER_0__
    placeholder_pattern = r'__NTABLE_PLACEHOLDER_(\d+)__'
    placeholder_matches = list(re.finditer(placeholder_pattern, cell_text))
    
    # If placeholders found, we need to extract <ntable> blocks from the original OTSL
    # But we don't have access to the original here, so we'll skip placeholder handling
    # and just process <ntable> tags directly
    
    # Pattern to match <ntable>...</ntable> blocks
    ntable_pattern = r'<ntable>(.*?)</ntable>'
    
    def convert_ntable(match):
        nested_otsl = match.group(1)
        # Recursively convert nested table OTSL to HTML
        nested_html = convert_single_table_otsl_to_html(nested_otsl)
        if nested_html:
            # Return the full table HTML
            return nested_html
        return ''
    
    # Replace all <ntable>...</ntable> blocks with their HTML equivalents
    processed_text = re.sub(ntable_pattern, convert_ntable, cell_text, flags=re.DOTALL)
    
    # If there are still placeholders, we can't recover them here
    # This shouldn't happen if otsl_extract_tokens_and_text worked correctly
    # But if it does, at least escape the placeholder so it doesn't break HTML
    if '__NTABLE_PLACEHOLDER_' in processed_text:
        # Replace remaining placeholders with empty string (they should have been <ntable> tags)
        processed_text = re.sub(r'__NTABLE_PLACEHOLDER_\d+__', '', processed_text)
    
    return processed_text


def export_to_html(table_data: TableData) -> str:
    nrows = table_data.num_rows
    ncols = table_data.num_cols
    # print(nrows, ncols)

    if not table_data.table_cells:
        return ""

    current_grid = table_data.grid

    html_str_list = []

    for i in range(nrows):
        html_str_list.append("<tr>")
        for j in range(ncols):
            cell: TableCell = current_grid[i][j]

            if cell.start_row_offset_idx != i or cell.start_col_offset_idx != j:
                continue

            # Process cell text (includes handling nested tables)
            cell_text = cell.text.strip()
            
            # First, process nested tables (<ntable> tags) - convert them to HTML
            cell_text = process_nested_tables_in_text(cell_text)
            
            # Preserve <br> tags before HTML escaping
            # Use a placeholder that won't appear in normal text
            BR_PLACEHOLDER = "__BR_TAG_PLACEHOLDER__"
            # Replace <br> tags (both <br> and <br/>) with placeholder
            cell_text = re.sub(r'<br\s*/?>', BR_PLACEHOLDER, cell_text, flags=re.IGNORECASE)
            # HTML escape the content to preserve entities like &lt; < > &amp; etc.
            # But don't escape already-converted HTML tables (they should be valid HTML)
            # Split by potential table boundaries and escape only non-HTML parts
            content_parts = re.split(r'(<table[^>]*>.*?</table>)', cell_text, flags=re.DOTALL | re.IGNORECASE)
            escaped_parts = []
            for part in content_parts:
                if part.strip().startswith('<table') and part.strip().endswith('</table>'):
                    # This is already HTML table, don't escape
                    escaped_parts.append(part)
                else:
                    # Regular text, escape it
                    escaped_parts.append(html.escape(part))
            content = ''.join(escaped_parts)
            
            # Restore <br> tags (placeholder will be escaped, so restore escaped version)
            content = content.replace(BR_PLACEHOLDER, '<br>')
            cell_tag_name = "th" if cell.column_header else "td"

            opening_tag_parts = [f"<{cell_tag_name}"]
            if cell.row_span > 1:
                opening_tag_parts.append(f' rowspan="{cell.row_span}"')
            if cell.col_span > 1:
                opening_tag_parts.append(f' colspan="{cell.col_span}"')
            opening_tag_parts.append(">")
            opening_tag = "".join(opening_tag_parts)

            html_str_list.append(f"{opening_tag}{content}</{cell_tag_name}>")
        html_str_list.append("</tr>")

    body_content = "".join(html_str_list)
    return f"<table border=\"1\">{body_content}</table>"

def convert_otsl_to_html(otsl_content: str) -> str:
    """
    Convert OTSL content to HTML format.
    Handles single tables with nested tables inside cells (using <ntable> tags).
    <ntable> tags inside cell content will be converted to HTML tables.
    """
    # Process as a single table - nested tables inside cells will be handled
    # by export_to_html -> process_nested_tables_in_text
    result = convert_single_table_otsl_to_html(otsl_content)
    if result:
        return f"<html>\n{result}\n</html>"
    return ""


def convert_single_table_otsl_to_html(otsl_content: str) -> str:
    """
    Convert a single table's OTSL content to HTML format.
    This is the core conversion logic for one table.
    """
    if not otsl_content or not otsl_content.strip():
        return ""
    
    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)

    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)

    table_data = TableData(
                num_rows=len(split_row_tokens),
                num_cols=(
                    max(len(row) for row in split_row_tokens) if split_row_tokens else 0
                ),
                table_cells=table_cells,
            )

    result = export_to_html(table_data)
    return result

