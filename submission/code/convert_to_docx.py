"""
Markdown to IEEE-Formatted DOCX Converter
Transforms research paper markdown documents into beautifully styled Word documents (.docx)
with full support for IEEE headings, metadata blocks, tables, embedded figures, and citations.
"""

import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table):
    """Adds subtle professional borders to table."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="333333"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_formatted_runs(paragraph, text, base_font_size=Pt(10), base_italic=False, base_bold=False, base_color=RGBColor(30, 30, 30)):
    """Parses markdown bold, italic, code formatting within a line of text."""
    # Pattern to match bold (**), italic (*), code (`)
    tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', text)
    
    for token in tokens:
        if not token:
            continue
        is_bold = base_bold
        is_italic = base_italic
        content = token
        
        if token.startswith('**') and token.endswith('**'):
            is_bold = True
            content = token[2:-2]
        elif token.startswith('*') and token.endswith('*'):
            is_italic = True
            content = token[1:-1]
        elif token.startswith('`') and token.endswith('`'):
            content = token[1:-1]
            
        run = paragraph.add_run(content)
        run.font.name = 'Times New Roman'
        run.font.size = base_font_size
        run.font.bold = is_bold
        run.font.italic = is_italic
        run.font.color.rgb = base_color

def convert_markdown_to_docx(md_path, docx_path, base_image_dir="."):
    """Converts markdown file into an IEEE-formatted DOCX."""
    doc = docx.Document()

    # Set page margins (0.75 inch for academic paper feel)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_table = False
    table_lines = []
    in_abstract = False
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\r\n')
        stripped = line.strip()

        # Handle empty lines
        if not stripped:
            if in_table and table_lines:
                render_table(doc, table_lines)
                table_lines = []
                in_table = False
            i += 1
            continue

        # Check for table rows
        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            table_lines.append(stripped)
            i += 1
            continue
        elif in_table:
            # Table ended
            render_table(doc, table_lines)
            table_lines = []
            in_table = False

        # Image tag: ![Caption](path)
        img_match = re.match(r'!\[(.*?)\]\((.*?)\)', stripped)
        if img_match:
            caption = img_match.group(1)
            img_rel_path = img_match.group(2)
            img_full_path = os.path.normpath(os.path.join(base_image_dir, img_rel_path))
            
            if not os.path.exists(img_full_path):
                # Try directly from workspace root or results dir
                if os.path.exists(img_rel_path):
                    img_full_path = img_rel_path
                elif os.path.exists(os.path.join("submission", "results", os.path.basename(img_rel_path))):
                    img_full_path = os.path.join("submission", "results", os.path.basename(img_rel_path))

            if os.path.exists(img_full_path):
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(8)
                p_img.paragraph_format.space_after = Pt(3)
                run_img = p_img.add_run()
                run_img.add_picture(img_full_path, width=Inches(5.5))

                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_cap.paragraph_format.space_before = Pt(0)
                p_cap.paragraph_format.space_after = Pt(10)
                run_cap = p_cap.add_run(f"Fig. {caption}")
                run_cap.font.name = 'Times New Roman'
                run_cap.font.size = Pt(9)
                run_cap.font.italic = True
            else:
                p_err = doc.add_paragraph(f"[Image not found: {img_rel_path}]")
                p_err.runs[0].font.italic = True
            i += 1
            continue

        # Document Title (# Title)
        if stripped.startswith('# '):
            title_text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(12)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(title_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(18)
            run.font.bold = True
            run.font.color.rgb = RGBColor(15, 30, 60)
            i += 1
            continue

        # Authors / Affiliation Block (blockquotes or italic metadata)
        if stripped.startswith('> Author') or stripped.startswith('> **Author') or stripped.startswith('> Department') or stripped.startswith('**Authors:**'):
            clean_meta = stripped.lstrip('> ').strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(3)
            add_formatted_runs(p, clean_meta, base_font_size=Pt(9.5), base_color=RGBColor(60, 60, 60))
            i += 1
            continue

        # Heading 1 (## I. INTRODUCTION, etc.)
        if stripped.startswith('## '):
            h_text = stripped[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            
            if 'ABSTRACT' in h_text.upper():
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(h_text.upper())
                run.font.name = 'Times New Roman'
                run.font.size = Pt(11)
                run.font.bold = True
                run.font.color.rgb = RGBColor(20, 20, 20)
            else:
                run = p.add_run(h_text)
                run.font.name = 'Times New Roman'
                run.font.size = Pt(11.5)
                run.font.bold = True
                run.font.color.rgb = RGBColor(20, 40, 80)
            i += 1
            continue

        # Heading 2 (### A. Preprocessing Pipeline, etc.)
        if stripped.startswith('### '):
            h_text = stripped[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10.5)
            run.font.bold = True
            run.font.italic = True
            run.font.color.rgb = RGBColor(40, 50, 70)
            i += 1
            continue

        # Heading 3 (#### 1) Sub-subsection)
        if stripped.startswith('#### '):
            h_text = stripped[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(h_text)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = RGBColor(50, 50, 50)
            i += 1
            continue

        # Blockquote (Callouts / Alerts / Keywords)
        if stripped.startswith('> '):
            quote_text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.4)
            p.paragraph_format.right_indent = Inches(0.4)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_runs(p, quote_text, base_font_size=Pt(9.5), base_italic=True, base_color=RGBColor(50, 50, 50))
            i += 1
            continue

        # List item (- or * or numbered 1.)
        bullet_match = re.match(r'^([-*]|\d+\.)\s+(.*)', stripped)
        if bullet_match:
            marker = bullet_match.group(1)
            item_text = bullet_match.group(2)
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.15
            
            run_bullet = p.add_run("• " if marker in ['-', '*'] else f"{marker} ")
            run_bullet.font.name = 'Times New Roman'
            run_bullet.font.size = Pt(10)
            run_bullet.font.bold = True
            
            add_formatted_runs(p, item_text, base_font_size=Pt(10))
            i += 1
            continue

        # Standard Body Paragraph
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4.5)
        p.paragraph_format.line_spacing = 1.15
        add_formatted_runs(p, stripped, base_font_size=Pt(10))
        i += 1

    # Clean up trailing table
    if in_table and table_lines:
        render_table(doc, table_lines)

    doc.save(docx_path)
    print(f"[SUCCESS] Exported IEEE-formatted Word document to: {docx_path}")

def render_table(doc, table_lines):
    """Renders markdown table lines into a docx.Table."""
    if len(table_lines) < 2:
        return

    # Extract rows
    rows_data = []
    for line in table_lines:
        parts = [p.strip() for p in line.strip('|').split('|')]
        # Filter separator row (e.g. ---|:---:|---)
        if all(re.match(r'^:?-+:?$', p) for p in parts):
            continue
        rows_data.append(parts)

    if not rows_data:
        return

    num_rows = len(rows_data)
    num_cols = max(len(r) for r in rows_data)

    table = doc.add_table(rows=num_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)

    for r_idx, row_data in enumerate(rows_data):
        row = table.rows[r_idx]
        is_header = (r_idx == 0)
        
        for c_idx in range(num_cols):
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
            
            text = row_data[c_idx] if c_idx < len(row_data) else ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (is_header or c_idx > 0) else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)

            if is_header:
                set_cell_background(cell, "2B5C8F")
                add_formatted_runs(p, text, base_font_size=Pt(9), base_bold=True, base_color=RGBColor(255, 255, 255))
            else:
                if r_idx % 2 == 1:
                    set_cell_background(cell, "F7F9FC")
                else:
                    set_cell_background(cell, "FFFFFF")
                add_formatted_runs(p, text, base_font_size=Pt(9), base_color=RGBColor(30, 30, 30))

    # Add space after table
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(2)
    p_spacer.paragraph_format.space_after = Pt(6)

if __name__ == "__main__":
    import sys
    md_in = sys.argv[1] if len(sys.argv) > 1 else "./submission/final_paper.md"
    docx_out = sys.argv[2] if len(sys.argv) > 2 else "./submission/final_paper.docx"
    convert_markdown_to_docx(md_in, docx_out)
