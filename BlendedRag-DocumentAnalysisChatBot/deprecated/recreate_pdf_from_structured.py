"""
recreate_pdf_from_structured.py

Recreates a PDF from the structured JSON produced by pdf_to_structured.py

Features:
- Reads the JSON structured format (pages -> elements)
- Renders text blocks, tables, images, shapes (basic), headers/footers
- Honors page size and element positions (converts top-based y to ReportLab bottom-based y)
- Handles multi-line text with wrapping inside element width
- Draws simple table grids and places table cell text

Requirements:
pip install reportlab pillow

Usage:
python recreate_pdf_from_structured.py input.json output.pdf

Notes & assumptions:
- JSON positions are in PDF points (same as ReportLab). The JSON uses top-based y (0 at top). This script converts to ReportLab coordinates.
- If fonts in JSON are not available, Helvetica/Times are used as fallback.
- Images referenced in the JSON must exist at the path stored (or provide --images-dir to map).
- Table cell sizing is computed from element bbox; if heights are zero, the script estimates row heights using font size.

"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import portrait
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from PIL import Image
import textwrap

# ------------------ Helpers ------------------

def ensure_font(c, font_name: str, font_size: float):
    # ReportLab builtin fonts: 'Helvetica', 'Times-Roman', 'Courier'
    # map common names
    if not font_name:
        font_name = 'Helvetica'
    mapping = {
        'Helvetica': 'Helvetica',
        'Times': 'Times-Roman',
        'Times-Roman': 'Times-Roman',
        'Courier': 'Courier'
    }
    chosen = mapping.get(font_name.split(',')[0], 'Helvetica')
    try:
        c.setFont(chosen, font_size)
    except Exception:
        c.setFont('Helvetica', font_size)


def pdf_top_to_reportlab_y(top_y: float, height: float, page_height: float) -> float:
    """Convert top-based Y (y from top) to ReportLab bottom-based coordinate."""
    return page_height - (top_y + height)


def draw_wrapped_text(c: canvas.Canvas, x: float, y_top: float, w: float, h: float, text: str, font_name: str = 'Helvetica', font_size: float = 10, leading: float = None):
    if not text:
        return
    if leading is None:
        leading = font_size * 1.1
    # simple wrapping using textwrap at approx chars per line based on width and font_size
    # approximate avg char width
    avg_char_width = font_size * 0.5
    max_chars = max(1, int(w / avg_char_width))
    lines = []
    for paragraph in text.split('\n'):
        wrapped = textwrap.wrap(paragraph, width=max_chars)
        if not wrapped:
            lines.append('')
        else:
            lines.extend(wrapped)
    # Clip lines to fit height
    max_lines = int(h // leading)
    lines = lines[:max_lines]
    # draw lines starting from y_top (which is top coordinate); convert to baseline y
    y = y_top
    ensure_font(c, font_name, font_size)
    for i, line in enumerate(lines):
        # baseline y for ReportLab text is y (bottom-left). Our y_top is top of box. We'll compute baseline for each line
        baseline = pdf_top_to_reportlab_y(y_top, h, c._pagesize[1]) + (h - (i + 1) * leading) + (leading - font_size) / 2
        # Slightly simpler: compute start y from top
        text_x = x
        text_y = pdf_top_to_reportlab_y(y_top, h, c._pagesize[1]) + h - (i + 1) * leading + (leading - font_size) / 2
        c.drawString(text_x, text_y, line)


def draw_table(c: canvas.Canvas, x: float, y_top: float, w: float, h: float, headers: List[str], cells: List[List[Any]], font_name: str = 'Helvetica', font_size: float = 9):
    # Determine rows/cols
    n_rows = len(cells)
    n_cols = len(cells[0]) if n_rows > 0 else len(headers)
    if n_rows == 0:
        return
    # compute column widths equally
    col_w = w / max(1, n_cols)
    # compute row heights: if h>0 use equally, else estimate by font size
    if h and h > 0:
        row_h = h / max(1, n_rows)
    else:
        row_h = font_size * 1.6
    # Draw grid
    x0 = x
    y0 = pdf_top_to_reportlab_y(y_top, h if h>0 else (row_h*n_rows), c._pagesize[1])
    # horizontal lines
    for r in range(n_rows + 1):
        y_line = y0 + r * row_h
        c.setStrokeColor(colors.black)
        c.line(x0, y_line, x0 + w, y_line)
    # vertical lines
    for col in range(n_cols + 1):
        x_line = x0 + col * col_w
        c.line(x_line, y0, x_line, y0 + row_h * n_rows)
    # Fill cell texts
    ensure_font(c, font_name, font_size)
    for r in range(n_rows):
        for col in range(n_cols):
            cell_text = str(cells[r][col]) if cells[r][col] is not None else ""
            # compute text origin inside cell
            tx = x0 + col * col_w + 4
            ty = y0 + (n_rows - r - 1) * row_h + (row_h - font_size) / 2
            c.drawString(tx, ty, cell_text)

# ------------------ Main renderer ------------------

def render_document(struct: Dict[str, Any], out_pdf_path: str, images_dir_map: str = None):
    pages = struct.get('pages', [])
    # create canvas with first page size as default; we'll set pagesize per page when calling showPage
    first_page = pages[0] if pages else None
    default_size = (595, 842) if not first_page else (first_page['size']['width'], first_page['size']['height'])
    c = canvas.Canvas(out_pdf_path, pagesize=default_size)

    for p in pages:
        pw = p['size']['width']
        ph = p['size']['height']
        # set page size
        c.setPageSize((pw, ph))
        elements = p.get('elements', [])
        for elem in elements:
            etype = elem.get('type')
            pos = elem.get('position', {'x':0,'y':0,'w':pw,'h':0})
            x = pos.get('x', 0)
            y = pos.get('y', 0)
            w = pos.get('w', pw)
            h = pos.get('h', 0)
            if etype == 'text':
                font = elem.get('font') or {}
                fname = font.get('name', 'Helvetica')
                fsize = float(font.get('size', 10) or 10)
                draw_wrapped_text(c, x, y, w, h if h>0 else (ph - y), elem.get('content',''), font_name=fname, font_size=fsize)
            elif etype == 'table':
                headers = elem.get('headers', [])
                cells = elem.get('cells', [])
                # if cells is list-of-lists but headers included as first row, keep as-is
                draw_table(c, x, y, w if w>0 else pw - x, h if h>0 else 0, headers=headers, cells=cells, font_size=9)
            elif etype == 'image':
                src = elem.get('source')
                if images_dir_map and src and os.path.isabs(src) and os.path.exists(os.path.join(images_dir_map, os.path.basename(src))):
                    src = os.path.join(images_dir_map, os.path.basename(src))
                try:
                    if src and os.path.exists(src):
                        # ReportLab expects bottom-left y; convert
                        y_report = pdf_top_to_reportlab_y(y, h if h>0 else Image.open(src).height, ph)
                        # preserve aspect ratio if needed
                        if not h or h == 0:
                            img = Image.open(src)
                            iw, ih = img.size
                            # scale to fit w
                            if w and w>0:
                                scale = w / iw
                                draw_w = w
                                draw_h = ih * scale
                            else:
                                draw_w = iw
                                draw_h = ih
                        else:
                            draw_w = w
                            draw_h = h
                        c.drawImage(src, x, y_report, width=draw_w, height=draw_h, preserveAspectRatio=True)
                except Exception as e:
                    # skip image if any problem
                    print(f"Warning: failed to draw image {src}: {e}")
            else:
                # unknown element: try to render content if present
                if 'content' in elem and elem.get('content'):
                    draw_wrapped_text(c, x, y, w, h if h>0 else (ph - y), elem.get('content',''))
        c.showPage()
    c.save()

# ------------------ CLI ------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input_json')
    ap.add_argument('output_pdf')
    ap.add_argument('--images-dir', default=None, help='optional directory to map image basenames')
    args = ap.parse_args()
    with open(args.input_json, 'r', encoding='utf-8') as f:
        struct = json.load(f)
    render_document(struct, args.output_pdf, images_dir_map=args.images_dir)
    print('Rendered', args.output_pdf)
