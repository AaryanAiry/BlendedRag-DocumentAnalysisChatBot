"""
pdf_to_structured.py

Converts a PDF into a structured JSON representation suitable for RAG + editing + reconstruction.

Features:
- Extracts text blocks (with font size, bbox, reading order)
- Detects multi-column layouts via simple 1D k-means on word x-centers
- Extracts tables using Camelot (fallback to pdfplumber table detection)
- Extracts images using PyMuPDF, runs OCR on images (pytesseract) and records image bbox and saved path
- Stores structured JSON with pages -> ordered elements (text, table, image)
- Keeps position (x,y,w,h) and column index for layout reconstruction

Limitations / Notes:
- Camelot works best on lattice/stream-detectable tables (tables with ruling lines or regular whitespace)
- OCR quality depends on Tesseract installation + language models
- Exact font names may not always be available from pdfplumber

Requirements (install via pip):
pip install pdfplumber pymupdf pillow pytesseract camelot-py[cv] pandas numpy

You must have Tesseract OCR installed on your system for pytesseract to work.
On Arch Linux: `sudo pacman -S tesseract tesseract-data-eng`

Usage:
python pdf_to_structured.py input.pdf output.json --images-dir ./extracted_images

"""

import os
import json
import math
import tempfile
from typing import List, Dict, Any, Tuple

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

# camelot may not be usable in all environments; import lazily
try:
    import camelot
    HAS_CAMELOT = True
except Exception:
    HAS_CAMELOT = False

import numpy as np
import pandas as pd

# ---------------------------
# Helpers: geometry & kmeans1d
# ---------------------------

def bbox_from_word(word: Dict[str, Any]) -> Dict[str, float]:
    # pdfplumber word has x0, top, x1, bottom
    return {
        "x": float(word.get("x0", 0)),
        "y": float(word.get("top", 0)),
        "w": float(word.get("x1", 0)) - float(word.get("x0", 0)),
        "h": float(word.get("bottom", 0)) - float(word.get("top", 0)),
    }


def rect_union(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    x = min(a["x"], b["x"])
    y = min(a["y"], b["y"])
    x2 = max(a["x"] + a["w"], b["x"] + b["w"])
    y2 = max(a["y"] + a["h"], b["y"] + b["h"])
    return {"x": x, "y": y, "w": x2 - x, "h": y2 - y}


def kmeans_1d(values: List[float], k: int = 2, max_iters: int = 50) -> List[int]:
    """
    Simple 1D k-means clustering. Returns cluster index for each value.
    """
    if not values:
        return []
    arr = np.array(values, dtype=float)
    # Initialize centers by percentiles
    centers = np.percentile(arr, np.linspace(0, 100, k + 2)[1:-1])
    for _ in range(max_iters):
        # assign
        dists = np.abs(arr.reshape(-1, 1) - centers.reshape(1, -1))
        labels = np.argmin(dists, axis=1)
        new_centers = []
        for i in range(k):
            members = arr[labels == i]
            if len(members) == 0:
                new_centers.append(centers[i])
            else:
                new_centers.append(members.mean())
        new_centers = np.array(new_centers)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels.tolist()

# ---------------------------
# Core extraction functions
# ---------------------------

def extract_images_with_fitz(pdf_path: str, images_dir: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Extract images from PDF using PyMuPDF (fitz).
    Returns a dict keyed by (page_num, xref) where xref is fitz image xref.
    Each value contains: path, bbox (approx), width, height
    Note: fitz gives image streams; mapping to exact bbox on page requires additional steps.
    We'll gather images and also approximate bbox by scanning page.get_images and searching occurrences.
    """
    os.makedirs(images_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images = {}
    for pnum in range(len(doc)):
        page = doc[pnum]
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            fmt = base_image.get("ext", "png")
            img_name = f"page{pnum+1}_img{xref}.{fmt}"
            path = os.path.join(images_dir, img_name)
            with open(path, "wb") as f:
                f.write(img_bytes)
            # attempt to get bbox by searching for image occurrences on page (approx)
            # page.get_images does not give bbox; use display list & pixmap rendering approach
            images[(pnum + 1, xref)] = {
                "path": path,
                "xref": xref,
                "width": base_image.get("width"),
                "height": base_image.get("height"),
                # bbox to be filled later if needed
                "bbox": None
            }
    doc.close()
    return images


def ocr_image_get_data(image_path: str) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    try:
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception as e:
        ocr_data = {"text": "", "conf": []}
    # Build concatenated text and boxes
    text_items = []
    n = len(ocr_data.get("text", []))
    for i in range(n):
        txt = ocr_data["text"][i].strip()
        if not txt:
            continue
        x = int(ocr_data["left"][i])
        y = int(ocr_data["top"][i])
        w = int(ocr_data["width"][i])
        h = int(ocr_data["height"][i])
        conf = int(float(ocr_data.get("conf", ["-1"])[i])) if i < len(ocr_data.get("conf", [])) else -1
        text_items.append({"text": txt, "bbox": {"x": x, "y": y, "w": w, "h": h}, "conf": conf})
    full_text = " ".join([t["text"] for t in text_items])
    return {"text": full_text, "items": text_items}


def normalize_y(y: float, page_height: float) -> float:
    # Convert pdfplumber y (top) to a consistent coordinate if needed. Keep as-is for now.
    return y


def extract_structured_from_pdf(pdf_path: str, images_dir: str = "extracted_images") -> Dict[str, Any]:
    """
    Main driver: parse PDF and return structured JSON-like dict.
    """
    doc_struct = {
        "metadata": {},
        "pages": []
    }

    images_map = extract_images_with_fitz(pdf_path, images_dir)

    with pdfplumber.open(pdf_path) as pdf:
        doc_meta = pdf.metadata or {}
        doc_struct["metadata"] = {k: doc_meta.get(k) for k in ["Title", "Author", "CreationDate"] if doc_meta.get(k)}

        for pnum, page in enumerate(pdf.pages, start=1):
            width = page.width
            height = page.height
            page_obj = {"page_number": pnum, "size": {"width": width, "height": height}, "layout": {}, "elements": []}

            # -------------------
            # Extract words & chars
            # -------------------
            words = page.extract_words(extra_attrs=["fontname", "size"])  # list of dicts
            # If no words, fallback to extract_text
            if not words:
                raw_text = page.extract_text()
                if raw_text:
                    page_obj["elements"].append({
                        "type": "text",
                        "content": raw_text,
                        "font": None,
                        "position": {"x": 0, "y": 0, "w": width, "h": height},
                        "column": 0
                    })
                doc_struct["pages"].append(page_obj)
                continue

            # Determine columns by clustering word x-centers
            x_centers = []
            for w in words:
                x0 = float(w.get("x0", 0))
                x1 = float(w.get("x1", 0))
                xc = (x0 + x1) / 2.0
                x_centers.append(xc)

            # Decide number of columns: try k=1..3 and pick best silhouette-ish heuristic
            best_k = 1
            best_score = -1e9
            assignments_for_best = [0] * len(x_centers)
            for k in range(1, 4):
                if len(x_centers) < k:
                    break
                labels = kmeans_1d(x_centers, k=k)
                # score: prefer well-separated clusters (variance between centers / within)
                centers = [np.mean([x_centers[i] for i in range(len(x_centers)) if labels[i] == c]) for c in range(k)]
                between_var = np.var(centers)
                within_var = np.mean([np.var([x_centers[i] for i in range(len(x_centers)) if labels[i] == c]) if any(labels[i] == c for i in range(len(x_centers))) else 0 for c in range(k)])
                score = (between_var - within_var)
                if score > best_score:
                    best_score = score
                    best_k = k
                    assignments_for_best = labels

            num_columns = best_k
            page_obj["layout"]["columns"] = num_columns

            # Group words into blocks by vertical proximity and column
            blocks = []  # each block: {"words": [...], "bbox": {...}, "text": "...", "font": {...}}
            # We'll cluster words into lines then into blocks
            # Build preliminary word objects with positions
            word_objs = []
            for idx, w in enumerate(words):
                o = {
                    "text": w.get("text", ""),
                    "x0": float(w.get("x0", 0)),
                    "x1": float(w.get("x1", 0)),
                    "top": float(w.get("top", 0)),
                    "bottom": float(w.get("bottom", 0)),
                    "fontname": w.get("fontname"),
                    "size": w.get("size"),
                    "col": int(assignments_for_best[idx])
                }
                word_objs.append(o)

            # group into lines by similar top coordinate
            lines = []
            word_objs_sorted = sorted(word_objs, key=lambda x: (x["col"], x["top"], x["x0"]))
            line_tolerance = 3  # points
            for w in word_objs_sorted:
                if not lines:
                    lines.append({"col": w["col"], "top": w["top"], "words": [w]})
                    continue
                last = lines[-1]
                if abs(w["top"] - last["top"]) <= line_tolerance and w["col"] == last["col"]:
                    last["words"].append(w)
                    last["top"] = (last["top"] + w["top"]) / 2.0
                else:
                    lines.append({"col": w["col"], "top": w["top"], "words": [w]})

            # merge consecutive lines into paragraph blocks if vertically close
            blocks = []
            para_tolerance = 6
            for ln in lines:
                text = " ".join([w["text"] for w in ln["words"]])
                bbox = {
                    "x": min([w["x0"] for w in ln["words"]]),
                    "y": min([w["top"] for w in ln["words"]]),
                    "w": max([w["x1"] for w in ln["words"]]) - min([w["x0"] for w in ln["words"]]),
                    "h": max([w["bottom"] for w in ln["words"]]) - min([w["top"] for w in ln["words"]])
                }
                block = {"text": text, "bbox": bbox, "col": ln["col"], "font": {"name": ln["words"][0].get("fontname"), "size": ln["words"][0].get("size")}}
                # try to append to previous block if close vertically and same column
                if blocks and blocks[-1]["col"] == block["col"] and abs(block["bbox"]["y"] - (blocks[-1]["bbox"]["y"] + blocks[-1]["bbox"]["h"])) <= para_tolerance:
                    # merge
                    blocks[-1]["text"] += "\n" + block["text"]
                    blocks[-1]["bbox"] = rect_union(blocks[-1]["bbox"], block["bbox"])
                else:
                    blocks.append(block)

            # Add text blocks as elements in reading order: sort by column then y ascending
            blocks_sorted = sorted(blocks, key=lambda b: (b["col"], b["bbox"]["y"]))
            for b in blocks_sorted:
                elem = {
                    "type": "text",
                    "content": b["text"],
                    "font": b.get("font"),
                    "position": {"x": b["bbox"]["x"], "y": b["bbox"]["y"], "w": b["bbox"]["w"], "h": b["bbox"]["h"]},
                    "column": int(b["col"])
                }
                page_obj["elements"].append(elem)

            # -------------------
            # Extract tables (prefer Camelot)
            # -------------------
            tables_on_page = []
            if HAS_CAMELOT:
                try:
                    # Camelot uses page numbers starting at 1 and supports flavor 'lattice' and 'stream'
                    tables = camelot.read_pdf(pdf_path, pages=str(pnum), flavor='stream')
                    for t in tables:
                        df = t.df
                        # build table element
                        table_elem = {
                            "type": "table",
                            "title": None,
                            "n_rows": int(df.shape[0]),
                            "n_cols": int(df.shape[1]),
                            "headers": df.iloc[0].tolist() if df.shape[0] > 0 else [],
                            "cells": df.values.tolist(),
                            "position": {"x": 0, "y": 0, "w": width, "h": 0},
                            "page": pnum
                        }
                        tables_on_page.append(table_elem)
                except Exception:
                    tables_on_page = []

            # fallback: pdfplumber table extraction
            if not tables_on_page:
                try:
                    pdfpl_tables = page.extract_tables()
                    for t in pdfpl_tables:
                        df = pd.DataFrame(t)
                        table_elem = {
                            "type": "table",
                            "title": None,
                            "n_rows": int(df.shape[0]),
                            "n_cols": int(df.shape[1]),
                            "headers": df.iloc[0].tolist() if df.shape[0] > 0 else [],
                            "cells": df.values.tolist(),
                            "position": {"x": 0, "y": 0, "w": width, "h": 0},
                            "page": pnum
                        }
                        tables_on_page.append(table_elem)
                except Exception:
                    pass

            # attach tables into elements at approximate positions (we didn't compute bbox precisely)
            for t in tables_on_page:
                page_obj["elements"].append(t)

            # -------------------
            # Attach images with OCR & captions
            # -------------------
            # match images_map keys with this page
            page_images = {k: v for k, v in images_map.items() if k[0] == pnum}
            for (page_no, xref), imginfo in page_images.items():
                img_path = imginfo["path"]
                ocr = ocr_image_get_data(img_path)
                img_elem = {
                    "type": "image",
                    "caption": None,
                    "alt_text": ocr.get("text", ""),
                    "source": img_path,
                    "ocr": ocr,
                    "position": imginfo.get("bbox") or {"x": 0, "y": 0, "w": imginfo.get("width"), "h": imginfo.get("height")},
                }
                page_obj["elements"].append(img_elem)

            # Optionally: detect captions by nearby small-font text under images
            # naive approach: for each image, search for text blocks below its bbox and attach as caption
            for elem in page_obj["elements"]:
                if elem["type"] == "image":
                    img_pos = elem["position"]
                    # find nearest text block whose y > img bottom within tolerance
                    candidates = [e for e in page_obj["elements"] if e["type"] == "text"]
                    img_bottom = img_pos["y"] + img_pos.get("h", 0)
                    cap = None
                    for t in candidates:
                        ty = t["position"]["y"]
                        if ty >= img_bottom and ty - img_bottom < 40 and t["position"]["x"] >= img_pos["x"] - 20 and t["position"]["x"] <= img_pos["x"] + img_pos.get("w", width) + 20:
                            cap = t["content"]
                            break
                    if cap:
                        elem["caption"] = cap

            # final: sort elements by column then y
            page_obj["elements"] = sorted(page_obj["elements"], key=lambda e: (e.get("column", 0), e["position"]["y"]))

            doc_struct["pages"].append(page_obj)

    return doc_struct


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDF to structured JSON representation")
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("out", help="Output JSON path")
    parser.add_argument("--images-dir", default="extracted_images", help="Directory to save extracted images")

    args = parser.parse_args()
    pdf_path = args.pdf
    out_path = args.out
    images_dir = args.images_dir

    print(f"Processing {pdf_path} -> {out_path}")
    struct = extract_structured_from_pdf(pdf_path, images_dir=images_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(struct, f, indent=2, ensure_ascii=False)
    print("Done.")
