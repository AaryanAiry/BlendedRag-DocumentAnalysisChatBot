import json
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

def reconstruct_pdf_from_json(json_path, output_pdf_path="recreated.pdf"):
    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        pdf_data = json.load(f)

    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    page_width, page_height = letter

    # Allowed built-in fonts
    allowed_fonts = ["Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Times-Roman", "Courier"]

    # Resolve base folder for relative paths
    base_dir = os.path.dirname(json_path)

    for page in pdf_data["pages"]:
        page_w = page.get("width", page_width)
        page_h = page.get("height", page_height)
        c.setPageSize((page_w, page_h))

        for element in page["elements"]:
            etype = element["type"]
            pos = element["position"]

            if etype == "textbox":
                font_name = element.get("font", {}).get("name", "Helvetica")
                font_size = element.get("font", {}).get("size", 12)
                bold = element.get("font", {}).get("bold", False)
                italic = element.get("font", {}).get("italic", False)

                # Map unknown fonts to Helvetica variants
                if "Bold" in font_name or bold:
                    font_name = "Helvetica-Bold"
                elif "Oblique" in font_name or italic:
                    font_name = "Helvetica-Oblique"
                elif font_name not in allowed_fonts:
                    font_name = "Helvetica"

                try:
                    c.setFont(font_name, font_size)
                except Exception:
                    c.setFont("Helvetica", 12)

                x = pos["x"]
                y = page_h - pos["y"] - pos["height"]
                text = element.get("content", "")
                c.drawString(x, y, text)

            elif etype == "image":
                img_path = element.get("src")
                if img_path:
                    # Fix relative path issue
                    if not os.path.isabs(img_path):
                        img_path = os.path.join(base_dir, img_path)

                    if os.path.exists(img_path):
                        try:
                            img = ImageReader(img_path)
                            x = pos["x"]
                            y = page_h - pos["y"] - pos["height"]
                            width = pos["width"]
                            height = pos["height"]
                            c.drawImage(img, x, y, width=width, height=height)
                            print(f"✅ Placed image: {img_path}")
                        except Exception as e:
                            print(f"⚠️ Failed to draw image {img_path}: {e}")
                    else:
                        print(f"⚠️ Image not found: {img_path}")

            elif etype == "table":
                x0 = pos["x"]
                y0 = page_h - pos["y"] - pos["height"]
                table_data = element.get("content", [])

                if table_data:
                    num_rows = len(table_data)
                    num_cols = len(table_data[0]) if num_rows > 0 else 0
                    if num_rows > 0 and num_cols > 0:
                        cell_w = pos["width"] / num_cols
                        cell_h = pos["height"] / num_rows

                        # Draw cell borders
                        for i in range(num_rows + 1):
                            c.line(x0, y0 + i * cell_h, x0 + pos["width"], y0 + i * cell_h)
                        for j in range(num_cols + 1):
                            c.line(x0 + j * cell_w, y0, x0 + j * cell_w, y0 + pos["height"])

                        # Draw cell content
                        for i, row in enumerate(table_data):
                            for j, cell_text in enumerate(row):
                                text_x = x0 + j * cell_w + 2
                                text_y = y0 + pos["height"] - (i + 1) * cell_h + 2
                                c.setFont("Helvetica", 10)
                                c.drawString(text_x, text_y, str(cell_text))

        c.showPage()

    c.save()
    print(f"✅ Recreated PDF saved at: {output_pdf_path}")


if __name__ == "__main__":
    reconstruct_pdf_from_json("output_json/sample.json", "recreated_sample.pdf")
