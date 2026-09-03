import io


def create_pdf_with_image_and_text(image_bytes: bytes, text: str) -> bytes:
    """Create a simple PDF containing the provided image and text.

    Tries to use ReportLab if available; otherwise falls back to Pillow-based PDF.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        width, height = A4

        # draw image at top, scaled to fit page width with aspect
        try:
            img = ImageReader(io.BytesIO(image_bytes))
            iw, ih = img.getSize()
            scale = width / iw
            new_h = ih * scale
            c.drawImage(img, 0, height - new_h, width=width, height=new_h)
            text_y = height - new_h - 40
        except Exception:
            # if image fails, reserve space
            text_y = height - 100

        # draw text
        text_lines = text.splitlines()
        y = text_y
        for line in text_lines:
            if y < 40:
                c.showPage()
                y = height - 40
            c.drawString(40, y, line)
            y -= 14

        c.showPage()
        c.save()
        buf.seek(0)
        return buf.read()
    except Exception:
        # fallback using Pillow (PIL is in requirements)
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # create a new image for the PDF page
            page_w, page_h = img.size
            # append text below image
            lines = text.splitlines()
            font = ImageFont.load_default()
            line_h = font.getsize("A")[1]
            pad = 10
            total_h = page_h + pad + line_h * max(1, len(lines)) + pad
            page = Image.new("RGB", (page_w, total_h), "white")
            page.paste(img, (0, 0))
            draw = ImageDraw.Draw(page)
            y = page_h + pad
            for line in lines:
                draw.text((10, y), line, fill="black", font=font)
                y += line_h

            out = io.BytesIO()
            page.save(out, format="PDF")
            out.seek(0)
            return out.read()
        except Exception:
            # last resort: return a minimal empty PDF
            return b"%PDF-1.4\n%EOF"
