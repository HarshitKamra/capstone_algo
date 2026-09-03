from utils.pdf_export import create_pdf_with_image_and_text


def test_create_pdf_with_image_and_text():
    # create a tiny red PNG
    from PIL import Image
    import io

    img = Image.new("RGB", (100, 50), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    pdf = create_pdf_with_image_and_text(img_bytes, "Hello\nWorld")
    assert pdf[:4] == b"%PDF"
