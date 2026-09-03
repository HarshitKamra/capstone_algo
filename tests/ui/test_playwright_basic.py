import os
import pytest


RUN_UI = os.environ.get("RUN_UI_TESTS") == "1"


@pytest.mark.skipif(not RUN_UI, reason="UI tests disabled; set RUN_UI_TESTS=1 to enable")
def test_streamlit_homepage_playwright(page):
    # Playwright pytest fixture `page` is provided by pytest-playwright
    page.goto("http://localhost:8501")
    assert "Poster AOI Visualizer" in page.title()
