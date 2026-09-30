"""
Renders a Folium map to PNG bytes via headless Chrome.
Falls back to None on any failure so the caller can skip the photo.
"""
from utils.timeutil import now_ist, utc_now
import base64
import os
import tempfile


def render_folium_map_to_png(map_html: str, width=900, height=650):
    """
    Given Folium's `_repr_html_()` output, return PNG bytes.
    Returns None if Selenium/Chrome are unavailable.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service as ChromeService
    except ImportError:
        print("[MapRenderer] selenium not installed, skipping PNG render.")
        return None

    tmp_html = None
    driver = None
    try:
        # Write HTML to a temp file so the browser loads it with proper file:// context
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".html", mode="w", encoding="utf-8"
        )
        tmp.write(map_html)
        tmp.close()
        tmp_html = tmp.name

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--window-size=900,650")
        opts.add_argument("--hide-scrollbars")

        driver = webdriver.Chrome(options=opts)
        driver.set_window_size(width, height)
        driver.get(f"file:///{tmp_html.replace(os.sep, '/')}")

        # give Leaflet tiles time to load
        import time
        time.sleep(3)

        png = driver.get_screenshot_as_png()
        return png

    except Exception as e:
        print(f"[MapRenderer] render failed: {e}")
        return None

    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass
        try:
            if tmp_html and os.path.exists(tmp_html):
                os.remove(tmp_html)
        except Exception:
            pass