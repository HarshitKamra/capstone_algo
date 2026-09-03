import http.server
import os
import socketserver
import subprocess
import threading
import time
import requests
import re


def start_streamlit():
    # Launch the Streamlit app on port 8502
    cmd = [
        os.sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/app.py",
        "--server.port",
        "8502",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        target = f"http://127.0.0.1:8502{self.path}"
        try:
            r = requests.get(target, stream=True, timeout=10)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(b"Bad Gateway")
            return

        content = r.content
        content_type = r.headers.get("Content-Type", "")

        # If HTML, rewrite <title> to desired one
        if "text/html" in content_type:
            try:
                text = content.decode(r.encoding or "utf-8")
            except Exception:
                text = content.decode("utf-8", errors="ignore")
            # replace the <title>...</title>
            text = re.sub(r"<title>.*?</title>", "<title>Poster AOI Visualizer</title>", text, flags=re.IGNORECASE | re.DOTALL)
            content = text.encode("utf-8")

        self.send_response(r.status_code)
        for k, v in r.headers.items():
            # skip hop-by-hop headers
            if k.lower() in ("content-encoding", "transfer-encoding", "content-length", "connection"):
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def start_proxy(port=8501):
    # Not used in static iframe mode
    server = socketserver.TCPServer(("", port), ProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    # create a small static index page with a sandboxed iframe so the Streamlit app
    # cannot manipulate the parent document title. The iframe loads the app on 8502.
    tmp_dir = os.path.abspath(".run_server_static")
    os.makedirs(tmp_dir, exist_ok=True)
    index_html = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Poster AOI Visualizer</title>
        <style>html,body,iframe{height:100%;margin:0;padding:0;border:0}</style>
      </head>
      <body>
        <!-- sandbox the iframe so embedded Streamlit cannot change parent title -->
        <iframe src="http://localhost:8502/" style="width:100%;height:100%;border:0"
                sandbox="allow-scripts allow-forms allow-popups allow-popups-to-escape-sandbox"></iframe>
      </body>
    </html>
    """
    with open(os.path.join(tmp_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    st_proc = start_streamlit()
    # give Streamlit a moment to start
    time.sleep(2)

    # serve static index on 8501
    os.chdir(tmp_dir)
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", 8501), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            st_proc.terminate()
        except Exception:
            pass
        httpd.shutdown()


if __name__ == "__main__":
    main()
