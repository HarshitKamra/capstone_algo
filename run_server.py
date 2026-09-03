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
    server = socketserver.TCPServer(("", port), ProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    st_proc = start_streamlit()
    # give Streamlit a moment to start
    time.sleep(2)
    proxy = start_proxy(8501)

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
        proxy.shutdown()


if __name__ == "__main__":
    main()
