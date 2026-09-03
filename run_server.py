import http.server
import os
import socketserver
import subprocess
import threading
import time


INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Poster AOI Visualizer</title>
    <style>html,body,iframe{height:100%;margin:0;padding:0;border:0}</style>
  </head>
  <body>
    <iframe src="http://localhost:8502/" style="width:100%;height:100%;border:0"></iframe>
  </body>
</html>
"""


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


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def start_index_server(directory, port=8501):
    os.chdir(directory)
    handler = QuietHandler
    httpd = socketserver.TCPServer(("", port), handler)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def main():
    # write index.html into a temp dir
    tmp_dir = os.path.abspath(".run_server_static")
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(INDEX_HTML)

    st_proc = start_streamlit()
    # give Streamlit a moment to start
    time.sleep(2)
    httpd = start_index_server(tmp_dir, port=8501)

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
