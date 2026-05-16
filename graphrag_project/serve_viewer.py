"""
一键启动: 导出 GraphRAG 数据 → 启动 HTTP 服务 → 打开浏览器
用法: python serve_viewer.py [--top N] [--port PORT]
"""
import argparse
import http.server
import os
import subprocess
import sys
import threading
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=0, help="Only export top N entities by degree")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    export_cmd = [sys.executable, os.path.join(ROOT, "export_graph_json.py")]
    if args.top > 0:
        export_cmd += ["--top", str(args.top)]

    print("=== Exporting graph data ===")
    ret = subprocess.run(export_cmd, cwd=ROOT)
    if ret.returncode != 0:
        print("[ERROR] Export failed.")
        sys.exit(1)

    os.chdir(ROOT)
    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(("0.0.0.0", args.port), handler)

    url = f"http://localhost:{args.port}/graph_viewer.html"
    print(f"\n=== Serving at {url} ===")
    print("Press Ctrl+C to stop.\n")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
