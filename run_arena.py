"""
Quantum RL Arena - Web Application Launcher.

Starts the FastAPI server on port 8000 and serves the custom interactive React web application.

Usage:
    python run_arena.py
    # or with options:
    python run_arena.py --port 8000 --no-browser
"""

import os
import sys
import subprocess
import webbrowser
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
FRONTEND_DIST = os.path.join(FRONTEND_DIR, "dist")


def ensure_frontend_built():
    """Ensure the React TypeScript frontend is compiled and ready in frontend/dist."""
    index_html = os.path.join(FRONTEND_DIST, "index.html")
    if not os.path.exists(index_html):
        print("[*] Frontend distribution not found. Building React application via npm...")
        try:
            subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True, shell=True)
            print("[+] Frontend build successful.")
        except Exception as e:
            print(f"[!] Warning: Frontend build failed ({e}). Proceeding in API mode.")
    else:
        print("[+] Frontend distribution verified at:", FRONTEND_DIST)


def main():
    parser = argparse.ArgumentParser(description="Launch Quantum RL Arena Web Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch browser")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn hot-reload")
    args = parser.parse_args()

    print("=" * 70)
    print("   QUANTUM RL ARENA - INTERACTIVE RESEARCH LABORATORY")
    print("   Classical vs Hybrid Quantum Policy Optimization Platform")
    print("=" * 70)

    # 1. Ensure frontend is built
    ensure_frontend_built()

    url = f"http://{args.host}:{args.port}"
    print(f"\n[*] Starting Quantum RL Arena at: {url}")
    print("[*] Press Ctrl+C in this terminal to shut down.\n")

    # 2. Open browser after short delay
    if not args.no_browser:
        def open_browser():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        import threading
        threading.Thread(target=open_browser, daemon=True).start()

    # 3. Launch Uvicorn
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
