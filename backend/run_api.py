#!/usr/bin/env python
"""
Simple wrapper to start the backend API with correct Python path.
"""
import os
import sys
from pathlib import Path

# Windows redirected stdout/stderr can default to a legacy "charmap"
# encoding, which crashes when Rich/log output contains symbols like
# warning signs. Force UTF-8 so console output cannot fail the request.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# Add src to path so qtrial_backend is importable
backend_dir = Path(__file__).parent
src_dir = backend_dir / "src"
sys.path.insert(0, str(src_dir))

# Set working directory
os.chdir(str(backend_dir))

if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting backend from: {backend_dir}")
    print(f"Python path includes: {src_dir}")
    print(f"API will be available at: http://127.0.0.1:8000")
    print(f"Docs at: http://127.0.0.1:8000/docs")
    print(f"\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "qtrial_backend.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(src_dir)]
    )
