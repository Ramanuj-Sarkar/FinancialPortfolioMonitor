
"""Entry point — run with: python run.py"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# sys.path is process-local — child processes spawned by uvicorn's reloader
# won't inherit it. PYTHONPATH is an env var and IS inherited, so set both.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[PROJECT_ROOT],
    )
