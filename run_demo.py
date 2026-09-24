"""Entry point for local development.

A small wrapper so the demo can be launched without remembering
the full uvicorn command.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
