"""Run the control deck service."""

import uvicorn

uvicorn.run("mantisnet.deck.app:app", host="0.0.0.0", port=8000)
