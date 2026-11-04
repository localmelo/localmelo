# tests/smoke: real-backend smoke

`test_backend_smoke.py` runs one chat round trip against a local Ollama
(`127.0.0.1:11434`) and a local MLC server (`127.0.0.1:8400`). It is opt-in:

```bash
LOCALMELO_SMOKE=1 python -m pytest -m smoke_backend tests/smoke -q
```

- A plain `pytest` run skips both tests, with a hint, and collecting the file
  opens no socket.
- With `LOCALMELO_SMOKE=1` a test connects to its backend's health URL when it
  runs (0.5 s timeout), sends a real chat request to the server that answers, and
  is skipped for the one that does not.
- `-m 'not smoke_backend'` deselects them.
