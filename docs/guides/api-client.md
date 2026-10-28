# Guide: API client

## A script: create, run, approve, read

One Python file, standard library only. It creates a session with a first
turn, follows the event stream to the end, answers the approval the turn
asks for, and reads the answer back. Point it at a gateway whose model is
a scripted or local model that wants a `python_exec` approval for a plain
sentence; the workspace must exist and hold a few files. The script reads the
gateway token from its private file; it accepts only HTTP loopback URLs,
disables proxies and refuses redirects so it does not forward the token to
another host. Make sure the port belongs to your gateway: another local process
can impersonate it.

```python
"""Create a session, run a turn, answer its approval, read the result."""

import json
import sys
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

BASE = sys.argv[1].rstrip("/")
WORKSPACE = sys.argv[2]
url = urllib.parse.urlsplit(BASE)
if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or url.username is not None or url.password is not None
        or url.path or url.query or url.fragment):
    raise SystemExit("Use an HTTP loopback gateway URL without a path")
TOKEN = (Path.home() / ".cache/localmelo/gateway/auth/token").read_text().strip()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def request(path, *, data=None):
    return urllib.request.Request(
        BASE + path, data=data,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


def post(path, body):
    req = request(path, data=json.dumps(body).encode())
    req.add_header("Content-Type", "application/json")
    with opener.open(req) as response:
        return response.status, json.load(response)


def get(path):
    with opener.open(request(path)) as response:
        return json.load(response)


def stream(session_id):
    """Every event of the session, from id 0, as the stream sends it."""
    path = f"/api/sessions/{session_id}/events?after=0"
    with opener.open(request(path)) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("data:"):
                yield json.loads(line[5:].strip())


status, created = post(
    "/api/sessions",
    {
        "workspace": WORKSPACE,
        "turn": {
            "text": "run a short Python program",
            "client_request_id": str(uuid.uuid4()),
        },
    },
)
session_id = created["session"]["id"]
print(f"created {status} session {session_id} turn {created['turn_id']}")

for event in stream(session_id):
    print("event", event["type"])
    if event["type"] == "approval.requested":
        request_id = event["data"]["request_id"]
        print(f"  asks to run {event['data']['tool_name']}: {event['data']['preview']}")
        status, answered = post(f"/api/approvals/{request_id}", {"choice": "once"})
        print(f"  answered {status} {answered['outcome']}")
    if event["type"] == "turn.finished":
        print(f"turn {event['data']['status']} cause {event['data']['cause']!r}")
        break

turn = get(f"/api/sessions/{session_id}/turns?limit=1")["turns"][0]
print(f"answer: {turn['answer']}")
print(f"tokens: {turn['result']['usage'].get('completion_tokens', 0)}")
```

Run as `python demo.py http://127.0.0.1:8401 ~/demo-project`, against a
gateway whose model answered the message with a `python_exec` call. This is
illustrative output: additional events such as `run.progress` may appear, and
token counts depend on the model and provider:

```text
created 201 session edb425a382be turn 2e2b132e1dcc49e283c52782858ee051
event turn.started
event session.title
event run.started
event step.started
event assistant.message
event tool.started
event approval.requested
  asks to run python_exec: print(6*7)
  answered 200 allowed
event approval.resolved
event tool.finished
event step.started
event assistant.message
event run.finished
event turn.finished
turn completed cause 'run'
answer: The program printed 42.
tokens: 24
```

Two details the script relies on. `urllib` reads an event stream line by
line, so a frame is complete when its `data:` line arrives; a client that
answers an approval mid-stream keeps reading the same response. And a
turn's outcome and its `turn.finished` are written in one transaction, so
the `turns` read after the stream ends sees the answer.
