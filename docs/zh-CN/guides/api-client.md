<a id="guide-api-client"></a>

# 指南：API 客户端

<a id="a-script-create-run-approve-read"></a>

## 脚本：创建、运行、审批、读取

这个 Python 文件只使用标准库。它创建一个包含首轮消息的会话，跟踪事件流直到结束，响应该轮请求的审批，再读取答案。请将它指向一个网关，其脚本化模型或本地模型会针对一句普通的话请求 `python_exec` 审批；工作区必须存在，并包含几个文件。

```python
"""Create a session, run a turn, answer its approval, read the result."""

import json
import sys
import urllib.request
import uuid

BASE = sys.argv[1]
WORKSPACE = sys.argv[2]


def post(path, body):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return response.status, json.load(response)


def get(path):
    with urllib.request.urlopen(BASE + path) as response:
        return json.load(response)


def stream(session_id):
    """Every event of the session, from id 0, as the stream sends it."""
    url = f"{BASE}/api/sessions/{session_id}/events?after=0"
    with urllib.request.urlopen(url) as response:
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

使用 `python demo.py http://127.0.0.1:8401 ~/demo-project` 运行，连接的网关中，模型以 `python_exec` 调用响应了消息：

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

脚本依赖两个细节。`urllib` 逐行读取事件流，因此收到 `data:` 行时，一个帧就完整了；在流传输过程中响应审批的客户端继续读取同一个响应。另外，一轮的结果及其 `turn.finished` 写在同一个事务中，因此流结束后读取 `turns` 就能看到答案。
