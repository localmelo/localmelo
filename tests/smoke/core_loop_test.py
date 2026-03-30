#!/usr/bin/env python3
"""数据驱动的核心循环基准/演示框架。

从 tests/smoke/data/ 加载场景与后端配置，对每个组合执行核心循环
（种入记忆 → 检索 → 代理回答），收集嵌入/对话/端到端三类延迟指标，
并将详细 JSON + Markdown 报告写入 tests/smoke/output/。

支持单后端或多后端运行；多后端时自动生成延迟与召回率对比报告。
后端不可用时记录 skipped 而非崩溃。

Usage
-----
    # 运行默认后端 (ollama)、全部场景:
    python tests/smoke/core_loop_test.py

    # 仅运行 mlc 后端:
    python tests/smoke/core_loop_test.py --backends mlc

    # 多后端对比:
    python tests/smoke/core_loop_test.py --backends ollama,mlc

    # 指定场景:
    python tests/smoke/core_loop_test.py --scenarios personal_preference

    # 全部后端:
    python tests/smoke/core_loop_test.py --backends all

    # 自定义输出目录:
    python tests/smoke/core_loop_test.py --out-dir ./my_reports

    # 环境变量覆盖 (优先于 backends.json 默认值):
    SMOKE_CHAT_URL=http://host:port/v1 \
    SMOKE_CHAT_MODEL=my-model \
        python tests/smoke/core_loop_test.py --backends ollama
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re as _re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure the repo root is importable as package "localmelo".
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO.parent))

from localmelo.melo.agent.agent import Agent  # noqa: E402
from localmelo.melo.contracts.providers import BaseLLMProvider  # noqa: E402
from localmelo.melo.schema import Message  # noqa: E402
from localmelo.support.backends.tokenization import count_tokens  # noqa: E402
from localmelo.support.providers.embedding.openai_compat import (  # noqa: E402
    OpenAICompatEmbedding,
)
from localmelo.support.providers.llm.ollama_chat import (  # noqa: E402
    OllamaNativeChat,
)
from localmelo.support.providers.llm.openai_compat import (  # noqa: E402
    OpenAICompatLLM,
)

log = logging.getLogger("smoke")


def _safe_int(value: Any) -> int:
    """Coerce a value to non-negative int; return 0 on failure."""
    if value is None:
        return 0
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(n, 0)


def _safe_float(value: Any) -> float:
    """Coerce a value to non-negative float; return 0.0 on failure."""
    if value is None:
        return 0.0
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(n, 0.0)


def _split_thinking_and_answer(text: str) -> tuple[str, str]:
    """Extract ``<think>...</think>`` block from *text*.

    Returns (thinking, answer) where *thinking* is the content inside the
    first ``<think>`` tag (stripped) and *answer* is the remaining text
    (stripped).  When no ``<think>`` tag is present both parts default to
    ``("", text.strip())``.
    """
    if not text:
        return ("", "")
    m = _re.search(r"<think>(.*?)</think>", text, _re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        answer = (text[: m.start()] + text[m.end() :]).strip()
        return (thinking, answer)
    return ("", text.strip())


def _split_thinking(resp: Message) -> tuple[str, str]:
    """Extract (thinking, answer_only) from a response.

    Handles:
    - Ollama native: resp.thinking has the thinking text
    - MLC style: resp.content contains <think>...</think> inline
    """
    thinking = resp.thinking or ""
    content = resp.content

    # If thinking came from native field, answer_only is just content
    if thinking:
        # Content might be the combined form or just the answer
        # If content starts with <think>, strip it to get answer_only
        answer_only = _re.sub(
            r"<think>.*?</think>\s*", "", content, flags=_re.DOTALL
        ).strip()
        if not answer_only:
            answer_only = content
        return thinking, answer_only

    # MLC style: parse <think>...</think> from content
    match = _re.match(r"<think>\s*(.*?)\s*</think>\s*(.*)", content, _re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return "", content


_SMOKE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SMOKE_DIR / "data"
_DEFAULT_OUT_DIR = _SMOKE_DIR / "output"


# ── Data loading ────────────────────────────────────────────


def load_scenarios(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    p = path or (_DATA_DIR / "scenarios.json")
    return json.loads(p.read_text("utf-8"))  # type: ignore[no-any-return]


def load_backends(
    path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    p = path or (_DATA_DIR / "backends.json")
    return json.loads(p.read_text("utf-8"))  # type: ignore[no-any-return]


# ── Instrumented providers ──────────────────────────────────


class LoggingLLM(BaseLLMProvider):
    """Instrumented wrapper around any :class:`BaseLLMProvider`.

    Delegates ``chat()`` to the wrapped *inner* provider while recording
    both backend-reported usage metrics (for operational visibility)
    and normalized token counts computed by the shared backend tokenizer
    (for cross-backend comparison).
    """

    def __init__(self, inner: BaseLLMProvider) -> None:
        self._inner = inner
        self.call_log: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return getattr(self._inner, "model", "unknown")

    async def chat(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> Message:
        req = [{"role": m.role, "content": m.content} for m in messages]
        t0 = time.time()
        resp = await self._inner.chat(messages, tools)
        elapsed = (time.time() - t0) * 1000

        # -- Backend-reported usage (keep as-is) --
        usage = resp.usage or {}
        prompt_tokens = _safe_int(usage.get("prompt_tokens"))
        completion_tokens = _safe_int(usage.get("completion_tokens"))
        total_tokens = _safe_int(usage.get("total_tokens"))
        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens
        elapsed_s = elapsed / 1000 if elapsed > 0 else 1e-9

        # -- Normalized token counts (shared tokenizer) --
        norm_prompt = sum(count_tokens(m.content) for m in messages)
        thinking, answer_only = _split_thinking(resp)
        norm_thinking = count_tokens(thinking)
        norm_answer = count_tokens(answer_only)
        norm_completion = norm_thinking + norm_answer
        norm_total = norm_prompt + norm_completion

        self.call_log.append(
            {
                "request_messages": req,
                "tools_sent": bool(tools),
                "response_role": resp.role,
                "response_content": resp.content[:500],
                "has_tool_call": resp.tool_call is not None,
                "elapsed_ms": round(elapsed, 1),
                # Backend-reported usage
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "completion_tokens_per_s": (
                    round(completion_tokens / elapsed_s, 1) if completion_tokens else 0
                ),
                "total_tokens_per_s": (
                    round(total_tokens / elapsed_s, 1) if total_tokens else 0
                ),
                "has_usage": bool(resp.usage),
                # Thinking / answer split
                "thinking": thinking[:500],
                "answer_only": answer_only[:500],
                "normalized_thinking_tokens": norm_thinking,
                "normalized_answer_tokens": norm_answer,
                # Normalized (shared tokenizer)
                "normalized_prompt_tokens": norm_prompt,
                "normalized_completion_tokens": norm_completion,
                "normalized_total_tokens": norm_total,
                "normalized_completion_tokens_per_s": (
                    round(norm_completion / elapsed_s, 1) if norm_completion else 0
                ),
                "normalized_total_tokens_per_s": (
                    round(norm_total / elapsed_s, 1) if norm_total else 0
                ),
            }
        )
        return resp

    async def close(self) -> None:
        await self._inner.close()


class LoggingEmbedding(OpenAICompatEmbedding):
    """Thin wrapper that records every embed call with timing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.call_log: list[dict[str, Any]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        t0 = time.time()
        result = await super().embed(texts)
        elapsed = (time.time() - t0) * 1000
        self.call_log.append(
            {
                "n_texts": len(texts),
                "elapsed_ms": round(elapsed, 1),
            }
        )
        return result


# ── Health check ────────────────────────────────────────────


def check_backend_health(
    backend: dict[str, Any],
    timeout: float = 5.0,
) -> tuple[bool, str]:
    """Probe the backend with a quick HTTP GET; return (ok, message)."""
    from urllib.error import URLError
    from urllib.request import urlopen

    url = backend.get(
        "health_url",
        backend["chat_url"].rstrip("/") + "/models",
    )
    try:
        with urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return True, f"OK ({resp.status})"
    except URLError as exc:
        return False, str(exc.reason)
    except OSError as exc:
        return False, str(exc)


# ── SQLite helpers ──────────────────────────────────────────


def _count_rows(db_path: str, table: str) -> int:
    if not os.path.exists(db_path):
        return -1
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(f"SELECT count(*) FROM {table}")  # noqa: S608
        return cur.fetchone()[0]  # type: ignore[no-any-return]
    except sqlite3.OperationalError:
        return -1
    finally:
        conn.close()


# ── Core scenario runner ───────────────────────────────────


async def run_scenario(
    backend_cfg: dict[str, Any],
    scenario: dict[str, Any],
    mem_dir: str,
) -> dict[str, Any]:
    """Run one scenario against one backend. Return the full record."""
    # Select the appropriate chat provider based on backend type.
    backend_id = backend_cfg.get("id", "")
    if backend_id == "ollama":
        # Use native Ollama /api/chat with think:true support.
        # Strip /v1 suffix if present (backends.json stores the OpenAI-compat URL).
        chat_url = backend_cfg["chat_url"].rstrip("/")
        if chat_url.endswith("/v1"):
            chat_url = chat_url[:-3]
        inner_llm: BaseLLMProvider = OllamaNativeChat(
            base_url=chat_url,
            model=backend_cfg["chat_model"],
            timeout=300.0,
        )
    else:
        inner_llm = OpenAICompatLLM(
            base_url=backend_cfg["chat_url"],
            model=backend_cfg["chat_model"],
            timeout=300.0,
        )
    llm = LoggingLLM(inner=inner_llm)
    embedding = LoggingEmbedding(
        base_url=backend_cfg["embed_url"],
        model=backend_cfg["embed_model"],
        timeout=60.0,
    )

    os.environ["LOCALMELO_PERSIST_MEMORY"] = "1"
    os.environ["LOCALMELO_MEMORY_DIR"] = mem_dir

    agent = Agent(llm=llm, embedding=embedding)

    t_start = time.time()

    # ── Seed memories ──
    n_seeds = len(scenario["seed_memories"])
    print(f"  [seed] 种入 {n_seeds} 条记忆 ...")
    for si, fact in enumerate(scenario["seed_memories"], 1):
        t_s = time.time()
        await agent.hippo.memorize(fact)
        ms = (time.time() - t_s) * 1000
        print(f"    seed [{si}/{n_seeds}] {ms:.0f}ms  {fact[:60]}")

    # ── Run each query ──
    n_queries = len(scenario["queries"])
    query_results: list[dict[str, Any]] = []
    for qi, q in enumerate(scenario["queries"], 1):
        print(f"  [query {qi}/{n_queries}] {q['text']}")

        t_q = time.time()
        answer = await agent.run(q["text"])
        q_elapsed = (time.time() - t_q) * 1000

        expected = q.get("expected_keywords", [])
        found = [kw for kw in expected if kw in answer]
        recall = len(found) / len(expected) if expected else 1.0

        elapsed_total = (time.time() - t_start) * 1000
        print(
            f"    -> {q_elapsed:.0f}ms  "
            f"recall={recall:.0%}  "
            f"answer={answer[:80]}..."
            if len(answer) > 80
            else f"    -> {q_elapsed:.0f}ms  "
            f"recall={recall:.0%}  "
            f"answer={answer}"
        )
        print(f"    elapsed={elapsed_total / 1000:.1f}s total")

        # Parse thinking from answer string (works for both MLC and
        # Ollama since content always contains the <think> wrapper).
        think_match = _re.match(
            r"<think>\s*(.*?)\s*</think>\s*(.*)", answer, _re.DOTALL
        )
        if think_match:
            q_thinking = think_match.group(1).strip()
            q_answer_only = think_match.group(2).strip()
        else:
            q_thinking = ""
            q_answer_only = answer

        query_results.append(
            {
                "text": q["text"],
                "answer": answer,
                "thinking": q_thinking,
                "answer_only": q_answer_only,
                "expected_keywords": expected,
                "keywords_found": found,
                "recall_score": recall,
                "query_elapsed_ms": round(q_elapsed, 1),
            }
        )

    total_elapsed = (time.time() - t_start) * 1000

    # ── Aggregate metrics ──
    embed_times = [c["elapsed_ms"] for c in embedding.call_log]
    chat_times = [c["elapsed_ms"] for c in llm.call_log]

    # Backend-reported token aggregation
    total_prompt_tokens = sum(c.get("prompt_tokens", 0) for c in llm.call_log)
    total_completion_tokens = sum(c.get("completion_tokens", 0) for c in llm.call_log)
    total_tokens = sum(c.get("total_tokens", 0) for c in llm.call_log)
    chat_total_ms = sum(chat_times)
    chat_total_s = chat_total_ms / 1000 if chat_total_ms > 0 else 1e-9
    n_chat_calls = len(chat_times)

    # Normalized token aggregation (shared tokenizer)
    norm_prompt_total = sum(c.get("normalized_prompt_tokens", 0) for c in llm.call_log)
    norm_completion_total = sum(
        c.get("normalized_completion_tokens", 0) for c in llm.call_log
    )
    norm_total_total = sum(c.get("normalized_total_tokens", 0) for c in llm.call_log)
    norm_thinking_total = sum(
        c.get("normalized_thinking_tokens", 0) for c in llm.call_log
    )
    norm_answer_total = sum(c.get("normalized_answer_tokens", 0) for c in llm.call_log)

    history_db = os.path.join(mem_dir, "history.db")
    long_db = os.path.join(mem_dir, "long_term.db")

    record: dict[str, Any] = {
        "backend": backend_cfg.get("name", "unknown"),
        "backend_id": backend_cfg.get("id", "unknown"),
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "status": "completed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "chat_url": backend_cfg["chat_url"],
        "chat_model": backend_cfg["chat_model"],
        "embed_url": backend_cfg["embed_url"],
        "embed_model": backend_cfg["embed_model"],
        "mem_dir": mem_dir,
        "seed_memories": scenario["seed_memories"],
        "metrics": {
            "embedding": {
                "total_calls": len(embed_times),
                "total_ms": round(sum(embed_times), 1),
                "per_call_ms": embed_times,
            },
            "chat": {
                "total_calls": n_chat_calls,
                "total_ms": round(chat_total_ms, 1),
                "per_call_ms": chat_times,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "avg_completion_tokens_per_call": (
                    round(total_completion_tokens / n_chat_calls, 1)
                    if n_chat_calls
                    else 0
                ),
                "avg_total_tokens_per_call": (
                    round(total_tokens / n_chat_calls, 1) if n_chat_calls else 0
                ),
                "completion_tokens_per_s_overall": (
                    round(total_completion_tokens / chat_total_s, 1)
                    if total_completion_tokens
                    else 0
                ),
                "total_tokens_per_s_overall": (
                    round(total_tokens / chat_total_s, 1) if total_tokens else 0
                ),
            },
            "normalized": {
                "total_prompt_tokens": norm_prompt_total,
                "total_thinking_tokens": norm_thinking_total,
                "total_answer_tokens": norm_answer_total,
                "total_completion_tokens": norm_completion_total,
                "total_tokens": norm_total_total,
                "avg_completion_tokens_per_call": (
                    round(norm_completion_total / n_chat_calls, 1)
                    if n_chat_calls
                    else 0
                ),
                "completion_tokens_per_s_overall": (
                    round(norm_completion_total / chat_total_s, 1)
                    if norm_completion_total
                    else 0
                ),
                "total_tokens_per_s_overall": (
                    round(norm_total_total / chat_total_s, 1) if norm_total_total else 0
                ),
            },
            "combined": {
                "total_ms": round(total_elapsed, 1),
            },
        },
        "queries": query_results,
        "llm_calls": llm.call_log,
        "sqlite": {
            "history_db": history_db,
            "long_term_db": long_db,
            "tasks_rows": _count_rows(history_db, "tasks"),
            "steps_rows": _count_rows(history_db, "steps"),
            "long_term_rows": _count_rows(long_db, "long_term"),
        },
    }

    await agent.close()
    return record


def _make_skipped_record(
    backend_cfg: dict[str, Any],
    scenario: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "backend": backend_cfg.get("name", "unknown"),
        "backend_id": backend_cfg.get("id", "unknown"),
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "status": "skipped",
        "status_reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "chat_url": backend_cfg.get("chat_url", ""),
        "chat_model": backend_cfg.get("chat_model", ""),
        "embed_url": backend_cfg.get("embed_url", ""),
        "embed_model": backend_cfg.get("embed_model", ""),
        "mem_dir": "",
        "seed_memories": scenario["seed_memories"],
        "metrics": None,
        "queries": [],
        "llm_calls": [],
        "sqlite": {},
    }


def _make_failed_record(
    backend_cfg: dict[str, Any],
    scenario: dict[str, Any],
    reason: str,
    mem_dir: str,
) -> dict[str, Any]:
    rec = _make_skipped_record(backend_cfg, scenario, reason)
    rec["status"] = "failed"
    rec["mem_dir"] = mem_dir
    return rec


# ── Report rendering ───────────────────────────────────────


def render_comparison_markdown(
    backend_summaries: list[dict[str, Any]],
    expected_backend_ids: list[str],
) -> str:
    """Render compare_test.md from backend summary dicts.

    Always produces output even with a single backend; notes missing
    backends when not all expected backends were run.
    """
    lines: list[str] = []
    lines.append("# 多后端对比报告")
    lines.append("")
    lines.append(f"**生成时间:** {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    lines.append("")

    present_ids = {s["backend_id"] for s in backend_summaries}
    missing = [bid for bid in expected_backend_ids if bid not in present_ids]
    if missing:
        for bid in missing:
            lines.append(f"> **注意:** 后端 `{bid}` 的结果缺失。")
        lines.append("")

    def _ordered_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_backend = {r.get("backend_id", r.get("backend", "")): r for r in records}
        ordered: list[dict[str, Any]] = []
        for bid in expected_backend_ids:
            rec = by_backend.get(bid)
            if rec is not None:
                ordered.append(rec)
        for r in records:
            rid = r.get("backend_id", r.get("backend", ""))
            if rid not in expected_backend_ids:
                ordered.append(r)
        return ordered

    def _avg_recall(record: dict[str, Any]) -> float:
        queries = record.get("queries", [])
        if not queries:
            return 0.0
        return sum(_safe_float(q.get("recall_score")) for q in queries) / len(queries)

    # Collect all scenario records from every backend summary.
    all_records: list[dict[str, Any]] = []
    for summary in backend_summaries:
        all_records.extend(summary["scenarios"])

    # Group by scenario.
    scenarios: dict[str, list[dict[str, Any]]] = {}
    for r in all_records:
        scenarios.setdefault(r["scenario_id"], []).append(r)

    ordered_summaries: list[dict[str, Any]] = []
    by_summary_id = {s["backend_id"]: s for s in backend_summaries}
    for bid in expected_backend_ids:
        found = by_summary_id.get(bid)
        if found is not None:
            ordered_summaries.append(found)
    for summary in backend_summaries:
        if summary["backend_id"] not in expected_backend_ids:
            ordered_summaries.append(summary)

    # ── Real conversations first ──
    lines.append("## Real Conversation")
    lines.append("")
    for _sid, records in scenarios.items():
        name = records[0]["scenario_name"]
        lines.append(f"### {name}")
        lines.append("")
        completed = [r for r in _ordered_records(records) if r["status"] == "completed"]
        if not completed:
            lines.append("> 这一场景没有可展示的完成结果。")
            lines.append("")
            continue

        n_queries = max(len(r["queries"]) for r in completed)
        for qi in range(n_queries):
            q_text = ""
            for r in completed:
                if qi < len(r["queries"]):
                    q_text = r["queries"][qi]["text"]
                    break
            lines.append(f"#### Q{qi + 1}. {q_text}")
            lines.append("")
            for r in completed:
                lines.append(f"**{r['backend']}**")
                if qi < len(r["queries"]):
                    q = r["queries"][qi]
                    lines.append(f"关键词命中: {q['recall_score']:.0%}")
                    lines.append("")

                    # --- Question / Thinking / Answer ---
                    lines.append("**Question**")
                    lines.append(q.get("text") or q_text)
                    lines.append("")

                    raw_answer = q.get("answer") or ""
                    thinking_text = q.get("thinking") or ""
                    answer_text = q.get("answer_only") or ""
                    if not thinking_text and not answer_text:
                        thinking_text, answer_text = _split_thinking_and_answer(
                            raw_answer
                        )

                    lines.append("**Thinking**")
                    lines.append(thinking_text if thinking_text else "—")
                    lines.append("")
                    lines.append("**Answer**")
                    lines.append(answer_text if answer_text else "—")
                else:
                    lines.append("—")
                lines.append("")

    # ── Overall comparison second ──
    lines.append("## Overall Comparison")
    lines.append("")
    lines.append("### 场景总览")
    lines.append("")
    lines.append(
        "| 场景 | 后端 | 状态 | avg 关键词命中 | 嵌入总耗时 | 对话总耗时 | 端到端耗时 |"
    )
    lines.append(
        "|------|------|------|--------------:|-----------:|-----------:|-----------:|"
    )
    for _sid, records in scenarios.items():
        name = records[0]["scenario_name"]
        for r in _ordered_records(records):
            if r["status"] != "completed" or not r.get("metrics"):
                lines.append(
                    f"| {name} | {r['backend']} | {r['status']} | 0% | — | — | — |"
                )
                continue
            m = r["metrics"]
            lines.append(
                f"| {name} | {r['backend']} | {r['status']}"
                f" | {_avg_recall(r):.0%}"
                f" | {_safe_float(m['embedding'].get('total_ms')):.1f}"
                f" | {_safe_float(m['chat'].get('total_ms')):.1f}"
                f" | {_safe_float(m['combined'].get('total_ms')):.1f} |"
            )
    lines.append("")

    # ── Overall token summary per backend (backend-reported) ──
    lines.append("### 整体 Token 汇总 (后端上报)")
    lines.append("")
    lines.append(
        "| 后端 | prompt tok | completion tok | total tok | chat ms | prompt tok/s | completion tok/s | total tok/s |"
    )
    lines.append(
        "|------|----------:|-------------:|---------:|---------:|------------:|----------------:|----------:|"
    )
    for summary in ordered_summaries:
        bid = summary["backend_name"]
        prm_tok = 0
        comp_tok = 0
        tot_tok = 0
        chat_ms = 0.0
        for r in summary["scenarios"]:
            if r["status"] == "completed" and r.get("metrics"):
                cm = r["metrics"].get("chat", {})
                prompt_tok = _safe_int(cm.get("total_prompt_tokens"))
                completion_tok = _safe_int(cm.get("total_completion_tokens"))
                total_tok = _safe_int(cm.get("total_tokens"))
                if total_tok == 0 and (prompt_tok or completion_tok):
                    total_tok = prompt_tok + completion_tok
                prm_tok += prompt_tok
                comp_tok += completion_tok
                tot_tok += total_tok
                chat_ms += _safe_float(cm.get("total_ms"))
        chat_s = chat_ms / 1000 if chat_ms > 0 else 1e-9
        prm_tps = round(prm_tok / chat_s, 1) if prm_tok else 0
        comp_tps = round(comp_tok / chat_s, 1) if comp_tok else 0
        tot_tps = round(tot_tok / chat_s, 1) if tot_tok else 0
        lines.append(
            f"| {bid} | {prm_tok} | {comp_tok} | {tot_tok} | {chat_ms:.0f} | {prm_tps} | {comp_tps} | {tot_tps} |"
        )
    lines.append("")

    # ── Overall normalized token summary (shared tokenizer) ──
    lines.append("### 整体统一 Tokenizer Token 汇总")
    lines.append("")
    lines.append("> 使用统一的确定性 tokenizer 计算，适合跨后端直接对比。")
    lines.append("")
    lines.append(
        "| 后端 | prompt tok | thinking tok | answer tok | completion tok | total tok | chat ms | completion tok/s | total tok/s |"
    )
    lines.append(
        "|------|----------:|------------:|-----------:|-------------:|---------:|---------:|----------------:|----------:|"
    )
    for summary in ordered_summaries:
        bid = summary["backend_name"]
        n_prm = 0
        n_think = 0
        n_ans = 0
        n_comp = 0
        n_tot = 0
        chat_ms = 0.0
        for r in summary["scenarios"]:
            if r["status"] == "completed" and r.get("metrics"):
                nm = r["metrics"].get("normalized", {})
                n_prm += _safe_int(nm.get("total_prompt_tokens"))
                n_think += _safe_int(nm.get("total_thinking_tokens"))
                n_ans += _safe_int(nm.get("total_answer_tokens"))
                n_comp += _safe_int(nm.get("total_completion_tokens"))
                n_tot += _safe_int(nm.get("total_tokens"))
                chat_ms += _safe_float(r["metrics"].get("chat", {}).get("total_ms"))
        chat_s = chat_ms / 1000 if chat_ms > 0 else 1e-9
        n_comp_tps = round(n_comp / chat_s, 1) if n_comp else 0
        n_tot_tps = round(n_tot / chat_s, 1) if n_tot else 0
        lines.append(
            f"| {bid} | {n_prm} | {n_think} | {n_ans} | {n_comp} | {n_tot} | {chat_ms:.0f} | {n_comp_tps} | {n_tot_tps} |"
        )
    lines.append("")

    # ── Backend-specific sections last ──
    for summary in ordered_summaries:
        backend_name = summary["backend_name"]
        lines.append(f"## {backend_name} 指标")
        lines.append("")
        lines.append(f"**对话模型:** `{summary.get('chat_model', '')}`")
        lines.append("")
        lines.append(f"**嵌入模型:** `{summary.get('embed_model', '')}`")
        lines.append("")

        lines.append("### 场景总览")
        lines.append("")
        lines.append(
            "| 场景 | 状态 | avg 关键词命中 | 嵌入总耗时 | 对话总耗时 | 端到端耗时 |"
        )
        lines.append(
            "|------|------|--------------:|-----------:|-----------:|-----------:|"
        )
        for r in summary["scenarios"]:
            if r["status"] != "completed" or not r.get("metrics"):
                lines.append(
                    f"| {r['scenario_name']} | {r['status']} | 0% | — | — | — |"
                )
                continue
            m = r["metrics"]
            lines.append(
                f"| {r['scenario_name']} | {r['status']}"
                f" | {_avg_recall(r):.0%}"
                f" | {_safe_float(m['embedding'].get('total_ms')):.1f}"
                f" | {_safe_float(m['chat'].get('total_ms')):.1f}"
                f" | {_safe_float(m['combined'].get('total_ms')):.1f} |"
            )
        lines.append("")

        completed = [r for r in summary["scenarios"] if r["status"] == "completed"]
        if completed:
            lines.append("### Token 对比 (后端上报)")
            lines.append("")
            lines.append(
                "| 场景 | chat 调用 | prompt tok | completion tok | total tok | completion tok/s | total tok/s | avg completion/call |"
            )
            lines.append(
                "|------|--------:|----------:|-------------:|---------:|----------------:|----------:|-------------------:|"
            )
            for r in completed:
                cm = r.get("metrics", {}).get("chat", {})
                prompt_tokens = _safe_int(cm.get("total_prompt_tokens"))
                completion_tokens = _safe_int(cm.get("total_completion_tokens"))
                total_tokens = _safe_int(cm.get("total_tokens"))
                if total_tokens == 0 and (prompt_tokens or completion_tokens):
                    total_tokens = prompt_tokens + completion_tokens
                lines.append(
                    f"| {r['scenario_name']}"
                    f" | {_safe_int(cm.get('total_calls'))}"
                    f" | {prompt_tokens}"
                    f" | {completion_tokens}"
                    f" | {total_tokens}"
                    f" | {_safe_float(cm.get('completion_tokens_per_s_overall')):.1f}"
                    f" | {_safe_float(cm.get('total_tokens_per_s_overall')):.1f}"
                    f" | {_safe_float(cm.get('avg_completion_tokens_per_call')):.1f}"
                    f" |"
                )
            lines.append("")

            lines.append("### 统一 Tokenizer Token 对比")
            lines.append("")
            lines.append("> 使用统一的确定性 tokenizer 计算，适合跨后端直接对比。")
            lines.append("")
            lines.append(
                "| 场景 | prompt tok | thinking tok | answer tok | completion tok | total tok | completion tok/s | total tok/s | avg completion/call |"
            )
            lines.append(
                "|------|----------:|------------:|-----------:|-------------:|---------:|----------------:|----------:|-------------------:|"
            )
            for r in completed:
                nm = r.get("metrics", {}).get("normalized", {})
                lines.append(
                    f"| {r['scenario_name']}"
                    f" | {_safe_int(nm.get('total_prompt_tokens'))}"
                    f" | {_safe_int(nm.get('total_thinking_tokens'))}"
                    f" | {_safe_int(nm.get('total_answer_tokens'))}"
                    f" | {_safe_int(nm.get('total_completion_tokens'))}"
                    f" | {_safe_int(nm.get('total_tokens'))}"
                    f" | {_safe_float(nm.get('completion_tokens_per_s_overall')):.1f}"
                    f" | {_safe_float(nm.get('total_tokens_per_s_overall')):.1f}"
                    f" | {_safe_float(nm.get('avg_completion_tokens_per_call')):.1f}"
                    f" |"
                )
            lines.append("")

    return "\n".join(lines)


# ── Apply environment overrides to backend config ──────────


def _apply_env_overrides(
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Let SMOKE_* env vars override backend defaults."""
    overrides = {
        "SMOKE_CHAT_URL": "chat_url",
        "SMOKE_CHAT_MODEL": "chat_model",
        "SMOKE_EMBED_URL": "embed_url",
        "SMOKE_EMBED_MODEL": "embed_model",
    }
    cfg = dict(cfg)
    for env_key, cfg_key in overrides.items():
        val = os.environ.get(env_key)
        if val:
            cfg[cfg_key] = val
    return cfg


# ── Main orchestrator ──────────────────────────────────────


async def run_all(
    backend_ids: list[str],
    scenario_ids: list[str] | None,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Run selected backends x scenarios and write reports."""
    backends = load_backends()
    scenarios = load_scenarios()

    if scenario_ids:
        scenarios = [s for s in scenarios if s["id"] in scenario_ids]

    selected: dict[str, dict[str, Any]] = {}
    for bid in backend_ids:
        if bid in backends:
            cfg = _apply_env_overrides(backends[bid])
            cfg["id"] = bid
            selected[bid] = cfg
        else:
            log.warning("未知后端: %s (可用: %s)", bid, ", ".join(backends))

    all_records: list[dict[str, Any]] = []

    for bid, bcfg in selected.items():
        print(f"\n{'=' * 60}")
        print(f"[smoke] 后端: {bcfg.get('name', bid)}")
        print(f"[smoke] 对话: {bcfg['chat_model']}" f" @ {bcfg['chat_url']}")
        print(f"[smoke] 嵌入: {bcfg['embed_model']}" f" @ {bcfg['embed_url']}")

        ok, msg = check_backend_health(bcfg)
        if not ok:
            print(f"[smoke] 后端不可用: {msg}")
            for sc in scenarios:
                all_records.append(_make_skipped_record(bcfg, sc, msg))
            continue

        print(f"[smoke] 健康检查通过: {msg}")

        n_scenarios = len(scenarios)
        for si, sc in enumerate(scenarios, 1):
            print(f"\n[smoke] 场景 [{si}/{n_scenarios}]: {sc['name']} ({sc['id']})")
            print(
                f"[smoke]   seeds={len(sc['seed_memories'])}  queries={len(sc['queries'])}"
            )
            mem_dir = tempfile.mkdtemp(prefix=f"melo_smoke_{bid}_{sc['id']}_")
            try:
                record = await run_scenario(bcfg, sc, mem_dir)
                all_records.append(record)
                combined = record["metrics"]["combined"]["total_ms"]
                n_q = len(record["queries"])
                avg_recall = (
                    sum(q["recall_score"] for q in record["queries"]) / n_q
                    if n_q
                    else 0
                )
                print(
                    f"[smoke] 完成 [{si}/{n_scenarios}] — "
                    f"{combined:.0f}ms  "
                    f"recall={avg_recall:.0%}"
                )
            except Exception:
                log.exception("场景执行失败")
                all_records.append(
                    _make_failed_record(bcfg, sc, "exception (see log)", mem_dir)
                )
                print(f"[smoke] 失败 [{si}/{n_scenarios}]")

    # ── Write reports ──
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group records by backend and write one summary JSON per backend.
    by_backend: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        by_backend.setdefault(record["backend_id"], []).append(record)

    backend_summaries: list[dict[str, Any]] = []
    for bid, records in by_backend.items():
        first = records[0]
        summary: dict[str, Any] = {
            "backend_id": bid,
            "backend_name": first["backend"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "chat_url": first["chat_url"],
            "chat_model": first["chat_model"],
            "embed_url": first["embed_url"],
            "embed_model": first["embed_model"],
            "scenarios": records,
        }
        backend_summaries.append(summary)
        json_path = out_dir / f"{bid}_test.json"
        json_path.write_text(
            json.dumps(summary, indent=2, default=str, ensure_ascii=False),
            "utf-8",
        )

    # Reload all backend summary JSONs from disk so sequential runs
    # (e.g. ollama first, mlc later) produce one combined compare report.
    known_backends = ["ollama", "mlc"]
    disk_summaries: list[dict[str, Any]] = []
    for bid in known_backends:
        json_path = out_dir / f"{bid}_test.json"
        if json_path.exists():
            disk_summaries.append(json.loads(json_path.read_text("utf-8")))

    comp_md = render_comparison_markdown(disk_summaries, known_backends)
    (out_dir / "compare_test.md").write_text(comp_md, "utf-8")
    print(f"\n[smoke] 对比报告: {out_dir / 'compare_test.md'}")

    return all_records


# ── CLI ─────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="数据驱动的核心循环基准/演示框架 (外部后端)",
    )
    p.add_argument(
        "--backends",
        default=os.environ.get("SMOKE_BACKENDS", "ollama"),
        help=(
            "逗号分隔的后端 ID 列表，或 'all'"
            " (默认: ollama; 环境变量: SMOKE_BACKENDS)"
        ),
    )
    p.add_argument(
        "--scenarios",
        default=None,
        help="逗号分隔的场景 ID 列表 (默认: 全部)",
    )
    p.add_argument(
        "--out-dir",
        default=str(_DEFAULT_OUT_DIR),
        help=f"输出目录 (默认: {_DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="跳过 benchmark 执行，直接用 out-dir 里已有的 JSON 生成对比报告",
    )
    return p.parse_args()


def _report_only(out_dir: Path, backend_ids: list[str]) -> None:
    """Regenerate compare_test.md from existing JSON files without re-running."""
    known_backends = ["ollama", "mlc"]
    disk_summaries: list[dict[str, Any]] = []
    for bid in known_backends:
        json_path = out_dir / f"{bid}_test.json"
        if json_path.exists():
            disk_summaries.append(json.loads(json_path.read_text("utf-8")))
            print(f"[smoke] 加载已有结果: {json_path}")
        else:
            print(f"[smoke] 未找到: {json_path}")

    if not disk_summaries:
        print("[smoke] 没有可用的 JSON 结果文件，无法生成报告。")
        return

    comp_md = render_comparison_markdown(disk_summaries, known_backends)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare_test.md").write_text(comp_md, "utf-8")
    print(f"\n[smoke] 对比报告已生成: {out_dir / 'compare_test.md'}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args()

    backends_cfg = load_backends()
    if args.backends == "all":
        backend_ids = list(backends_cfg.keys())
    else:
        backend_ids = [b.strip() for b in args.backends.split(",")]

    scenario_ids: list[str] | None = None
    if args.scenarios:
        scenario_ids = [s.strip() for s in args.scenarios.split(",")]

    out_dir = Path(args.out_dir)

    print("[smoke] 数据驱动核心循环基准框架")
    print(f"[smoke] 后端: {', '.join(backend_ids)}")
    if scenario_ids:
        print(f"[smoke] 场景: {', '.join(scenario_ids)}")
    else:
        print("[smoke] 场景: 全部")
    print(f"[smoke] 输出: {out_dir}")

    if args.report_only:
        _report_only(out_dir, backend_ids)
        return

    records = asyncio.run(run_all(backend_ids, scenario_ids, out_dir))

    completed = [r for r in records if r["status"] == "completed"]
    skipped = [r for r in records if r["status"] == "skipped"]
    failed = [r for r in records if r["status"] == "failed"]

    print(f"\n{'=' * 60}")
    print(
        f"[smoke] 完成: {len(completed)},"
        f" 跳过: {len(skipped)},"
        f" 失败: {len(failed)}"
    )
    print(f"[smoke] 报告输出至: {out_dir}")
    print("[smoke] 结束。")


if __name__ == "__main__":
    main()
