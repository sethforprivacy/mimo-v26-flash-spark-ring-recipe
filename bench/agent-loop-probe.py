#!/usr/bin/env python3
"""Multi-turn tool-use probe: does the model finish a small agent task without looping?

MiMo-V2.6-Flash-MOPD's stated fix over Flash-RL is tool-call repetition (the same or a
near-identical call emitted again and again). This drives a scripted file-system task
against any OpenAI-compatible endpoint, executes the model's tool calls against a fake
in-memory tree, and records per episode:

  turns, tool calls, duplicate calls (same name + same arguments seen before),
  max calls in one assistant turn, whether it finished (a final answer containing the
  expected token), parse failures (tool-call markup left in content), and reasoning
  leaks (thinking markup such as "</think>" in content).

Stdlib only.  Example (on rank 0's host):
  API=http://127.0.0.1:8025 MODEL=mimo-v2.6-flash KEYFILE=/path/to/api-keys EPISODES=12 \
    python3 agent-loop-probe.py --out agent-loop.json
Exit 0 iff every episode finished with no duplicate call and no leak. Written 2026-09-29.
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.error
import urllib.request

TREE = {
    "README.md": "Project Falcon. Config lives under conf/. See conf/app.toml for the port.",
    "conf/app.toml": "[server]\nhost = \"0.0.0.0\"\n# the real port is in conf/ports.env\n",
    "conf/ports.env": "HTTP_PORT=18443\nMETRICS_PORT=19090\n",
    "src/main.py": "import os\nPORT = int(os.environ['HTTP_PORT'])\n",
    "src/util.py": "def noop():\n    return None\n",
}
EXPECT = "18443"
TOOLS = [
    {"type": "function", "function": {"name": "list_dir", "description": "List files under a directory ('' = root).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file's full contents.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search all files for a substring.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
]
TASK = ("You are working in a repository. Find which TCP port the HTTP server listens on. Use the tools to "
        "inspect the files; do not guess. When you know it, reply with one sentence that contains the port number.")
LEAK_MARKS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "<parameter=")


def run_tool(name, args):
    if name == "list_dir":
        p = (args.get("path") or "").strip("/")
        names = sorted({k[len(p) + 1:].split("/")[0] if p else k.split("/")[0]
                        for k in TREE if not p or k.startswith(p + "/")})
        return "\n".join(names) or "(empty or no such directory)"
    if name == "read_file":
        return TREE.get((args.get("path") or "").strip("/"), "error: no such file")
    if name == "grep":
        pat = args.get("pattern") or ""
        hits = [f"{k}:{i + 1}: {line}" for k, v in TREE.items() for i, line in enumerate(v.splitlines()) if pat and pat in line]
        return "\n".join(hits) or "(no matches)"
    return f"error: unknown tool {name}"


def post(api, key, body, timeout):
    req = urllib.request.Request(api.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:400].decode(errors="replace")}


def episode(a, key, idx):
    msgs = [{"role": "user", "content": TASK}]
    seen, rec = set(), {"episode": idx, "turns": 0, "calls": 0, "duplicates": 0, "max_calls_per_turn": 0,
                        "finished": False, "parse_fail": False, "leak": False, "error": None, "answer": None}
    t0 = time.time()
    for _ in range(a.max_turns):
        body = {"model": a.model, "messages": msgs, "tools": TOOLS, "max_tokens": a.max_tokens,
                "temperature": a.temperature, "top_p": 0.95}
        if a.ctk:
            body["chat_template_kwargs"] = json.loads(a.ctk)
        status, data = post(a.api, key, body, a.timeout)
        rec["turns"] += 1
        if status != 200:
            rec["error"] = f"HTTP {status}: {str(data)[:300]}"
            break
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        if any(m in content for m in LEAK_MARKS):
            rec["leak" if ("think>" in content) else "parse_fail"] = True
        rec["calls"] += len(calls)
        rec["max_calls_per_turn"] = max(rec["max_calls_per_turn"], len(calls))
        if not calls:
            rec["answer"] = content[-400:]
            rec["finished"] = EXPECT in content
            break
        msgs.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        for c in calls:
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args, rec["parse_fail"] = {}, True
            sig = (fn, json.dumps(args, sort_keys=True))
            if sig in seen:
                rec["duplicates"] += 1
            seen.add(sig)
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": run_tool(fn, args)})
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("API", "http://127.0.0.1:8025"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "mimo-v2.6-flash"))
    ap.add_argument("--keys-file", default=os.environ.get("KEYFILE"))
    ap.add_argument("--episodes", type=int, default=int(os.environ.get("EPISODES", "12")))
    ap.add_argument("--parallel", type=int, default=int(os.environ.get("PARALLEL", "4")))
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ctk", default=os.environ.get("CTK", ""), help="chat_template_kwargs JSON; empty = model default")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="agent-loop.json")
    a = ap.parse_args()
    key = open(os.path.expanduser(a.keys_file)).readline().strip() if a.keys_file else "none"
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        rows = list(ex.map(lambda i: episode(a, key, i), range(a.episodes)))
    summary = {k: sum(r[k] for r in rows) for k in ("finished", "duplicates", "calls", "turns")}
    summary.update(episodes=len(rows), leaks=sum(r["leak"] for r in rows), parse_fail=sum(r["parse_fail"] for r in rows),
                   errors=sum(bool(r["error"]) for r in rows), max_calls_per_turn=max(r["max_calls_per_turn"] for r in rows),
                   model=a.model, temperature=a.temperature, ctk=a.ctk or "default")
    json.dump({"summary": summary, "episodes": rows}, open(os.path.expanduser(a.out), "w"), indent=1)
    print(json.dumps(summary))
    ok = summary["finished"] == len(rows) and not (summary["duplicates"] or summary["leaks"] or summary["parse_fail"] or summary["errors"])
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
