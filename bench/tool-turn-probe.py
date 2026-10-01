#!/usr/bin/env python3
"""Chat-template probe for tool turns (multi-call histories, tool results out of order).

Sends, with thinking off and temperature 0:
  1. a plain tool-call request (function definitions, expects a tool_call or an answer);
  2. a multi-turn history with TWO assistant tool calls followed by their tool results in
     REVERSED order (the template's reordering branch), asking for a summary;
  3. the same history with tool results in order;
  4. the same history with the assistant tool-call turn carrying null content (the `content is not none` guard).
Passes when every request returns HTTP 200, finish_reason stop, non-empty content, and the
reversed-order answer mentions both tool results.

  CTK='{"enable_thinking":false}' python3 tool-turn-probe.py --api http://127.0.0.1:8015 --model mimo-v2.6-flash --keys-file /path/to/api-keys
"""
import argparse, json, os, sys, urllib.request, urllib.error

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "get_time", "description": "Local time for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

def call(api, model, key, messages, tools=None, max_tokens=200):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": json.loads(os.environ["CTK"]) if os.environ.get("CTK") else {"reasoning_effort": "low"}}
    if tools: body["tools"] = tools
    req = urllib.request.Request(api.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:300].decode(errors="replace")}

def history(reverse, null_first=False):
    calls = [{"id": "call_w1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Seoul"})}},
             {"id": "call_t1", "type": "function", "function": {"name": "get_time", "arguments": json.dumps({"city": "Seoul"})}}]
    results = [{"role": "tool", "tool_call_id": "call_w1", "content": "Seoul: 24 C, light rain"},
               {"role": "tool", "tool_call_id": "call_t1", "content": "Seoul local time: 14:05"}]
    if null_first:
        # exercises the template's `content is not none` guard: assistant tool-call turn with
        # null content (vLLM rejects a null *tool* message content at request validation, so
        # that variant cannot reach the template through the OpenAI endpoint)
        pass
    if reverse: results = results[::-1]
    return [{"role": "user", "content": "What is the weather and the local time in Seoul? Use the tools."},
            {"role": "assistant", "content": None if null_first else "", "tool_calls": calls}] + results + \
           [{"role": "user", "content": "Summarize both tool results in one sentence."}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True); ap.add_argument("--model", required=True); ap.add_argument("--keys-file", required=True)
    a = ap.parse_args()
    key = open(a.keys_file).readline().strip()
    cases = [("tool_call", [{"role": "user", "content": "What's the weather in Seoul? Use a tool."}], TOOLS, None),
             ("results_reversed", history(True), TOOLS, ("24", "14:05")),
             ("results_in_order", history(False), TOOLS, ("24", "14:05")),
             ("assistant_null_content", history(False, null_first=True), TOOLS, ("24", "14:05"))]
    failed = 0
    for name, msgs, tools, expect in cases:
        code, d = call(a.api, a.model, key, msgs, tools)
        ok = code == 200 and d.get("choices")
        content = ""; finish = None
        if ok:
            ch = d["choices"][0]; finish = ch.get("finish_reason"); m = ch.get("message", {})
            content = (m.get("content") or "") ; tc = m.get("tool_calls")
            if name == "tool_call":
                ok = bool(tc) or bool(content.strip())
            else:
                ok = finish == "stop" and bool(content.strip()) and all(e in content for e in (expect or ()))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: HTTP {code} finish={finish} -> {json.dumps(content[:160] if content else d.get('error') or (d.get('choices') or [{}])[0].get('message', {}).get('tool_calls'))}")
        failed += 0 if ok else 1
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
