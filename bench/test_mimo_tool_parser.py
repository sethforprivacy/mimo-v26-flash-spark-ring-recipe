#!/usr/bin/env python3
"""Offline check of --tool-call-parser mimo (2026-09-30, our port of vllm#58019 in overlays/vllm-patches-tools). Runs inside the lane image,
CPU only:  docker run --rm --entrypoint python3 [-v overlay mounts] -v <model>:/models/m:ro -v <this>:/t.py IMAGE /t.py

Feeds the registered "mimo" tool parser the compact XML MiMo emits and prints the parsed arguments. With the
overlay a string value keeps its leading/trailing newline and typed parameters are still coerced by the tool
schema. Exit 0 iff the verbatim expectations hold, so running it against the stock image shows the base bug."""
import json, sys
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers import ToolParserManager

tok = AutoTokenizer.from_pretrained("/models/m", trust_remote_code=True)
cls = ToolParserManager.get_tool_parser("mimo")
print("parser:", cls.__module__, cls.__name__, "structural_tag_model:", getattr(cls, "structural_tag_model", None))
tools = [{"type": "function", "function": {"name": "write_file", "description": "write a file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                     "line": {"type": "integer"}, "append": {"type": "boolean"}},
    "required": ["path", "content"]}}}]
req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], tools=tools, tool_choice="auto")
cases = [("\ndef f():\n    return 1\n", "\ndef f():\n    return 1\n"), ("plain", "plain"), ("a\nb", "a\nb"),
         ("\n", "\n"), ("", "")]
ok = True
for raw, want in cases:
    out = (f"<tool_call><function=write_file><parameter=path>/tmp/x.py</parameter><parameter=content>{raw}</parameter>"
           "<parameter=line>42</parameter><parameter=append>true</parameter></function></tool_call>")
    p = cls(tok, tools=None) if "tools" in cls.__init__.__code__.co_varnames else cls(tok)
    res = p.extract_tool_calls(out, req)
    args = json.loads(res.tool_calls[0].function.arguments)
    good = args.get("content") == want and args.get("line") == 42 and args.get("append") is True
    ok &= good
    print(("PASS" if good else "FAIL"), json.dumps({"raw": raw, "content": args.get("content"), "line": args.get("line"),
                                                    "append": args.get("append")}))
sys.exit(0 if ok else 1)
