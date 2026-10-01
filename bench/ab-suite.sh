#!/usr/bin/env bash
# MiMo-V2.6-Flash-MOPD benchmark suite: the SAME cells against any OpenAI-compatible endpoint. Every number in the
# README's results table came from these cells (run on rank 0's host, client pinned to cores 0-4,10-14).
#
#   ab-suite.sh <label> <base-url> <model> <keyfile> [cells]
#   cells (comma list, default all): gates,agent,decode,prose,ladder,shared,prefill
#   extra cells (not in the default list): longctx (multi-turn ~100K sessions, LONG_CONC=1,4 LONG_PREFIX=90000
#   LONG_ROTATE=1), garbage (vllm#46669 corrupted-token probe, GARBAGE_C=32 GARBAGE_ROUNDS=2),
#   ppl (teacher-forced NLL on PPL_FILES, a fixed corpus of your choice), profile (decode-step torch traces)
#
# Thinking OFF everywhere (chat_template_kwargs.enable_thinking=false) so decode cells measure answer tokens; the
# agent-loop probe runs with the model default (thinking on) as users will.
# Needs: a RigMark checkout at RM (we ran othexmr/rigmark@40fabcaf, a fork of alexellis/rigmark) and the helper
# scripts beside this file (BIN). Results: $OUT_ROOT/<label>/ (default $HOME/mimo26-bench/<label>/).
set -uo pipefail
LABEL="${1:?label}"; URL="${2:?base url}"; MODEL="${3:?model}"; KEYFILE="${4:?keyfile}"
CELLS=",${5:-gates,agent,decode,prose,ladder,shared,prefill},"
HERE=$(cd "$(dirname "$0")" && pwd)
BIN="${BIN:-$HERE}"; RM="${RM:-$HOME/rigmark-othexmr}"
OUT="${OUT_ROOT:-$HOME/mimo26-bench}/$LABEL"; mkdir -p "$OUT"
export KEYFILE CTK='{"enable_thinking":false}'
XBODY='{"chat_template_kwargs":{"enable_thinking":false}}'
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/suite.log"; }
has() { [[ "$CELLS" == *",$1,"* ]]; }
export RIGMARK_KEY="$(head -1 "$KEYFILE")"
log "=== $LABEL: $URL model=$MODEL cells=$CELLS ==="
curl -sf --max-time 10 "$URL/health" >/dev/null || { log "FATAL: $URL/health"; exit 1; }
curl -s --max-time 10 -H "Authorization: Bearer $RIGMARK_KEY" "$URL/v1/models" > "$OUT/models.json"
cat > "$OUT/metadata.json" <<META
{"context_limit":${CTX_LIMIT:-524288},"kv_cache_dtype":"${KV_DTYPE:-unknown}","model_revision":"XiaomiMiMo/MiMo-V2.6-Flash-MOPD@2479e2d0","quantisation":"${QUANT:-MXFP4 experts}","serving_engine":"${ENGINE:-unknown}","hardware":"${HW:-unknown}","topology":"${TOPO:-unknown}","tensor_parallel_size":${TP:-1},"model":"$MODEL","comparison_note":"$LABEL","competing_traffic":"none (lane drained)"}
META

if has gates; then
  canary=$(curl -s --max-time 300 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -H "Authorization: Bearer $RIGMARK_KEY" \
    -d "{\"model\":\"$MODEL\",\"temperature\":0,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with just the number.\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}")
  echo "$canary" > "$OUT/canary.json"
  printf '%s' "$canary" | python3 -c 'import json,sys; c=json.load(sys.stdin)["choices"][0]["message"].get("content") or ""; sys.exit(0 if "391" in c else 1)' \
    && log "gate canary: PASS" || log "gate canary: FAIL $(head -c 300 "$OUT/canary.json")"
  python3 "$BIN/tool-turn-probe.py" --api "$URL" --model "$MODEL" --keys-file "$KEYFILE" > "$OUT/tool-turn.log" 2>&1
  log "gate tool-turn: $(tail -2 "$OUT/tool-turn.log" | tr '\n' ' ' | head -c 300)"
fi
if has agent; then
  CTK= python3 "$BIN/agent-loop-probe.py" --api "$URL" --model "$MODEL" --keys-file "$KEYFILE" --episodes 12 --parallel 4 \
    --out "$OUT/agent-loop.json" > "$OUT/agent-loop.log" 2>&1
  log "agent-loop (thinking default, T=1.0, 12 episodes): rc=$? $(tail -1 "$OUT/agent-loop.log")"
fi
COMMON=(--base-url "$URL" --model "$MODEL" --metadata "$OUT/metadata.json" --comparison-id "$LABEL" --runs 2
        --concurrency-runs 2 --timeout 1800 --extra-body "$XBODY" --api-key-env RIGMARK_KEY)
if has decode; then   # single-stream code/prose/structured + short-code concurrency C1..C16 (no /tokenize needed)
  (cd "$RM" && ./rigmark run "${COMMON[@]}" --label "$LABEL-code" --skip-prefill --concurrency "${CONC:-1,2,4,8,12,16}" \
    --output "$OUT/code.json" > "$OUT/code.log" 2>&1); log "rigmark code rc=$?"
fi
if has prose; then    # short-prose concurrency C1..C16; staggered arrivals only where the engine has /tokenize
  st=(); [ -n "${STAGGERED:-}" ] && st=(--staggered "$STAGGERED" --staggered-runs 2 --staggered-depth 32768 \
    --staggered-incumbent-tokens 1024 --staggered-arrival-tokens 256 --staggered-delay 1.0 --staggered-workload prose)
  (cd "$RM" && ./rigmark run "${COMMON[@]}" --label "$LABEL-prose" --skip-prefill --concurrency-workload prose \
    --concurrency "${CONC:-1,2,4,8,12,16}" "${st[@]}" --output "$OUT/prose.json" > "$OUT/prose.log" 2>&1); log "rigmark prose rc=$?"
fi
if has ladder; then   # the concurrency question: sampled prose at T=0.7, 400-token answers, C1..C32
  for rep in 1 2; do
    API="$URL" MODEL="$MODEL" RUNGS="${RUNGS:-1,2,4,8,16,24,32}" MAXTOK=400 PROMPT_WORDS=120 \
      OUT="$OUT/ladder-$rep.json" python3 "$BIN/conc-ladder.py" > "$OUT/ladder-$rep.log" 2>&1
    log "ladder rep $rep:"; tail -8 "$OUT/ladder-$rep.log" | tee -a "$OUT/suite.log"
  done
fi
if has shared; then   # agent-shaped: ~15K-token prompts, cold then shared-prefix rounds, c4 and c8
  for c in 4 8; do
    API="$URL" MODEL="$MODEL" MODE=cold C=$c ROUNDS=1 OUT="$OUT/shaped-cold-c$c.json" python3 "$BIN/shaped.py" > "$OUT/shaped-cold-c$c.log" 2>&1
    log "shaped cold c$c: $(tail -1 "$OUT/shaped-cold-c$c.log" | head -c 300)"
    API="$URL" MODEL="$MODEL" MODE=shared C=$c ROUNDS=3 OUT="$OUT/shaped-shared-c$c.json" python3 "$BIN/shaped.py" > "$OUT/shaped-shared-c$c.log" 2>&1
    log "shaped shared c$c:"; cut -c1-300 "$OUT/shaped-shared-c$c.log" | tee -a "$OUT/suite.log"
  done
fi
if has longctx; then  # the agentic production shape (median prompt ~100K, ~96 % prefix-cache hits, ~750-token answers,
  # mostly 1-2 concurrent): C sessions with their own ~LONG_PREFIX-token system prompt, 3 turns each.
  # Turn 0 is a cold prefill; turns 1-2 extend the conversation (hot prefix) and decode at ~100K context.
  lc="${LONG_CONC:-1,4}"
  for c in ${lc//,/ }; do
    API="$URL" MODEL="$MODEL" MODE=multiturn METRICS=1 ROTATE="${LONG_ROTATE:-0}" C=$c ROUNDS=3 PREFIX_TOKENS="${LONG_PREFIX:-90000}" TURN_TOKENS=1500 \
      MAXTOK=600 OUT="$OUT/longctx-c$c.json" python3 "$BIN/shaped.py" > "$OUT/longctx-c$c.log" 2>&1
    log "longctx c$c:"; cut -c1-260 "$OUT/longctx-c$c.log" | tee -a "$OUT/suite.log"
  done
fi
if has garbage; then  # vllm#46669 probe: corrupted tokens under async scheduling + MTP at concurrency
  API="$URL" MODEL="$MODEL" C="${GARBAGE_C:-32}" ROUNDS="${GARBAGE_ROUNDS:-2}" MAXTOK=1200 OUT="$OUT/garbage.json" \
    python3 "$BIN/garbage-probe.py" > "$OUT/garbage.log" 2>&1
  log "garbage rc=$? $(tail -1 "$OUT/garbage.log" | head -c 400)"
fi
if has ppl; then      # teacher-forced prompt logprobs on a fixed corpus; compare two arms with `ppl-probe.py compare`
  if [ -z "${PPL_FILES:-}" ]; then
    log "ppl: skipped (set PPL_FILES to a space-separated list of corpus files, the same for every arm)"
  else
    API="$URL" MODEL="$MODEL" FILES="$PPL_FILES" WINDOW=3000 NWIN=24 OUT="$OUT/ppl.json" \
      python3 "$BIN/ppl-probe.py" > "$OUT/ppl.log" 2>&1
    log "ppl rc=$? $(tail -1 "$OUT/ppl.log" | head -c 300)"
  fi
fi
if has prefill; then  # identical fresh prompts on every engine
  API="$URL" MODEL="$MODEL" SIZES="${SIZES:-9800:2,39000:2,78000:2,157000:1,300000:1}" OUT="$OUT/prefill.json" \
    python3 "$BIN/prefill.py" > "$OUT/prefill.log" 2>&1
  log "prefill:"; tee -a "$OUT/suite.log" < "$OUT/prefill.log"
fi
if has profile; then  # torch traces of decode steps (engine needs --profiler-config.profiler=torch); last, unmeasured
  API="$URL" MODEL="$MODEL" CTXS="${PROFILE_CTXS:-2000,60000}" STEPS="${PROFILE_STEPS:-120}" \
    python3 "$BIN/profile-decode.py" > "$OUT/profile.log" 2>&1
  log "profile rc=$? $(tr '\n' ' ' < "$OUT/profile.log" | head -c 400)"
fi
curl -s --max-time 30 "$URL/metrics" > "$OUT/metrics-end.txt" 2>/dev/null
log "=== $LABEL done ==="
