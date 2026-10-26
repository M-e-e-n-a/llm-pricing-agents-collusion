#!/usr/bin/env bash
# One-shot runner for prereg-v4 section 22 (threat versions) on gpt-oss-20b, from a Mac or Linux terminal.
# Sets up Python, logs in to Modal (browser, once), starts the vLLM server, waits for it, runs the experiment,
# stops the server, and prints where the results are. Safe to run again: each step skips what is already done,
# and pilot.fork resumes a half-finished run.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/fork_runs/threat_gptoss20b
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "1/6 Python 3.12 environment"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv --python 3.12
source .venv/bin/activate
uv pip install -q modal numpy pydantic openai

say "2/6 Modal login (a browser window opens the first time)"
modal profile current >/dev/null 2>&1 || modal setup

say "3/6 Server key"
KEYFILE=.vllm_key
[ -f "$KEYFILE" ] || python -c "import secrets;print(secrets.token_hex(16))" > "$KEYFILE"
export VLLM_API_KEY=$(cat "$KEYFILE")
modal secret create vllm-api-key VLLM_API_KEY="$VLLM_API_KEY" --force >/dev/null

say "4/6 Download gpt-oss-20b into Modal and deploy the server"
modal run pilot/modal_gptoss20b.py::download
URL=$(modal deploy pilot/modal_gptoss20b.py 2>&1 | tee /dev/stderr | grep -oE 'https://[a-zA-Z0-9.-]+\.modal\.run' | head -1)
[ -n "$URL" ] || { echo "Could not find the server URL in the deploy output above."; exit 1; }
echo "Server: $URL"

say "5/6 Waiting for the server to answer (first start takes a few minutes)"
for i in $(seq 1 90); do
  if curl -sf -m 20 -H "Authorization: Bearer $VLLM_API_KEY" "$URL/v1/models" | grep -q gpt-oss-20b; then
    echo "Server is up."; break
  fi
  [ "$i" = 90 ] && { echo "Server did not come up in 30 minutes."; exit 1; }
  sleep 20
done

say "6/6 Running the experiment (about 15-25 minutes)"
python -m pilot.fork --backend vllm-server --api-model openai/gpt-oss-20b --base-url "$URL/v1" \
  --agents plain --sessions 20 --workers 40 --repeat-offer --first-seed 100 \
  --texts pilot/fork_texts_threat.json --versions R,O,O_punish,O_threat,O_threat_punish \
  --out "$OUT" || echo "The run stopped early; whatever finished is in $OUT (run this script again to resume)."

modal app stop gptoss20b-server || true
cp "$OUT/results.jsonl" "$OUT/calls.jsonl" "$HOME/Desktop/" 2>/dev/null || true
say "Done. results.jsonl and calls.jsonl are on your Desktop. Attach both in the chat."
open "$HOME/Desktop" 2>/dev/null || true
