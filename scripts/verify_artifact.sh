#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ARTIFACT_ROOT}"

fail() {
  echo "[fail] $*" >&2
  exit 1
}

echo "[check] ASCII-only paths and text"
if find . -print | LC_ALL=C grep -n '[^ -~]' >/dev/null; then
  find . -print | LC_ALL=C grep -n '[^ -~]' >&2
  fail "non-ASCII path detected"
fi
if rg -n --pcre2 '[^\x00-\x7F]' . >/dev/null; then
  rg -n --pcre2 '[^\x00-\x7F]' . >&2
  fail "non-ASCII text detected"
fi

echo "[check] excluded runtime files"
if find . -type f \( -name '*.env' -o -name '*.pyc' -o -name '*.traj' -o -name 'hook_events.jsonl' \) -print | grep -q .; then
  find . -type f \( -name '*.env' -o -name '*.pyc' -o -name '*.traj' -o -name 'hook_events.jsonl' \) -print >&2
  fail "runtime file detected"
fi
if find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.venv' -o -name 'venv' \) -print | grep -q .; then
  find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.venv' -o -name 'venv' \) -print >&2
  fail "cache or virtual environment detected"
fi

echo "[check] obvious hardcoded secrets"
if rg -n --pcre2 '(?i)(sk-[a-z0-9_-]{16,}|bearer[[:space:]]+[a-z0-9._-]{16,})' . >/dev/null; then
  rg -n --pcre2 '(?i)(sk-[a-z0-9_-]{16,}|bearer[[:space:]]+[a-z0-9._-]{16,})' . >&2
  fail "possible hardcoded secret detected"
fi

echo "[check] shell syntax"
while IFS= read -r -d '' script; do
  bash -n "${script}"
done < <(find . -type f -name '*.sh' -print0)

echo "[check] Python syntax"
PYCACHE_ROOT="$(mktemp -d)"
trap 'rm -rf "${PYCACHE_ROOT}"' EXIT
export PYTHONPYCACHEPREFIX="${PYCACHE_ROOT}"
python3 -m compileall -q framework experiments scripts

echo "[check] JSON syntax"
while IFS= read -r -d '' payload; do
  python3 -m json.tool "${payload}" >/dev/null
done < <(find experiments -type f -name '*.json' -print0)

echo "[check] retained core evidence counts"
CORE_SUMMARIES="$(find experiments/core_f3/evidence/per_attempt -type f -name '*.summary.json' | wc -l)"
CORE_REPORTS="$(find experiments/core_f3/evidence/per_attempt -type f -name '*.official_eval.json' | wc -l)"
[[ "${CORE_SUMMARIES}" == "180" ]] || fail "expected 180 core summaries, found ${CORE_SUMMARIES}"
[[ "${CORE_REPORTS}" == "180" ]] || fail "expected 180 core official-eval reports, found ${CORE_REPORTS}"

echo "[ok] artifact verification passed"

