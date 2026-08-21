#!/bin/bash
# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# backtalk installer — environment, engines, models, and a gate that
# PROVES the voice loop works before it claims success.
# Safe to re-run; every step skips what's already done.
#
# MarQed fork changes (see MARQED.md):
#  - espeak-ng: checks the DATA files too, not just the library
#  - deterministic install from requirements.lock.txt
#  - CUDA 12 runtime alongside torch's CUDA 13, for faster-whisper
#  - a real verification gate: synthesize speech, transcribe it back,
#    and exit non-zero if the round trip fails
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

echo "== backtalk install =="

# --- uv (the Python environment manager) ---
if ! command -v uv >/dev/null 2>&1; then
  echo "-- uv not found. It's the fast Python manager this uses."
  read -r -p "   Install it now? [Y/n] " a
  if [ "$a" = "n" ] || [ "$a" = "N" ]; then
    echo "   Install uv yourself (https://docs.astral.sh/uv/) and re-run."
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- espeak-ng: the LIBRARY *and* its DATA ---
# Upstream only looked for the library. A box can carry libespeak-ng.so.1
# as some other package's dependency while espeak-ng-data is absent —
# the check then passes, and kokoro later calls exit() mid-install
# looking for a phontab that isn't there. Check both.
have_espeak_lib() {
  command -v espeak-ng >/dev/null 2>&1 && return 0
  for f in /opt/homebrew/lib/libespeak-ng.dylib /usr/local/lib/libespeak-ng.dylib \
           /usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 /usr/lib/libespeak-ng.so.1; do
    [ -e "$f" ] && return 0
  done
  return 1
}
have_espeak_data() {
  for d in /usr/lib/x86_64-linux-gnu /usr/share /usr/local/share /usr/lib \
           /opt/homebrew/share /usr/local/opt/espeak-ng/share; do
    [ -f "$d/espeak-ng-data/phontab" ] && return 0
  done
  return 1
}
if have_espeak_lib && have_espeak_data; then
  echo "-- espeak-ng: library and data both present"
else
  echo "-- installing espeak-ng (the voice engine needs its data files, not just the library)"
  case "$(uname -s)" in
    Darwin)
      if command -v brew >/dev/null 2>&1; then brew install espeak-ng
      else echo "   Homebrew not found — install it (https://brew.sh), then re-run."; exit 1; fi ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then sudo apt-get install -y espeak-ng espeak-ng-data
      elif command -v dnf >/dev/null 2>&1; then sudo dnf install -y espeak-ng espeak-ng-data
      elif command -v pacman >/dev/null 2>&1; then sudo pacman -S --noconfirm espeak-ng
      else echo "   Install espeak-ng with your package manager, then re-run."; exit 1; fi ;;
    *) echo "   Unknown platform — install espeak-ng manually, then re-run."; exit 1 ;;
  esac
  have_espeak_data || { echo "   espeak-ng-data still missing after install — stopping."; exit 1; }
fi

# --- Linux audio headers (sounddevice needs PortAudio) ---
if [ "$(uname -s)" = "Linux" ] && ! ldconfig -p 2>/dev/null | grep -q portaudio; then
  echo "-- installing PortAudio (mic + speaker access)"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y libportaudio2 portaudio19-dev
  fi
fi

# --- the Python environment, pinned ---
# ~6 GB on a Linux box with an NVIDIA card: torch alone carries the CUDA
# runtime. requirements.lock.txt pins all of it, so two machines get the
# same versions instead of "whatever resolved today".
echo "-- creating the environment from requirements.lock.txt (first run downloads ~6GB)"
uv venv .venv -q 2>/dev/null || true
if [ -f requirements.lock.txt ]; then
  uv pip install --python .venv/bin/python -q -r requirements.lock.txt
else
  echo "   (no lock file — resolving fresh)"
  uv pip install --python .venv/bin/python -q -e .
fi
uv pip install --python .venv/bin/python -q --no-deps -e .

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "-- GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
fi

# --- prefetch the models so the first conversation doesn't wait ---
if [ "$1" != "--no-models" ]; then
  echo "-- downloading the speech models (first run only, ~1GB total)"
  .venv/bin/python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from backtalk.ears import warm as warm_ears
from backtalk.mouth import warm as warm_mouth
warm_ears(); warm_mouth()
print("-- models ready")
PY
fi

# --- THE GATE ---
# Upstream stopped at "the models loaded". Loading proves nothing: the
# CUDA path loads fine and dies on the first real sentence, and kokoro
# without its espeak data kills the process. So speak a sentence, listen
# to it back, and fail loudly if the loop does not close.
if [ "$1" != "--no-models" ]; then
  echo "-- verifying the loop (speaking a sentence and transcribing it back)"
  .venv/bin/python verify.py || {
    echo ""
    echo "!! backtalk installed but the voice loop did NOT close."
    echo "   The output above says which half failed. Nothing is wired yet;"
    echo "   fix that first — see TROUBLESHOOTING.md."
    exit 1
  }
fi

echo ""
echo "== backtalk installed and verified =="
echo ""
echo "Next:"
echo "  1. Point it at your agent: edit backtalk.json (agent_dir + name),"
echo "     or open this folder in Claude Code and say:"
echo "         read backtalk.md and set me up"
echo "  2. ./run.sh — hold the key, talk, let go."
