#!/usr/bin/env python3
# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The install gate — MarQed addition.

Upstream's install ended at "the models loaded", which is not evidence.
Both real failures found on a fresh Ubuntu box passed that bar:

  * kokoro loads, then espeak-ng calls exit() on a missing phontab —
    the install script dies mid-run and still returned before the fix
  * faster-whisper LOADS on CUDA and only raises on the first actual
    transcription, so the break lands on the user's first sentence

So this speaks a known sentence through the real mouth, feeds the real
samples to the real ears, and compares the words that come back. It
exits non-zero when the loop does not close. Run it any time:

    .venv/bin/python verify.py
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ZIN = "The circuit board is listening, sir. Shall we begin the measurement?"
DREMPEL = 0.7          # fraction of words that must survive the round trip


def woorden(t):
    return {w.strip(".,!?").lower() for w in t.split() if w.strip(".,!?")}


def main() -> int:
    import numpy as np
    from backtalk.config import CFG

    print(f"   voice={CFG['voice']}  stt={CFG['stt_model']} "
          f"(requested: {CFG['stt_device']}/{CFG['stt_compute']})")

    # --- the mouth ---
    try:
        from backtalk.mouth import synth_stream, warm as warm_mouth
        warm_mouth()
        t0 = time.time()
        rate, chunks, eerste = None, [], None
        for r, pcm in synth_stream(ZIN):
            if eerste is None:
                eerste = time.time() - t0
            rate, _ = r, chunks.append(pcm)
        audio = np.concatenate(chunks)
    except SystemExit:
        # espeak-ng aborts the interpreter rather than raising
        print("   MOUTH: FAILED — the voice engine killed the process "
              "(espeak-ng data files missing)")
        return 1
    except Exception as e:
        print(f"   MOUTH: FAILED — {type(e).__name__}: {e}")
        return 1
    if audio.size == 0 or int(np.abs(audio).max()) < 500:
        print("   MOUTH: FAILED — the voice produced silence")
        return 1
    print(f"   MOUTH: ok — {audio.size / rate:.1f}s of audio, first "
          f"chunk after {eerste * 1000:.0f} ms")

    # --- the ears ---
    try:
        from backtalk.ears import transcribe, warm as warm_ears
        warm_ears()
        n16 = int(audio.size * 16000 / rate)
        a16 = np.interp(np.linspace(0, audio.size - 1, n16),
                        np.arange(audio.size),
                        audio.astype(float)).astype(np.int16)
        t0 = time.time()
        heard = transcribe(a16)
        duur = time.time() - t0
    except Exception as e:
        print(f"   EARS: FAILED — {type(e).__name__}: {e}")
        return 1

    w1, w2 = woorden(ZIN), woorden(heard)
    score = len(w1 & w2) / len(w1) if w1 else 0.0
    print(f"   EARS: heard {heard!r} in {duur:.2f}s "
          f"({score * 100:.0f}% of the words)")
    if score < DREMPEL:
        print(f"   EARS: FAILED — under the {DREMPEL * 100:.0f}% bar; "
              f"speech recognition is not working")
        return 1

    print("   LOOP CLOSED: the voice spoke and the ears understood it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
