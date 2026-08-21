# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CUDA plumbing — MarQed addition.

WHY THIS EXISTS. faster-whisper runs on ctranslate2, which is built
against the CUDA 12 runtime (it dlopens libcublas.so.12). Modern torch
wheels — pulled in by kokoro — ship the CUDA *13* runtime instead, into
`site-packages/nvidia/cu13/lib`. Both live in the same venv, so a GPU
machine ends up with libcublas.so.13 present and libcublas.so.12
missing, and ctranslate2 dies with:

    RuntimeError: Library libcublas.so.12 is not found or cannot be loaded

The install adds the CUDA 12 wheels alongside (different sonames, so
nothing collides), and this module dlopens them into the process before
ctranslate2 is imported. Once a library is in the process by soname, a
later dlopen finds it — so this needs no LD_LIBRARY_PATH, which means it
works from a desktop shortcut with a bare PATH just as well as from a
shell.

THE SECOND HALF, and the reason a probe exists at all: loading the model
on CUDA SUCCEEDS even when the runtime is broken. It only blows up on
the first real transcription — so an install that "verified the model
loads" reports green and dies on the user's first sentence. resolve()
therefore transcribes actual audio before it calls the GPU usable.
"""
import ctypes
import glob
import os
import sys

from backtalk.vlog import log

# Loaded in dependency order; cublasLt before cublas, cudnn's graph
# before the engines that call into it.
_SONAMES = ("libcublasLt.so.12", "libcublas.so.12",
            "libcudnn_graph.so.9", "libcudnn_ops.so.9", "libcudnn.so.9")


def _nvidia_lib_dirs() -> list[str]:
    """The nvidia/*/lib folders inside whatever environment we run in."""
    dirs = []
    for base in {os.path.dirname(os.path.dirname(os.__file__)),
                 *[p for p in sys.path if p.endswith("site-packages")]}:
        dirs += glob.glob(os.path.join(base, "nvidia", "*", "lib"))
    return sorted(set(dirs))


def preload_cuda() -> int:
    """dlopen the CUDA 12 libs ctranslate2 needs. Returns how many
    loaded. Never raises: a missing library here is not fatal, it just
    means the GPU path will be rejected by the probe below."""
    loaded = 0
    for soname in _SONAMES:
        for d in _nvidia_lib_dirs():
            path = os.path.join(d, soname)
            if not os.path.exists(path):
                continue
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                pass
            break
    return loaded


def resolve(want_device: str, want_compute: str) -> tuple[str, str]:
    """Decide the device the STT model will ACTUALLY work on.

    "cpu" is honoured as-is. "auto" and "cuda" are proven, not assumed:
    the libs get preloaded, a model is built, and a real (silent) buffer
    is transcribed through it. Anything that throws sends us to the CPU
    with an honest line in the log instead of a crash on the user's
    first sentence. On "cuda" the failure is still a fallback, never an
    exception — a voice line that runs slower is better than one that
    does not run.
    """
    if want_device == "cpu":
        return "cpu", (want_compute if want_compute != "float16" else "int8")

    import numpy as np

    n = preload_cuda()
    try:
        import ctranslate2
        have_gpu = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        have_gpu = False
    if not have_gpu:
        log("[gpu] no CUDA device — running the ears on the CPU")
        return "cpu", "int8"

    compute = want_compute if want_compute != "int8" else "float16"
    try:
        from faster_whisper import WhisperModel
        from backtalk.config import CFG
        m = WhisperModel(CFG["stt_model"], device="cuda",
                         compute_type=compute)
        # THE PROBE: loading is not proof. Transcribe real samples.
        list(m.transcribe(np.zeros(16000, dtype=np.float32),
                          temperature=0.0)[0])
        log(f"[gpu] CUDA ready ({n} runtime libs preloaded, "
            f"compute {compute})")
        return "cuda", compute
    except Exception as e:
        log(f"[gpu] CUDA present but unusable ({type(e).__name__}: "
            f"{str(e)[:90]}) — falling back to the CPU")
        return "cpu", "int8"
