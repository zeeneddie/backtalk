# MarQed fork — wat hier afwijkt van upstream

Fork van **[jaredrhod/backtalk](https://github.com/jaredrhod/backtalk)** (AGPL-3.0-or-later).
Alle eer voor het origineel gaat naar Jared Rhodenizer; deze fork bestaat
alleen om de installatie reproduceerbaar te maken op onze machines.
Fork-punt: tag `fork-point-b3b6cef`.

Upstream bijwerken: `git fetch upstream && git merge upstream/main`.

## Waarom deze fork bestaat

`./install.sh` van upstream **faalde op een verse Ubuntu 24.04-machine met
een NVIDIA-kaart**, op twee onafhankelijke punten. Beide braken pas ná het
moment waarop de installatie zichzelf geslaagd verklaarde.

### 1. espeak-ng: de library is niet genoeg, de data ook nodig

Upstream controleerde of `libespeak-ng.so.1` bestaat en concludeerde
"already present". Die library staat er vaak als afhankelijkheid van iets
anders, terwijl `espeak-ng-data` ontbreekt. Kokoro valt dan terug op het
pad dat in het `espeakng-loader`-wheel is ingebakken —
`/home/runner/work/espeakng-loader/...` — en espeak-ng **roept `exit()`
aan** in plaats van een fout te gooien. Het installatieproces sterft
zonder dat er een uitzondering te vangen valt.

*Gerepareerd:* `mouth._ensure_espeak()` zet nu ook `ESPEAK_DATA_PATH`, en
`install.sh` toetst library én data.

### 2. CUDA: ctranslate2 wil 12, torch brengt 13 mee

`faster-whisper` draait op ctranslate2, gebouwd tegen **CUDA 12**
(`libcublas.so.12`). `torch` — meegetrokken door kokoro — levert de
**CUDA 13**-runtime in `nvidia/cu13/lib`. Een GPU-machine heeft dus
`libcublas.so.13` en niet de `.12`, en ctranslate2 valt om met
`RuntimeError: Library libcublas.so.12 is not found`.

Erger: het model **laadt** wél op `cuda`. Het breekt pas bij de eerste
echte transcriptie — dus bij de eerste zin die de gebruiker uitspreekt.

*Gerepareerd:* `backtalk/gpu.py` (nieuw) laadt de CUDA 12-runtime met
`ctypes` in het proces vóór ctranslate2 geïmporteerd wordt — geen
`LD_LIBRARY_PATH` nodig, dus het werkt ook vanaf een bureaubladsnelkoppeling.
Daarna wordt het GPU-pad **bewezen** met een echte transcriptie; faalt die,
dan valt hij terug op CPU met een eerlijke regel in het log.

### 3. Eén cuDNN per omgeving

`nvidia-cudnn-cu12` en `nvidia-cudnn-cu13` installeren naar **hetzelfde
pad** (`nvidia/cudnn/lib/libcudnn.so.9`). Wie beide installeert, laat de
laatste winnen en torch tegen vreemde sublibraries draaien:
`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, en de stem valt om. Daarom
staat cuDNN 12 **bewust niet** in `requirements-cuda-linux.txt`; cuDNN 9
deelt één soname, dus ctranslate2 gebruikt gewoon het exemplaar dat torch
al meebracht.

## Wat er is toegevoegd

| bestand | wat |
|---|---|
| `backtalk/gpu.py` | CUDA 12-runtime laden + het GPU-pad bewijzen, met CPU-terugval |
| `verify.py` | de poort: spreekt een zin uit, luistert hem terug, faalt hard |
| `requirements.lock.txt` | 419 gepinde pakketten — twee machines krijgen dezelfde versies |
| `requirements-cuda-linux.txt` | de CUDA 12-toevoeging, met de reden erbij |
| `install.sh` | espeak-data-controle, deterministische install, en de poort aan het eind |

## Gemeten op deze machine (2026-08-21)

Ubuntu 24.04 · NVIDIA RTX A2000 Laptop (driver 580.173.02) · Python 3.11

```
[gpu] CUDA ready (5 runtime libs preloaded, compute float16)
   MOUTH: ok — 4.6s of audio, first chunk after 1027 ms
   EARS: heard 'The circuit board is listening, sir. Shall we begin the
         measurement?' in 0.21s (100% of the words)
   LOOP CLOSED
```

Spraakherkenning op GPU **0,21–0,56 s** tegen **2,16 s** op CPU voor
dezelfde 4,6 s audio — ruwweg 4× sneller, gelijke transcriptie.
De omgeving is ~6 GB (torch draagt de CUDA-runtime), niet de ~900 MB
die upstream noemt.

## Licentie

Ongewijzigd: **AGPL-3.0-or-later**. Wijzigingen in deze fork vallen onder
dezelfde licentie. Draait een gewijzigde versie als dienst voor derden,
dan moet de broncode van díé versie beschikbaar zijn.
