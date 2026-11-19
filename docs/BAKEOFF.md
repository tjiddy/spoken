# Voice bakeoff

Six Ashenvale lines, every engine, your voice, one page. Each run adds a column to
`bakeoff/index.html`; open it in a browser and listen across.

You need a reference clip first: `voices/me/ref.wav` (10–20 s, you, loud and angry, clean room)
and `voices/me/ref.txt` with its exact words. The server engines require the transcript.

## Round 1: Chatterbox (native Windows, its own venv)

```powershell
py -3.12 -m venv .venvs\chatterbox
.venvs\chatterbox\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-tts.txt
pip install chatterbox-tts==0.1.7 --no-deps

python -m jackson fetch
python -m jackson bakeoff --voice me --engine chatterbox --label chatterbox-0.7
python -m jackson bakeoff --voice me --engine chatterbox --label chatterbox-1.0 --exaggeration 1.0 --cfg-weight 0.3
```

## Rounds 2+: the big models, served by SGLang-Omni (Docker + WSL2)

SGLang-Omni serves many current models behind the same `/v1/audio/speech` API, so the pipeline
talks to all of them the same way. It runs on Linux; on Windows that means Docker Desktop with the
WSL2 backend (GPU support is built in; check with `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`).

Start the container once and keep it running (models are cached in the `hf` volume):

```powershell
docker pull hongccc/sglang-omni:dev
docker run -it --gpus all --ipc host --shm-size 32g -p 8000:8000 -v hf:/root/.cache/huggingface hongccc/sglang-omni:dev /bin/zsh
```

Inside the container, once:

```bash
pip install --upgrade pip uv
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install --prerelease=allow "sglang-omni==0.2.0"
git clone --depth 1 https://github.com/sgl-project/sglang-omni.git && cd sglang-omni   # for examples/configs
```

Then serve **one model at a time** (24 GB is one model's worth), run its bakeoff from a second
Windows terminal in a light venv (`pip install -r requirements-server.txt`, no PyTorch needed),
then Ctrl-C the server and start the next.

### Higgs Audio v3 (Boson AI, 4B) — inline emotion tags

```bash
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000
```
```powershell
python -m jackson bakeoff --voice me --engine server --server-model bosonai/higgs-audio-v3-tts-4b --label higgs-plain
python -m jackson bakeoff --voice me --engine server --server-model bosonai/higgs-audio-v3-tts-4b --label higgs-angry --delivery "<|emotion:anger|><|prosody:expressive_high|>"
python -m jackson bakeoff --voice me --engine server --server-model bosonai/higgs-audio-v3-tts-4b --label higgs-menace --delivery "<|emotion:determination|><|prosody:pitch_low|><|prosody:speed_slow|>"
```
Other tags worth a try: `<|style:shouting|>`, `<|emotion:disgust|>`, `<|emotion:bitterness|>`.
License: research and non-commercial, fine for a private pack.

### dots.tts — won a 2026 community blind test of open cloners

```bash
sgl-omni serve --config examples/configs/dots_tts.yaml --port 8000
```
```powershell
python -m jackson bakeoff --voice me --engine server --server-model dots-studio/dots.tts-mf --label dots
python -m jackson bakeoff --voice me --engine server --server-model dots-studio/dots.tts-mf --label dots-furious --server-extra '{"instructions": "furious, intense, gravelly, preacher cadence"}'
```
`--server-extra '{"speaker_scale": 2.0}'` pushes harder toward your timbre.

### MOSS-TTS v1.5 — second in that blind test, has a measured 24 GB profile

```bash
sgl-omni serve --model-path OpenMOSS-Team/MOSS-TTS-v1.5 --config examples/configs/moss_tts_24gb.yaml --port 8000
```
```powershell
python -m jackson bakeoff --voice me --engine server --server-model OpenMOSS-Team/MOSS-TTS-v1.5 --label moss
```

### Fish Audio S2-Pro

```bash
sgl-omni serve --model-path fishaudio/s2-pro --config examples/configs/s2pro_tts.yaml --port 8000
```
```powershell
python -m jackson bakeoff --voice me --engine server --server-model fishaudio/s2-pro --label fish-s2pro
```

### Qwen3-TTS 1.7B Base (Apache 2.0)

```bash
sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --config examples/configs/qwen3_tts_1_7b.yaml --port 8000
```
```powershell
python -m jackson bakeoff --voice me --engine server --server-model Qwen/Qwen3-TTS-12Hz-1.7B-Base --label qwen3
```

If a model refuses to start for lack of VRAM, its cookbook page in the sglang-omni repo
(`docs/cookbook/<model>.md`) lists the memory knobs.

## After the bakeoff

Render the zone with the winner, same flags as its column:

```powershell
python -m jackson render --region ashenvale --voice me --engine server --server-model bosonai/higgs-audio-v3-tts-4b --delivery "<|emotion:anger|><|prosody:expressive_high|>" --takes 1
```

`--takes 3` (the default) adds the Whisper word check, which needs PyTorch and transformers in the
client venv; with `--takes 1` the server venv stays light.
