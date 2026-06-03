# Voice model

The auto-assembler (`assemble.py`) uses a neural text-to-speech voice for a
natural-sounding narration. The voice model itself (~115 MB) is not committed -
download it once:

```bash
pip3 install piper-tts
cd voices
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high"
curl -sL -o en_US-ryan-high.onnx      "$BASE/en_US-ryan-high.onnx"
curl -sL -o en_US-ryan-high.onnx.json "$BASE/en_US-ryan-high.onnx.json"
```

Then `python3 assemble.py` will pick it up automatically. If the model is
absent, the assembler falls back to the macOS `say` voice.

Other good Piper voices (swap the path in assemble.py's `PIPER_MODEL`):
- `en_US-lessac-high` - clear, neutral
- `en_US-hfc_female-medium` - warm female
- browse: https://huggingface.co/rhasspy/piper-voices
