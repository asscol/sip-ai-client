# Testing SIP AI Client

## Setup

```bash
cd /home/ubuntu/repos/sip-ai-client
pip install -e ".[dev]"
```

## Lint

```bash
ruff check src/
ruff check src/ --fix  # auto-fix
```

## Locally Testable Components (no external services needed)

### Config Loading
- YAML config: `from sip_ai_client.config import load_config; config = load_config('config.yaml')`
- ENV vars use prefix `SIP_AI_` with `__` for nested (e.g. `SIP_AI_SIP__SERVER`)
- Defaults are defined via pydantic models in `src/sip_ai_client/config.py`

### Audio Processor (`src/sip_ai_client/audio/processor.py`)
- μ-law codec: `ulaw_to_pcm16()`, `pcm16_to_ulaw()` — pure Python, no deps
- Float conversion: `pcm16_to_float32()`, `float32_to_pcm16()`
- Resample: `resample(audio, orig_sr, target_sr)` — linear interpolation
- `AudioBuffer` — VAD with configurable silence threshold/duration

### CLI Entry Point
- `sip-ai-client --help` works without any external services
- Missing API key for cloud LLM triggers `SystemExit(1)` with error message

### Module Imports
- All modules can be imported without running services
- Whisper/Piper/Ollama are lazy-loaded only when `initialize()` is called

## External Services Required for Full E2E Testing

| Service | Purpose | How to Set Up |
|---------|---------|---------------|
| SIP server (Asterisk/FreeSWITCH) | Accept/make calls | Configure in `config.yaml` under `sip:` |
| Ollama | Local LLM >19B | `ollama pull llama3.1:70b`, runs on port 11434 |
| Piper TTS | Speech synthesis | Install binary, model auto-downloads on first use |
| Whisper | STT | Model auto-downloads via huggingface, needs ~4GB RAM |
| OpenAI API | Cloud LLM alternative | Set `cloud_api.api_key` in config or `SIP_AI_CLOUD_API__API_KEY` env var |

## Important Notes

- The `wave` stdlib module is used (NOT the PyPI `wave` package — that one pulls MySQL-python and breaks install)
- `pyproject.toml` line length is 100 chars
- Ruff rules: E, F, I, W
- CLI loads components in order: Whisper → Piper → LLM → SIP. The API key check happens during LLM init, so testing missing-key error via CLI may hang during Whisper load. Test via `create_llm()` directly instead.
- AudioBuffer VAD: frames per second = sample_rate / chunk_size. Default: 8000/160 = 50fps. Silence flush = silence_duration × fps frames.

## Devin Secrets Needed

- `SIP_AI_CLOUD_API__API_KEY` — OpenAI or compatible API key (only for cloud LLM testing)
- SIP server credentials — for full call flow testing
