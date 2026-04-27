"""Конфигурация SIP AI Client."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    CLOUD_API = "cloud_api"


class SIPConfig(BaseModel):
    """Настройки SIP-подключения."""

    server: str = "127.0.0.1"
    port: int = 5060
    username: str = ""
    password: str = ""
    transport: str = "udp"
    realm: str = ""
    display_name: str = "SIP AI Client"


class WhisperConfig(BaseModel):
    """Настройки распознавания речи Whisper."""

    model_size: str = "large-v3"
    language: str = "ru"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_duration_ms: int = 500


class PiperConfig(BaseModel):
    """Настройки синтеза речи Piper."""

    voice: str = "ru_RU-ruslan-medium"
    model_path: Optional[str] = None
    data_dir: str = str(Path.home() / ".local" / "share" / "piper-voices")
    speaker_id: Optional[int] = None
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    sentence_silence: float = 0.2
    sample_rate: int = 22050
    output_sample_rate: int = 8000


class OllamaConfig(BaseModel):
    """Настройки Ollama для локальных LLM."""

    host: str = "http://localhost:11434"
    model: str = "llama3.1:70b"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    system_prompt: str = (
        "Ты — голосовой ИИ-ассистент. Отвечай кратко, по-русски. "
        "Ты общаешься по телефону, поэтому отвечай лаконично и понятно."
    )
    context_window: int = 8192


class CloudAPIConfig(BaseModel):
    """Настройки облачного API для LLM."""

    provider: str = "openai"
    api_key: str = ""
    api_base: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 512
    system_prompt: str = (
        "Ты — голосовой ИИ-ассистент. Отвечай кратко, по-русски. "
        "Ты общаешься по телефону, поэтому отвечай лаконично и понятно."
    )


class AudioConfig(BaseModel):
    """Настройки аудио."""

    sample_rate: int = 8000
    channels: int = 1
    sample_width: int = 2
    chunk_size: int = 160
    silence_threshold: float = 500.0
    silence_duration: float = 1.5
    max_record_seconds: int = 30
    codec: str = "PCMU"


class AppConfig(BaseSettings):
    """Основная конфигурация приложения."""

    sip: SIPConfig = Field(default_factory=SIPConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    piper: PiperConfig = Field(default_factory=PiperConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    cloud_api: CloudAPIConfig = Field(default_factory=CloudAPIConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)

    llm_provider: LLMProvider = LLMProvider.OLLAMA

    log_level: str = "INFO"

    model_config = {"env_prefix": "SIP_AI_", "env_nested_delimiter": "__"}


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Загрузить конфигурацию из YAML-файла."""
    if config_path is None:
        config_path = Path("config.yaml")

    config_path = Path(config_path)

    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig(**raw)

    return AppConfig()
