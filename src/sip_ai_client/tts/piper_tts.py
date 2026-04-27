"""Синтез речи с помощью Piper TTS (голос Ruslan)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from sip_ai_client.audio.processor import resample
from sip_ai_client.config import PiperConfig


class PiperTTS:
    """Обёртка над Piper TTS для синтеза русской речи."""

    def __init__(self, config: PiperConfig):
        self.config = config
        self._piper_path: Optional[str] = None
        self._model_path: Optional[str] = None

    def initialize(self) -> None:
        """Инициализация Piper TTS."""
        self._piper_path = self._find_piper()
        if self.config.model_path:
            self._model_path = self.config.model_path
        else:
            self._model_path = self._download_voice()

        logger.info("Piper TTS инициализирован: голос={}", self.config.voice)

    def _find_piper(self) -> str:
        """Найти исполняемый файл piper."""
        try:
            result = subprocess.run(
                ["which", "piper"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            pass

        try:
            result = subprocess.run(
                ["python3", "-m", "piper", "--help"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return "python3 -m piper"
        except FileNotFoundError:
            pass

        return "piper"

    def _download_voice(self) -> str:
        """Скачать голосовую модель если не существует."""
        data_dir = Path(self.config.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        voice_name = self.config.voice
        model_file = data_dir / f"{voice_name}.onnx"
        config_file = data_dir / f"{voice_name}.onnx.json"

        if model_file.exists() and config_file.exists():
            logger.info("Голосовая модель найдена: {}", model_file)
            return str(model_file)

        logger.info("Скачивание голосовой модели: {}", voice_name)

        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
        lang_code = voice_name.split("-")[0]
        lang_family = lang_code.replace("_", "/")

        model_url = f"{base_url}/{lang_family}/{voice_name}/{voice_name}.onnx"
        config_url = f"{base_url}/{lang_family}/{voice_name}/{voice_name}.onnx.json"

        try:
            import httpx

            with httpx.Client(follow_redirects=True, timeout=120) as client:
                logger.info("Скачивание модели: {}", model_url)
                resp = client.get(model_url)
                resp.raise_for_status()
                model_file.write_bytes(resp.content)

                logger.info("Скачивание конфига: {}", config_url)
                resp = client.get(config_url)
                resp.raise_for_status()
                config_file.write_bytes(resp.content)

            logger.info("Голосовая модель скачана: {}", model_file)
        except Exception as e:
            logger.error("Ошибка скачивания модели: {}", e)
            raise RuntimeError(f"Не удалось скачать голосовую модель: {e}") from e

        return str(model_file)

    def synthesize(self, text: str) -> Optional[bytes]:
        """Синтезировать речь из текста.

        Args:
            text: Текст для синтеза.

        Returns:
            PCM16 аудио данные с частотой output_sample_rate или None.
        """
        if not text or not text.strip():
            return None

        logger.debug("Синтез речи: '{}'", text[:80])

        try:
            cmd = [
                "piper",
                "--model", self._model_path or "",
                "--output_raw",
                "--length-scale", str(self.config.length_scale),
                "--noise-scale", str(self.config.noise_scale),
                "--noise-w", str(self.config.noise_w),
                "--sentence-silence", str(self.config.sentence_silence),
            ]

            if self.config.speaker_id is not None:
                cmd.extend(["--speaker", str(self.config.speaker_id)])

            result = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=False,
                timeout=30,
                env=None,
            )

            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                logger.error("Piper TTS ошибка: {}", stderr)
                return None

            raw_audio = result.stdout
            if not raw_audio:
                logger.warning("Piper TTS вернул пустой результат")
                return None

            audio_float = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

            if self.config.sample_rate != self.config.output_sample_rate:
                audio_float = resample(
                    audio_float,
                    self.config.sample_rate,
                    self.config.output_sample_rate,
                )

            pcm_data = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

            logger.info(
                "Речь синтезирована: {} байт, {} сэмплов",
                len(pcm_data),
                len(pcm_data) // 2,
            )
            return pcm_data

        except subprocess.TimeoutExpired:
            logger.error("Piper TTS таймаут")
            return None
        except Exception as e:
            logger.error("Ошибка синтеза речи: {}", e)
            return None

    def shutdown(self) -> None:
        """Освободить ресурсы."""
        logger.info("Piper TTS остановлен")
