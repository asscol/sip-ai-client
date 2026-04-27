"""Распознавание речи с помощью Whisper (faster-whisper)."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from sip_ai_client.audio.processor import pcm16_to_float32, resample
from sip_ai_client.config import WhisperConfig


class WhisperSTT:
    """Обёртка над faster-whisper для распознавания речи."""

    def __init__(self, config: WhisperConfig):
        self.config = config
        self._model = None

    def initialize(self) -> None:
        """Загрузить модель Whisper."""
        from faster_whisper import WhisperModel

        logger.info(
            "Загрузка Whisper модели: {} (device={}, compute={})",
            self.config.model_size,
            self.config.device,
            self.config.compute_type,
        )
        self._model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )
        logger.info("Whisper модель загружена успешно")

    def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 8000,
    ) -> Optional[str]:
        """Распознать речь из PCM16 аудио данных.

        Args:
            audio_data: PCM16 аудио байты.
            sample_rate: Частота дискретизации входного аудио.

        Returns:
            Распознанный текст или None.
        """
        if self._model is None:
            logger.error("Whisper модель не инициализирована")
            return None

        audio_float = pcm16_to_float32(audio_data)

        if sample_rate != 16000:
            audio_float = resample(audio_float, sample_rate, 16000)

        logger.debug("Распознавание речи: {} сэмплов", len(audio_float))

        segments, info = self._model.transcribe(
            audio_float,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            vad_parameters={
                "min_silence_duration_ms": self.config.vad_min_silence_duration_ms,
            },
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        full_text = " ".join(text_parts).strip()

        if full_text:
            logger.info("Распознано: '{}' (язык: {}, вероятность: {:.2f})",
                        full_text, info.language, info.language_probability)
            return full_text

        logger.debug("Речь не распознана")
        return None

    def shutdown(self) -> None:
        """Освободить ресурсы модели."""
        self._model = None
        logger.info("Whisper модель выгружена")
