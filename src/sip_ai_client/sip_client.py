"""SIP-клиент для приёма и обработки звонков."""

from __future__ import annotations

import threading
from typing import Optional, Protocol

from loguru import logger

from sip_ai_client.audio.processor import (
    AudioBuffer,
    pcm16_to_ulaw,
    ulaw_to_pcm16,
)
from sip_ai_client.config import AppConfig, AudioConfig


class STTEngine(Protocol):
    """Протокол для движка распознавания речи."""

    def transcribe(self, audio_data: bytes, sample_rate: int) -> Optional[str]: ...


class TTSEngine(Protocol):
    """Протокол для движка синтеза речи."""

    def synthesize(self, text: str) -> Optional[bytes]: ...


class LLMEngine(Protocol):
    """Протокол для LLM движка."""

    def generate_response(self, user_text: str) -> Optional[str]: ...
    def reset_conversation(self) -> None: ...


class CallHandler:
    """Обработчик отдельного SIP-звонка."""

    def __init__(
        self,
        stt: STTEngine,
        tts: TTSEngine,
        llm: LLMEngine,
        audio_config: AudioConfig,
    ):
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.audio_config = audio_config
        self.audio_buffer = AudioBuffer(audio_config)
        self._playback_queue: list[bytes] = []
        self._playback_offset = 0
        self._lock = threading.Lock()
        self._active = True

    def handle_audio_frame(self, frame: bytes) -> Optional[bytes]:
        """Обработать входящий аудио фрейм.

        Args:
            frame: μ-law закодированный аудио фрейм.

        Returns:
            μ-law закодированный ответный фрейм или None.
        """
        if not self._active:
            return self._get_silence_frame()

        pcm_frame = ulaw_to_pcm16(frame)

        with self._lock:
            if self._playback_queue:
                return self._get_playback_frame()

        complete_audio = self.audio_buffer.add_frame(pcm_frame)
        if complete_audio is not None:
            threading.Thread(
                target=self._process_speech,
                args=(complete_audio,),
                daemon=True,
            ).start()

        return self._get_playback_frame()

    def _process_speech(self, audio_data: bytes) -> None:
        """Обработать распознанную речь: STT → LLM → TTS."""
        try:
            text = self.stt.transcribe(audio_data, self.audio_config.sample_rate)
            if not text:
                logger.debug("Речь не распознана, пропуск")
                return

            logger.info("Пользователь сказал: '{}'", text)

            response = self.llm.generate_response(text)
            if not response:
                logger.warning("LLM не сгенерировал ответ")
                return

            logger.info("ИИ отвечает: '{}'", response[:100])

            audio_response = self.tts.synthesize(response)
            if not audio_response:
                logger.warning("TTS не синтезировал речь")
                return

            chunk_size = self.audio_config.chunk_size * 2
            chunks = []
            for i in range(0, len(audio_response), chunk_size):
                chunk = audio_response[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk += b"\x00" * (chunk_size - len(chunk))
                chunks.append(chunk)

            with self._lock:
                self._playback_queue = chunks
                self._playback_offset = 0

            logger.info("Ответ поставлен в очередь: {} фреймов", len(chunks))

        except Exception as e:
            logger.error("Ошибка обработки речи: {}", e)

    def _get_playback_frame(self) -> bytes:
        """Получить следующий фрейм для воспроизведения."""
        with self._lock:
            if self._playback_queue and self._playback_offset < len(self._playback_queue):
                pcm_frame = self._playback_queue[self._playback_offset]
                self._playback_offset += 1
                if self._playback_offset >= len(self._playback_queue):
                    self._playback_queue.clear()
                    self._playback_offset = 0
                return pcm16_to_ulaw(pcm_frame)

        return self._get_silence_frame()

    def _get_silence_frame(self) -> bytes:
        """Получить фрейм тишины."""
        return b"\xff" * self.audio_config.chunk_size

    def hangup(self) -> None:
        """Завершить звонок."""
        self._active = False
        self.audio_buffer.reset()
        with self._lock:
            self._playback_queue.clear()
        self.llm.reset_conversation()
        logger.info("Звонок завершён")


class SIPClient:
    """SIP-клиент для подключения к серверу и обработки звонков."""

    def __init__(
        self,
        config: AppConfig,
        stt: STTEngine,
        tts: TTSEngine,
        llm: LLMEngine,
    ):
        self.config = config
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self._phone = None
        self._running = False
        self._active_calls: dict[str, CallHandler] = {}

    def start(self) -> None:
        """Запустить SIP-клиент и подключиться к серверу."""
        from pyVoIP.VoIP import VoIPPhone

        sip = self.config.sip

        logger.info(
            "Запуск SIP-клиента: {}@{}:{}",
            sip.username,
            sip.server,
            sip.port,
        )

        self._phone = VoIPPhone(
            server=sip.server,
            port=sip.port,
            username=sip.username,
            password=sip.password,
            callCallback=self._on_incoming_call,
        )

        self._phone.start()
        self._running = True
        logger.info("SIP-клиент запущен и ожидает звонки")

    def _on_incoming_call(self, call) -> None:
        """Обработать входящий звонок."""
        from pyVoIP.VoIP import CallState

        try:
            call_id = str(id(call))
            logger.info("Входящий звонок: {}", call_id)

            call.answer()
            logger.info("Звонок принят: {}", call_id)

            handler = CallHandler(
                stt=self.stt,
                tts=self.tts,
                llm=self.llm,
                audio_config=self.config.audio,
            )
            self._active_calls[call_id] = handler

            while call.state == CallState.ANSWERED:
                try:
                    audio_in = call.readAudio(
                        length=self.config.audio.chunk_size,
                        blocking=True,
                    )
                except Exception:
                    break

                audio_out = handler.handle_audio_frame(audio_in)
                if audio_out:
                    try:
                        call.writeAudio(audio_out)
                    except Exception:
                        break

            handler.hangup()
            del self._active_calls[call_id]
            logger.info("Звонок завершён: {}", call_id)

        except Exception as e:
            logger.error("Ошибка обработки звонка: {}", e)

    def make_call(self, destination: str) -> Optional[CallHandler]:
        """Совершить исходящий звонок.

        Args:
            destination: SIP URI или номер для вызова.

        Returns:
            CallHandler для управления звонком или None.
        """
        if self._phone is None:
            logger.error("SIP-клиент не запущен")
            return None

        try:
            logger.info("Исходящий звонок: {}", destination)
            call = self._phone.call(destination)

            call_id = str(id(call))
            handler = CallHandler(
                stt=self.stt,
                tts=self.tts,
                llm=self.llm,
                audio_config=self.config.audio,
            )
            self._active_calls[call_id] = handler
            return handler

        except Exception as e:
            logger.error("Ошибка исходящего звонка: {}", e)
            return None

    def stop(self) -> None:
        """Остановить SIP-клиент."""
        self._running = False

        for call_id, handler in self._active_calls.items():
            handler.hangup()
        self._active_calls.clear()

        if self._phone:
            self._phone.stop()
            self._phone = None

        logger.info("SIP-клиент остановлен")

    @property
    def is_running(self) -> bool:
        return self._running
