"""Обработка аудио — конвертация, VAD, буферизация."""

from __future__ import annotations

import io
import struct
import wave
from typing import Optional

import numpy as np
from loguru import logger

from sip_ai_client.config import AudioConfig

PCMU_DECODE_TABLE = [
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
    -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
    -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
    -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
    -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
    -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
    -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
    -876, -844, -812, -780, -748, -716, -684, -652,
    -620, -588, -556, -524, -492, -460, -428, -396,
    -372, -356, -340, -324, -308, -292, -276, -260,
    -244, -228, -212, -196, -180, -164, -148, -132,
    -120, -112, -104, -96, -88, -80, -72, -64,
    -56, -48, -40, -32, -24, -16, -8, 0,
    32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
    23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
    15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
    11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
    7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
    5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
    3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
    2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
    1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
    1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
    876, 844, 812, 780, 748, 716, 684, 652,
    620, 588, 556, 524, 492, 460, 428, 396,
    372, 356, 340, 324, 308, 292, 276, 260,
    244, 228, 212, 196, 180, 164, 148, 132,
    120, 112, 104, 96, 88, 80, 72, 64,
    56, 48, 40, 32, 24, 16, 8, 0,
]


def ulaw_to_pcm16(ulaw_data: bytes) -> bytes:
    """Декодировать μ-law (G.711) в PCM16."""
    pcm_samples = []
    for byte in ulaw_data:
        pcm_samples.append(PCMU_DECODE_TABLE[byte])
    return struct.pack(f"<{len(pcm_samples)}h", *pcm_samples)


def pcm16_to_ulaw(pcm_data: bytes) -> bytes:
    """Кодировать PCM16 в μ-law (G.711)."""
    samples = struct.unpack(f"<{len(pcm_data) // 2}h", pcm_data)
    ulaw_bytes = []
    for sample in samples:
        sign = 0
        if sample < 0:
            sign = 0x80
            sample = -sample
        sample = min(sample, 32635)
        sample += 0x84

        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (sample & mask):
            exponent -= 1
            mask >>= 1

        mantissa = (sample >> (exponent + 3)) & 0x0F
        ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
        ulaw_bytes.append(ulaw_byte)
    return bytes(ulaw_bytes)


def pcm16_to_float32(pcm_data: bytes) -> np.ndarray:
    """Конвертировать PCM16 в float32 numpy массив."""
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Конвертировать float32 numpy массив в PCM16."""
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767).astype(np.int16)
    return samples.tobytes()


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Простой ресемплинг через линейную интерполяцию."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def audio_to_wav_bytes(audio_data: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Обернуть PCM16 данные в WAV формат."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    return buf.getvalue()


class AudioBuffer:
    """Буфер для накопления аудио-фреймов с детекцией тишины."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self._buffer: list[bytes] = []
        self._silence_frames = 0
        self._has_speech = False
        self._frames_per_second = config.sample_rate // config.chunk_size

    def add_frame(self, frame: bytes) -> Optional[bytes]:
        """Добавить фрейм. Вернуть полный аудио если обнаружен конец речи."""
        rms = self._calculate_rms(frame)
        is_silence = rms < self.config.silence_threshold

        if not is_silence:
            self._has_speech = True
            self._silence_frames = 0
            self._buffer.append(frame)
        elif self._has_speech:
            self._silence_frames += 1
            self._buffer.append(frame)

            silence_seconds = self._silence_frames / self._frames_per_second
            if silence_seconds >= self.config.silence_duration:
                return self._flush()

        total_frames = len(self._buffer)
        max_frames = self.config.max_record_seconds * self._frames_per_second
        if total_frames >= max_frames and self._has_speech:
            return self._flush()

        return None

    def _flush(self) -> bytes:
        """Вернуть накопленные данные и сбросить буфер."""
        result = b"".join(self._buffer)
        self.reset()
        return result

    def reset(self) -> None:
        """Сбросить буфер."""
        self._buffer.clear()
        self._silence_frames = 0
        self._has_speech = False

    @staticmethod
    def _calculate_rms(frame: bytes) -> float:
        """Вычислить RMS уровень аудио фрейма."""
        if len(frame) < 2:
            return 0.0
        samples = np.frombuffer(frame, dtype=np.int16)
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
