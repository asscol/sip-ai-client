"""Главный модуль SIP AI Client."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import click
from loguru import logger

from sip_ai_client.config import AppConfig, LLMProvider, load_config


def setup_logging(log_level: str) -> None:
    """Настроить логирование."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        "logs/sip-ai-client.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )


def create_stt(config: AppConfig):
    """Создать движок распознавания речи."""
    from sip_ai_client.stt.whisper_stt import WhisperSTT

    stt = WhisperSTT(config.whisper)
    stt.initialize()
    return stt


def create_tts(config: AppConfig):
    """Создать движок синтеза речи."""
    from sip_ai_client.tts.piper_tts import PiperTTS

    tts = PiperTTS(config.piper)
    tts.initialize()
    return tts


def create_llm(config: AppConfig):
    """Создать LLM движок в зависимости от конфигурации."""
    if config.llm_provider == LLMProvider.OLLAMA:
        from sip_ai_client.llm.ollama_llm import OllamaLLM

        llm = OllamaLLM(config.ollama)
        llm.initialize()
        return llm
    else:
        from sip_ai_client.llm.cloud_llm import CloudLLM

        if not config.cloud_api.api_key:
            logger.error("API ключ для облачного LLM не указан!")
            logger.info(
                "Укажите ключ через config.yaml (cloud_api.api_key) "
                "или переменную окружения SIP_AI_CLOUD_API__API_KEY"
            )
            sys.exit(1)

        llm = CloudLLM(config.cloud_api)
        llm.initialize()
        return llm


@click.command()
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    type=click.Path(),
    help="Путь к файлу конфигурации",
)
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Уровень логирования",
)
@click.option(
    "--llm-provider",
    default=None,
    type=click.Choice(["ollama", "openai", "cloud_api"], case_sensitive=False),
    help="Провайдер LLM (ollama или cloud_api/openai)",
)
@click.option(
    "--sip-server",
    default=None,
    help="Адрес SIP-сервера",
)
@click.option(
    "--sip-username",
    default=None,
    help="SIP имя пользователя",
)
@click.option(
    "--sip-password",
    default=None,
    help="SIP пароль",
)
def main(
    config_path: str,
    log_level: str | None,
    llm_provider: str | None,
    sip_server: str | None,
    sip_username: str | None,
    sip_password: str | None,
) -> None:
    """SIP AI Client — голосовой ИИ-ассистент.

    Принимает звонки по SIP, распознаёт речь (Whisper),
    генерирует ответы (Ollama/Cloud LLM) и озвучивает их (Piper TTS).
    """
    config = load_config(config_path)

    if log_level:
        config.log_level = log_level.upper()
    if llm_provider:
        config.llm_provider = LLMProvider(llm_provider)
    if sip_server:
        config.sip.server = sip_server
    if sip_username:
        config.sip.username = sip_username
    if sip_password:
        config.sip.password = sip_password

    setup_logging(config.log_level)

    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("SIP AI Client v0.1.0")
    logger.info("=" * 60)
    logger.info("LLM провайдер: {}", config.llm_provider.value)
    logger.info("SIP сервер: {}:{}", config.sip.server, config.sip.port)
    logger.info("Whisper модель: {}", config.whisper.model_size)
    logger.info("Piper голос: {}", config.piper.voice)
    logger.info("=" * 60)

    logger.info("Инициализация компонентов...")

    stt = create_stt(config)
    logger.info("STT (Whisper) готов")

    tts = create_tts(config)
    logger.info("TTS (Piper) готов")

    llm = create_llm(config)
    logger.info("LLM ({}) готов", config.llm_provider.value)

    from sip_ai_client.sip_client import SIPClient

    sip_client = SIPClient(config, stt, tts, llm)

    shutdown_event = False

    def signal_handler(signum, frame):
        nonlocal shutdown_event
        if not shutdown_event:
            shutdown_event = True
            logger.info("Получен сигнал завершения, остановка...")
            sip_client.stop()
            stt.shutdown()
            tts.shutdown()
            llm.shutdown()
            logger.info("SIP AI Client остановлен")
            sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    sip_client.start()

    logger.info("SIP AI Client запущен. Нажмите Ctrl+C для остановки.")

    try:
        while sip_client.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if not shutdown_event:
            sip_client.stop()
            stt.shutdown()
            tts.shutdown()
            llm.shutdown()
            logger.info("SIP AI Client остановлен")


if __name__ == "__main__":
    main()
