"""Интеграция с Ollama для локальных LLM моделей >19B."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from sip_ai_client.config import OllamaConfig


class OllamaLLM:
    """Клиент Ollama для работы с большими языковыми моделями."""

    def __init__(self, config: OllamaConfig):
        self.config = config
        self._client = None
        self._conversation_history: list[dict[str, str]] = []

    def initialize(self) -> None:
        """Инициализировать клиент Ollama."""
        import ollama

        self._client = ollama.Client(host=self.config.host)

        try:
            models = self._client.list()
            model_names = [m.model for m in models.models]
            logger.info("Ollama доступные модели: {}", model_names)

            if self.config.model not in model_names:
                logger.warning(
                    "Модель {} не найдена. Доступные: {}",
                    self.config.model,
                    model_names,
                )
        except Exception as e:
            logger.warning("Не удалось подключиться к Ollama: {}", e)

        self._conversation_history = [
            {"role": "system", "content": self.config.system_prompt}
        ]

        logger.info("Ollama LLM инициализирован: модель={}", self.config.model)

    def generate_response(self, user_text: str) -> Optional[str]:
        """Сгенерировать ответ на текст пользователя.

        Args:
            user_text: Текст от пользователя.

        Returns:
            Ответ модели или None.
        """
        if self._client is None:
            logger.error("Ollama клиент не инициализирован")
            return None

        self._conversation_history.append({"role": "user", "content": user_text})

        try:
            response = self._client.chat(
                model=self.config.model,
                messages=self._conversation_history,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_tokens,
                    "num_ctx": self.config.context_window,
                },
            )

            assistant_text = response.message.content.strip()

            self._conversation_history.append(
                {"role": "assistant", "content": assistant_text}
            )

            self._trim_history()

            logger.info("Ollama ответ: '{}'", assistant_text[:100])
            return assistant_text

        except Exception as e:
            logger.error("Ошибка Ollama: {}", e)
            self._conversation_history.pop()
            return None

    def _trim_history(self, max_messages: int = 20) -> None:
        """Обрезать историю диалога, сохраняя системный промпт."""
        if len(self._conversation_history) > max_messages:
            system_msg = self._conversation_history[0]
            self._conversation_history = [system_msg] + self._conversation_history[-(max_messages - 1):]

    def reset_conversation(self) -> None:
        """Сбросить историю диалога."""
        self._conversation_history = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        logger.info("История диалога сброшена")

    def shutdown(self) -> None:
        """Освободить ресурсы."""
        self._client = None
        self._conversation_history.clear()
        logger.info("Ollama LLM остановлен")
