"""Интеграция с облачными LLM API (OpenAI и совместимые)."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from sip_ai_client.config import CloudAPIConfig


class CloudLLM:
    """Клиент для облачных LLM API (OpenAI, совместимые API)."""

    def __init__(self, config: CloudAPIConfig):
        self.config = config
        self._client = None
        self._conversation_history: list[dict[str, str]] = []

    def initialize(self) -> None:
        """Инициализировать клиент облачного API."""
        from openai import OpenAI

        kwargs: dict = {"api_key": self.config.api_key}
        if self.config.api_base:
            kwargs["base_url"] = self.config.api_base

        self._client = OpenAI(**kwargs)
        self._conversation_history = [
            {"role": "system", "content": self.config.system_prompt}
        ]

        logger.info(
            "Cloud LLM инициализирован: provider={}, model={}",
            self.config.provider,
            self.config.model,
        )

    def generate_response(self, user_text: str) -> Optional[str]:
        """Сгенерировать ответ на текст пользователя.

        Args:
            user_text: Текст от пользователя.

        Returns:
            Ответ модели или None.
        """
        if self._client is None:
            logger.error("Cloud LLM клиент не инициализирован")
            return None

        self._conversation_history.append({"role": "user", "content": user_text})

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=self._conversation_history,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            assistant_text = response.choices[0].message.content
            if assistant_text:
                assistant_text = assistant_text.strip()
            else:
                assistant_text = ""

            self._conversation_history.append(
                {"role": "assistant", "content": assistant_text}
            )

            self._trim_history()

            logger.info("Cloud LLM ответ: '{}'", assistant_text[:100])
            return assistant_text

        except Exception as e:
            logger.error("Ошибка Cloud LLM: {}", e)
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
        logger.info("Cloud LLM остановлен")
