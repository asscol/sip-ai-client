# SIP AI Client

Голосовой ИИ-ассистент, работающий по SIP-протоколу. Принимает телефонные звонки, распознаёт речь, генерирует ответы с помощью LLM и озвучивает их.

## Возможности

- **SIP-протокол** — подключение к любому SIP-серверу (Asterisk, FreeSWITCH и др.)
- **Whisper STT** — распознавание русской речи (faster-whisper)
- **Piper TTS** — синтез речи голосом Ruslan (русский)
- **Ollama** — локальные LLM модели >19B параметров (llama3.1:70b и др.)
- **Cloud API** — облачные LLM (OpenAI GPT-4o и совместимые API)
- **Автоматический VAD** — детекция голосовой активности
- **Docker** — готовая контейнеризация

## Архитектура

```
Телефон → SIP-сервер → SIP AI Client
                           ├── Whisper (STT: речь → текст)
                           ├── Ollama / Cloud API (LLM: текст → ответ)
                           └── Piper TTS (TTS: ответ → речь) → SIP → Телефон
```

## Быстрый старт

### 1. Установка

```bash
# Клонировать репозиторий
git clone https://github.com/asscol/sip-ai-client.git
cd sip-ai-client

# Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# Установить зависимости
pip install -e .
```

### 2. Установка Piper TTS

```bash
# Linux x86_64
curl -sSL https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
  | sudo tar -xz -C /usr/local/bin/ --strip-components=1
```

### 3. Установка Ollama (для локальных моделей)

```bash
curl -fsSL https://ollama.com/install.sh | sh

# Загрузить модель >19B параметров
ollama pull llama3.1:70b
```

### 4. Конфигурация

```bash
cp config.yaml.example config.yaml
# Отредактируйте config.yaml — укажите SIP-сервер, учётные данные и т.д.
```

### 5. Запуск

```bash
# С Ollama (локальная LLM)
sip-ai-client --config config.yaml --llm-provider ollama

# С облачным API (OpenAI)
sip-ai-client --config config.yaml --llm-provider openai

# С указанием SIP-сервера через CLI
sip-ai-client --sip-server 192.168.1.100 --sip-username 1001 --sip-password secret
```

## Docker

```bash
# Скопировать конфиг
cp config.yaml.example config.yaml

# Запуск с Ollama
docker-compose up -d

# Только SIP-клиент (Ollama на хосте)
docker build -t sip-ai-client .
docker run --network host -v ./config.yaml:/app/config.yaml sip-ai-client
```

## Конфигурация

### Через config.yaml

См. [`config.yaml.example`](config.yaml.example) — подробные комментарии к каждому параметру.

### Через переменные окружения

Все параметры можно задать через ENV с префиксом `SIP_AI_`:

```bash
export SIP_AI_LLM_PROVIDER=ollama
export SIP_AI_SIP__SERVER=192.168.1.100
export SIP_AI_SIP__USERNAME=1001
export SIP_AI_OLLAMA__MODEL=llama3.1:70b
export SIP_AI_CLOUD_API__API_KEY=sk-your-key
```

## Провайдеры LLM

### Ollama (локальные модели)

Для работы с моделями >19B параметров рекомендуется:

| Модель | Параметры | RAM | Описание |
|--------|-----------|-----|----------|
| `llama3.1:70b` | 70B | ~40 GB | Лучшее качество |
| `qwen2.5:32b` | 32B | ~20 GB | Хорошее качество, быстрее |
| `gemma2:27b` | 27B | ~16 GB | Компактная, быстрая |
| `mixtral:8x22b` | 141B | ~80 GB | MoE, высокое качество |
| `command-r:35b` | 35B | ~20 GB | Хорошо для диалога |

### Cloud API (облачные)

Поддерживаются OpenAI и совместимые API:

```yaml
cloud_api:
  provider: "openai"
  api_key: "sk-your-key"
  model: "gpt-4o"
  # Для совместимых API (например, Together AI):
  # api_base: "https://api.together.xyz/v1"
```

## Требования

- Python 3.10+
- SIP-сервер (Asterisk, FreeSWITCH)
- Ollama (для локальных моделей) или API-ключ облачного провайдера
- Piper TTS (для синтеза речи)
- ~4 GB RAM (для Whisper large-v3)
- GPU рекомендуется для Whisper и Ollama

## Лицензия

MIT
