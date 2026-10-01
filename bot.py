"""Простой Telegram-помощник с долговременной памятью в Pinecone."""

from __future__ import annotations

import os
import time
from collections import defaultdict

import telebot
from dotenv import load_dotenv
from telebot import apihelper
from telebot.types import Message

from pinecone_manager import PineconeManager

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "").strip()
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
SHORT_TERM_TURNS = 8

SYSTEM_PROMPT = """Ты персональный помощник в Telegram. Отвечай по-русски, кратко и по делу.

Ниже может быть долговременная память о собеседнике. Личные факты о человеке
бери только оттуда и не выдумывай биографию, которой в памяти нет.

На обычные вопросы (что такое Pinecone, как устроена твоя память, технологии)
отвечай как помощник с общими знаниями. Если спросят — можно объяснить, что
факты о пользователе хранятся в векторной базе Pinecone.
"""

if not TELEGRAM_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_TOKEN. Укажите токен бота в файле .env.")

if TELEGRAM_PROXY:
    apihelper.proxy = {"http": TELEGRAM_PROXY, "https": TELEGRAM_PROXY}

apihelper.CONNECT_TIMEOUT = 60
apihelper.READ_TIMEOUT = 60

memory = PineconeManager()
memory.ensure_index()

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)
recent_chats: dict[str, list[dict[str, str]]] = defaultdict(list)


def _namespace(user_id: int | str) -> str:
    return f"user_{user_id}"


def _telegram_user(message: Message):
    """Возвращает (user_id, user). Если from_user нет — берём chat.id."""
    user = message.from_user
    if user is not None:
        return str(user.id), user
    return str(message.chat.id), None


def _display_name(user) -> str:
    if user is None:
        return "друг"
    return (user.first_name or user.username or "").strip() or "друг"


def _user_metadata(user, extra: dict | None = None) -> dict:
    """Собирает только заполненные поля профиля Telegram."""
    payload: dict = dict(extra or {})
    if user is None:
        return payload
    payload["user_id"] = str(user.id)
    if user.first_name:
        payload["first_name"] = user.first_name
    if user.last_name:
        payload["last_name"] = user.last_name
    if user.username:
        payload["username"] = user.username
    if user.language_code:
        payload["language_code"] = user.language_code
    return payload


def _memory_texts(matches: list[dict]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for match in matches:
        text = (match.get("metadata") or {}).get("text")
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return texts


def _ask_model(messages: list[dict[str, str]]) -> str:
    client = memory.openai_client
    if client is None:
        raise RuntimeError("Не задан OPENAI_API_KEY. Укажите ключ в файле .env.")
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.4,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def _remember_user_message(user_id: str, user, text: str) -> None:
    result = memory.upsert_text(
        text,
        metadata=_user_metadata(user, {"role": "user", "user_id": user_id}),
        namespace=_namespace(user_id),
    )
    print(
        f"memory: action={result.get('action')} "
        f"similarity={result.get('similarity_score')} "
        f"text={text!r}"
    )


def _append_recent(user_id: str, role: str, text: str) -> None:
    history = recent_chats[user_id]
    history.append({"role": role, "content": text})
    if len(history) > SHORT_TERM_TURNS * 2:
        del history[: len(history) - SHORT_TERM_TURNS * 2]


@bot.message_handler(commands=["start"])
def handle_start(message: Message) -> None:
    _, user = _telegram_user(message)
    bot.reply_to(
        message,
        f"Привет, {_display_name(user)}! Я помощник с долговременной памятью.\n"
        "Можно просто писать — я запомню важное о вас.\n"
        "Команда /forget стирает вашу память.",
    )


@bot.message_handler(commands=["forget"])
def handle_forget(message: Message) -> None:
    user_id, _ = _telegram_user(message)
    memory.delete(delete_all=True, namespace=_namespace(user_id))
    recent_chats.pop(user_id, None)
    bot.reply_to(message, "Память о вас очищена. Можем начать заново.")


@bot.message_handler(content_types=["text"])
def handle_text(message: Message) -> None:
    text = (message.text or "").strip()
    if not text or text.startswith("/"):
        return

    user_id, user = _telegram_user(message)
    namespace = _namespace(user_id)

    try:
        matches = memory.query_by_text(text, top_k=5, namespace=namespace)
        known = _memory_texts(matches)
        system = SYSTEM_PROMPT
        if known:
            system += "\n\nИзвестно о пользователе:\n- " + "\n- ".join(known)

        messages = [{"role": "system", "content": system}]
        messages.extend(recent_chats[user_id])
        messages.append({"role": "user", "content": text})

        answer = _ask_model(messages)
        if not answer:
            answer = "Не получилось сформулировать ответ. Попробуйте ещё раз."

        _remember_user_message(user_id, user, text)
        _append_recent(user_id, "user", text)
        _append_recent(user_id, "assistant", answer)
        bot.reply_to(message, answer[:4000])
    except Exception as exc:
        bot.reply_to(
            message,
            "Не удалось обработать сообщение. Проверьте ключи в .env и попробуйте снова.",
        )
        print(f"Ошибка обработки сообщения: {exc}")


if __name__ == "__main__":
    print("Бот запущен. Ожидаю сообщения в Telegram...")
    print("Если здесь появляются таймауты api.telegram.org — это сеть, не Pinecone.")
    while True:
        try:
            bot.infinity_polling(
                skip_pending=False,
                timeout=60,
                long_polling_timeout=50,
                interval=3,
            )
        except KeyboardInterrupt:
            print("Бот остановлен.")
            break
        except Exception as exc:
            print(f"Нет связи с Telegram ({exc}). Повтор через 5 секунд...")
            time.sleep(5)
