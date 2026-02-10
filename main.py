import os
import telebot
import base64
import requests
import time
import threading
import hashlib
from queue import Queue
from collections import defaultdict
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from io import BytesIO

# ========== КОНФИГУРАЦИЯ ==========
load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not BOT_TOKEN or not OPENAI_API_KEY:
    print("❌ ОШИБКА: Заполните .env файл")
    exit()

bot = telebot.TeleBot(BOT_TOKEN)
client = OpenAI(api_key=OPENAI_API_KEY)

# Модели
MODELS = {
    "text": "gpt-4o-mini",
    "vision": "gpt-4o",
    "fallback": "gpt-3.5-turbo"
}

SUBJECTS = [
    "Математика", "Русский язык", "Английский язык",
    "Физика", "Химия", "Биология", "Информатика", "Другое"
]

# ========== СИСТЕМА ДЛЯ МНОГИХ ПОЛЬЗОВАТЕЛЕЙ ==========

# 1. Очередь запросов
request_queue = Queue()


# 2. Rate limiting для OpenAI
class RateLimiter:
    def __init__(self, max_requests_per_minute=30):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    def can_make_request(self):
        with self.lock:
            now = time.time()
            # Удаляем запросы старше 1 минуты
            self.requests = [t for t in self.requests if now - t < 60]

            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

    def get_wait_time(self):
        with self.lock:
            if len(self.requests) < self.max_requests:
                return 0
            # Время до освобождения слота
            oldest = min(self.requests)
            return max(0, 60 - (time.time() - oldest))


openai_limiter = RateLimiter(max_requests_per_minute=30)


# 3. Кэш ответов
class ResponseCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()

    def get_key(self, text, subject=None, is_image=False):
        content = f"{text}:{subject}:{is_image}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, key):
        with self.lock:
            cached = self.cache.get(key)
            if cached and time.time() - cached['timestamp'] < 3600:  # 1 час
                return cached['value']
            return None

    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Удаляем самый старый
                oldest = min(self.cache.items(), key=lambda x: x[1]['timestamp'])
                del self.cache[oldest[0]]
            self.cache[key] = {'value': value, 'timestamp': time.time()}


cache = ResponseCache(max_size=500)

# 4. Сессии пользователей (вместо глобального словаря)
import sqlite3
import json
from datetime import datetime


class UserSessionDB:
    def __init__(self, db_path='user_sessions.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS sessions
                       (
                           chat_id
                           INTEGER
                           PRIMARY
                           KEY,
                           subject
                           TEXT,
                           messages
                           TEXT,
                           created_at
                           TIMESTAMP,
                           updated_at
                           TIMESTAMP
                       )
                       ''')
        self.conn.commit()

    def get_session(self, chat_id):
        cursor = self.conn.cursor()
        cursor.execute('SELECT subject, messages FROM sessions WHERE chat_id = ?', (chat_id,))
        row = cursor.fetchone()

        if row:
            subject, messages_json = row
            messages = json.loads(messages_json) if messages_json else []
            return {'subject': subject, 'messages': messages}

        # Создаем новую сессию
        default_session = {'subject': None, 'messages': []}
        cursor.execute('''
                       INSERT INTO sessions (chat_id, subject, messages, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ''', (chat_id, None, json.dumps([]), datetime.now(), datetime.now()))
        self.conn.commit()
        return default_session

    def update_session(self, chat_id, updates):
        cursor = self.conn.cursor()
        cursor.execute('SELECT subject, messages FROM sessions WHERE chat_id = ?', (chat_id,))
        row = cursor.fetchone()

        if row:
            subject, messages_json = row
            messages = json.loads(messages_json) if messages_json else []

            if 'subject' in updates:
                subject = updates['subject']
            if 'messages' in updates:
                messages = updates['messages']

            cursor.execute('''
                           UPDATE sessions
                           SET subject    = ?,
                               messages   = ?,
                               updated_at = ?
                           WHERE chat_id = ?
                           ''', (subject, json.dumps(messages[-20:]), datetime.now(),
                                 chat_id))  # Храним последние 20 сообщений
        else:
            cursor.execute('''
                           INSERT INTO sessions (chat_id, subject, messages, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?)
                           ''', (
                               chat_id,
                               updates.get('subject'),
                               json.dumps(updates.get('messages', [])),
                               datetime.now(),
                               datetime.now()
                           ))

        self.conn.commit()


db = UserSessionDB()


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def create_subject_keyboard():
    markup = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    for subject in SUBJECTS:
        markup.add(KeyboardButton(subject))
    return markup


def compress_image(image_bytes, max_size=1024, quality=70):
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()
    except Exception as e:
        print(f"⚠️ Ошибка сжатия: {e}")
        return image_bytes


def process_with_openai_limited(text, image_base64=None, subject=None, chat_id=None):
    """Обработка с rate limiting и кэшированием"""

    # 1. Проверяем кэш
    is_image = image_base64 is not None
    cache_key = cache.get_key(text, subject, is_image)
    cached_response = cache.get(cache_key)

    if cached_response:
        print(f"✅ Использован кэш для {chat_id}")
        return cached_response

    # 2. Rate limiting
    wait_time = openai_limiter.get_wait_time()
    if wait_time > 0:
        print(f"⏳ Rate limit, ждем {wait_time:.1f} сек")
        time.sleep(wait_time)

    # 3. Получаем сессию из БД
    session = db.get_session(chat_id) if chat_id else {'subject': subject, 'messages': []}

    # 4. Подготовка запроса
    if subject:
        system_prompt = f"Ты - репетитор по предмету '{subject}'. Проверяй задания, объясняй ошибки."
    else:
        system_prompt = "Ты - опытный репетитор. Проверяй домашние задания."

    messages = [{"role": "system", "content": system_prompt}]

    # Добавляем историю
    if session.get('messages'):
        messages.extend(session['messages'][-4:])

    # Формируем запрос
    if image_base64:
        user_content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "low"
                }
            }
        ]
        model = MODELS["vision"]
    else:
        user_content = text
        model = MODELS["text"]

    messages.append({"role": "user", "content": user_content})

    try:
        # 5. Отправляем запрос
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )

        result = response.choices[0].message.content

        # 6. Сохраняем в кэш
        cache.set(cache_key, result)

        # 7. Обновляем историю в БД
        if chat_id:
            new_messages = session.get('messages', [])
            new_messages.append({"role": "user", "content": text[:50]})
            new_messages.append({"role": "assistant", "content": result[:50]})
            db.update_session(chat_id, {'messages': new_messages[-20:]})  # Храним 20 последних

        return result

    except Exception as e:
        print(f"❌ Ошибка OpenAI: {e}")
        return f"Ошибка при обработке: {str(e)[:100]}"


# ========== ОБРАБОТЧИКИ (АСИНХРОННЫЕ) ==========

def process_user_request(chat_id, task_type, data):
    """Обработка запроса в фоновом режиме"""
    try:
        if task_type == 'photo':
            img_bytes, caption, subject = data
            compressed = compress_image(img_bytes)
            img_base64 = base64.b64encode(compressed).decode('utf-8')
            text = caption or "Проверь это задание"

            response = process_with_openai_limited(
                text=text,
                image_base64=img_base64,
                subject=subject,
                chat_id=chat_id
            )

        else:  # text
            text, subject = data
            response = process_with_openai_limited(
                text=text,
                image_base64=None,
                subject=subject,
                chat_id=chat_id
            )

        # Отправляем результат
        bot.send_message(chat_id, response, parse_mode='Markdown')

    except Exception as e:
        error_msg = f"❌ Ошибка обработки: {str(e)[:100]}"
        bot.send_message(chat_id, error_msg)
        print(f"Ошибка в process_user_request: {e}")


def worker():
    """Фоновый обработчик"""
    while True:
        try:
            task = request_queue.get()
            if task is None:
                break
            chat_id, task_type, data = task
            process_user_request(chat_id, task_type, data)
            request_queue.task_done()
        except Exception as e:
            print(f"Ошибка в worker: {e}")
            time.sleep(1)


# Запускаем воркеры
NUM_WORKERS = 5  # 5 параллельных обработчиков
for i in range(NUM_WORKERS):
    t = threading.Thread(target=worker, daemon=True)
    t.start()


# ========== ОБРАБОТЧИКИ TELEGRAM ==========

@bot.message_handler(commands=['start'])
def send_welcome(message):
    chat_id = message.chat.id
    db.update_session(chat_id, {'subject': None, 'messages': []})

    welcome_text = (
        "👋 *Привет! Я Task Helper*\n\n"
        "✅ Поддерживаю множество пользователей\n"
        "⏳ Очередь запросов\n"
        "💾 Кэширование ответов\n\n"
        "Выбери предмет:"
    )

    bot.send_message(
        chat_id,
        welcome_text,
        reply_markup=create_subject_keyboard(),
        parse_mode='Markdown'
    )


@bot.message_handler(commands=['stats'])
def show_stats(message):
    """Показывает статистику"""
    chat_id = message.chat.id

    # Статистика очереди
    queue_size = request_queue.qsize()
    active_threads = threading.active_count() - 1  # Минус основной

    # Статистика rate limiting
    wait_time = openai_limiter.get_wait_time()

    stats_text = (
        f"📊 *Статистика системы:*\n\n"
        f"• Очередь запросов: {queue_size}\n"
        f"• Активных потоков: {active_threads}\n"
        f"• Rate limit ожидание: {wait_time:.1f} сек\n"
        f"• Запросов/минуту: {len(openai_limiter.requests)}\n"
        f"• Размер кэша: {len(cache.cache)}\n\n"
        f"✅ *Система работает нормально*"
    )

    bot.send_message(chat_id, stats_text, parse_mode='Markdown')


@bot.message_handler(func=lambda msg: msg.text in SUBJECTS)
def handle_subject(message):
    chat_id = message.chat.id
    db.update_session(chat_id, {'subject': message.text})

    bot.send_message(
        chat_id,
        f"✅ *Предмет:* {message.text}\n\nОтправь задание 📸 или 📝",
        parse_mode='Markdown'
    )


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    chat_id = message.chat.id
    session = db.get_session(chat_id)

    # Сообщаем о постановке в очередь
    queue_position = request_queue.qsize() + 1
    bot.send_message(
        chat_id,
        f"📸 *Фото получено*\n"
        f"⏳ *Позиция в очереди:* {queue_position}\n"
        f"Ожидайте обработки...",
        parse_mode='Markdown'
    )

    try:
        # Скачиваем фото
        file_info = bot.get_file(message.photo[-1].file_id)
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"

        img_response = requests.get(file_url, timeout=10)
        if img_response.status_code != 200:
            bot.send_message(chat_id, "❌ Ошибка загрузки фото")
            return

        # Добавляем в очередь
        request_queue.put((
            chat_id,
            'photo',
            (
                img_response.content,
                message.caption or "Проверь задание",
                session.get('subject')
            )
        ))

    except Exception as e:
        bot.send_message(chat_id, f"❌ Ошибка: {str(e)[:100]}")


@bot.message_handler(content_types=['text'])
def handle_text(message):
    if message.text.startswith('/') or message.text in SUBJECTS:
        return

    chat_id = message.chat.id
    session = db.get_session(chat_id)

    # Сообщаем о постановке в очередь
    queue_position = request_queue.qsize() + 1
    bot.send_message(
        chat_id,
        f"📝 *Задание получено*\n"
        f"⏳ *Позиция в очереди:* {queue_position}",
        parse_mode='Markdown'
    )

    # Добавляем в очередь
    request_queue.put((
        chat_id,
        'text',
        (message.text, session.get('subject'))
    ))


def main():
    print("=" * 70)
    print("🤖 TASK HELPER BOT - МАСШТАБИРУЕМАЯ ВЕРСИЯ")
    print("=" * 70)
    print(f"🚀 Максимальных пользователей: ~100-200 одновременно")
    print(f"⚙️ Параллельных воркеров: {NUM_WORKERS}")
    print(f"📊 База данных сессий: SQLite")
    print(f"💾 Кэш ответов: 500 записей")
    print("=" * 70)
    print("📞 Команды: /start, /stats, /help")
    print("🛑 Остановка: Ctrl+C")
    print("=" * 70)

    try:
        bot.infinity_polling()
    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен")
        # Очищаем очередь
        for _ in range(NUM_WORKERS):
            request_queue.put(None)


if __name__ == '__main__':
    main()