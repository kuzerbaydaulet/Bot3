# database.py
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import hashlib


class UserDatabase:
    def __init__(self, db_path='users.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()

        # Таблица пользователей
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS users
                       (
                           user_id
                           INTEGER
                           PRIMARY
                           KEY,
                           chat_id
                           INTEGER
                           UNIQUE,
                           username
                           TEXT,
                           first_name
                           TEXT,
                           last_name
                           TEXT,
                           tariff
                           TEXT
                           DEFAULT
                           'free',
                           requests_today
                           INTEGER
                           DEFAULT
                           0,
                           requests_total
                           INTEGER
                           DEFAULT
                           0,
                           subscription_until
                           TIMESTAMP,
                           created_at
                           TIMESTAMP,
                           updated_at
                           TIMESTAMP
                       )
                       ''')

        # Таблица платежей
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS payments
                       (
                           payment_id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           INTEGER,
                           amount
                           REAL,
                           currency
                           TEXT
                           DEFAULT
                           'RUB',
                           tariff
                           TEXT,
                           period_days
                           INTEGER,
                           payment_method
                           TEXT,
                           payment_date
                           TIMESTAMP,
                           status
                           TEXT
                           DEFAULT
                           'pending',
                           FOREIGN
                           KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           user_id
                       )
                           )
                       ''')

        # Таблица запросов (для статистики и лимитов)
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS requests
                       (
                           request_id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           INTEGER,
                           request_type
                           TEXT, -- text, photo, voice
                           tokens_used
                           INTEGER,
                           model_used
                           TEXT,
                           timestamp
                           TIMESTAMP,
                           cost
                           REAL
                           DEFAULT
                           0,
                           FOREIGN
                           KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           user_id
                       )
                           )
                       ''')

        # Таблица промокодов
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS promo_codes
                       (
                           code
                           TEXT
                           PRIMARY
                           KEY,
                           discount_percent
                           INTEGER
                           DEFAULT
                           0,
                           tariff_upgrade
                           TEXT,
                           valid_until
                           TIMESTAMP,
                           uses_left
                           INTEGER
                           DEFAULT
                           1,
                           used_count
                           INTEGER
                           DEFAULT
                           0
                       )
                       ''')

        self.conn.commit()

    def get_or_create_user(self, chat_id, username=None, first_name=None, last_name=None):
        cursor = self.conn.cursor()

        cursor.execute(
            'SELECT * FROM users WHERE chat_id = ?',
            (chat_id,)
        )
        user = cursor.fetchone()

        if user:
            # Возвращаем существующего пользователя
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, user))

        # Создаем нового пользователя
        cursor.execute('''
                       INSERT INTO users (chat_id, username, first_name, last_name,
                                          created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', (
                           chat_id, username, first_name, last_name,
                           datetime.now(), datetime.now()
                       ))

        self.conn.commit()

        # Получаем созданного пользователя
        cursor.execute('SELECT * FROM users WHERE chat_id = ?', (chat_id,))
        user = cursor.fetchone()
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, user))

    def update_user_tariff(self, chat_id, tariff, subscription_days=30):
        cursor = self.conn.cursor()

        if tariff == 'free':
            subscription_until = None
        else:
            subscription_until = datetime.now() + timedelta(days=subscription_days)

        cursor.execute('''
                       UPDATE users
                       SET tariff             = ?,
                           subscription_until = ?,
                           updated_at         = ?
                       WHERE chat_id = ?
                       ''', (tariff, subscription_until, datetime.now(), chat_id))

        self.conn.commit()
        return True

    def can_make_request(self, chat_id, request_type='text'):
        """Проверяет, может ли пользователь сделать запрос"""
        cursor = self.conn.cursor()

        cursor.execute('''
                       SELECT tariff, requests_today, subscription_until
                       FROM users
                       WHERE chat_id = ?
                       ''', (chat_id,))

        result = cursor.fetchone()
        if not result:
            return False, "Пользователь не найден"

        tariff, requests_today, subscription_until = result

        # Проверяем подписку
        if tariff in ['pro', 'premium'] and subscription_until:
            if datetime.now() > datetime.fromisoformat(subscription_until):
                # Подписка истекла
                cursor.execute('''
                               UPDATE users
                               SET tariff             = 'free',
                                   subscription_until = NULL
                               WHERE chat_id = ?
                               ''', (chat_id,))
                self.conn.commit()
                tariff = 'free'

        # Лимиты по тарифам
        limits = {
            'free': {
                'daily': 5,
                'types': ['text'],  # Только текст
                'model': 'gpt-3.5-turbo'
            },
            'pro': {
                'daily': 100,  # Мягкий лимит
                'types': ['text', 'photo'],
                'model': 'gpt-4o-mini'
            },
            'premium': {
                'daily': 1000,
                'types': ['text', 'photo', 'voice'],
                'model': 'gpt-4o'
            }
        }

        tariff_limit = limits.get(tariff, limits['free'])

        # Проверяем тип запроса
        if request_type not in tariff_limit['types']:
            return False, f"Тариф '{tariff}' не поддерживает {request_type}"

        # Проверяем дневной лимит
        if requests_today >= tariff_limit['daily']:
            return False, f"Достигнут дневной лимит ({tariff_limit['daily']} запросов)"

        return True, tariff_limit['model']

    def log_request(self, chat_id, request_type, tokens_used, model_used, cost=0):
        """Логирует запрос пользователя"""
        cursor = self.conn.cursor()

        # Получаем user_id
        cursor.execute('SELECT user_id FROM users WHERE chat_id = ?', (chat_id,))
        user_result = cursor.fetchone()
        if not user_result:
            return False

        user_id = user_result[0]

        # Логируем запрос
        cursor.execute('''
                       INSERT INTO requests
                           (user_id, request_type, tokens_used, model_used, timestamp, cost)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', (user_id, request_type, tokens_used, model_used, datetime.now(), cost))

        # Обновляем счетчики
        cursor.execute('''
                       UPDATE users
                       SET requests_today = requests_today + 1,
                           requests_total = requests_total + 1,
                           updated_at     = ?
                       WHERE chat_id = ?
                       ''', (datetime.now(), chat_id))

        self.conn.commit()
        return True

    def reset_daily_limits(self):
        """Сбрасывает дневные лимиты (вызывать раз в день)"""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE users SET requests_today = 0')
        self.conn.commit()
        print("✅ Дневные лимиты сброшены")

    def get_user_stats(self, chat_id):
        """Получает статистику пользователя"""
        cursor = self.conn.cursor()

        cursor.execute('''
                       SELECT u.username,
                              u.first_name,
                              u.tariff,
                              u.requests_today,
                              u.requests_total,
                              u.subscription_until,
                              (SELECT COUNT(*)
                               FROM requests r
                               WHERE r.user_id = u.user_id AND r.request_type = 'photo') as photo_requests,
                              (SELECT COUNT(*)
                               FROM requests r
                               WHERE r.user_id = u.user_id AND r.request_type = 'text')  as text_requests
                       FROM users u
                       WHERE u.chat_id = ?
                       ''', (chat_id,))

        result = cursor.fetchone()
        if not result:
            return None

        columns = ['username', 'first_name', 'tariff', 'requests_today',
                   'requests_total', 'subscription_until', 'photo_requests', 'text_requests']

        stats = dict(zip(columns, result))

        # Форматируем дату подписки
        if stats['subscription_until']:
            sub_date = datetime.fromisoformat(stats['subscription_until'])
            stats['subscription_until_formatted'] = sub_date.strftime('%d.%m.%Y')
            stats['days_left'] = (sub_date - datetime.now()).days
        else:
            stats['subscription_until_formatted'] = None
            stats['days_left'] = 0

        return stats

    def create_payment(self, chat_id, amount, tariff, period_days, payment_method='manual'):
        """Создает запись о платеже"""
        cursor = self.conn.cursor()

        # Получаем user_id
        cursor.execute('SELECT user_id FROM users WHERE chat_id = ?', (chat_id,))
        user_result = cursor.fetchone()
        if not user_result:
            return None

        user_id = user_result[0]

        # Создаем платеж
        cursor.execute('''
                       INSERT INTO payments
                       (user_id, amount, tariff, period_days, payment_method, payment_date, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ''', (user_id, amount, tariff, period_days, payment_method, datetime.now(), 'completed'))

        payment_id = cursor.lastrowid

        # Обновляем тариф пользователя
        self.update_user_tariff(chat_id, tariff, period_days)

        self.conn.commit()
        return payment_id


# -- Таблица пользователей
# CREATE TABLE users (
#     user_id INTEGER PRIMARY KEY,
#     telegram_id INTEGER UNIQUE,
#     current_tariff TEXT DEFAULT 'free',
#     requests_today INTEGER DEFAULT 0,
#     subscription_ends_at TIMESTAMP,
#     created_at TIMESTAMP
# );
#
# -- Таблица платежей
# CREATE TABLE payments (
#     payment_id INTEGER PRIMARY KEY AUTOINCREMENT,
#     user_id INTEGER,
#     amount REAL,
#     currency TEXT DEFAULT 'KZT',
#     tariff TEXT,
#     provider TEXT,
#     status TEXT,
#     invoice_payload TEXT, -- Для идентификации платежа в Telegram
#     created_at TIMESTAMP,
#     FOREIGN KEY (user_id) REFERENCES users(user_id)
# );
#
# -- Тарифы для справки
# CREATE TABLE tariffs (
#     tariff_id TEXT PRIMARY KEY, -- 'free', 'pro', 'premium'
#     name TEXT,
#     price_monthly_kzt REAL,
#     requests_per_day INTEGER,
#     features TEXT -- JSON с описанием возможностей
# );