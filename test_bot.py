import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY не найден в .env файле!")
    exit()

client = OpenAI(api_key=OPENAI_API_KEY)

print("🧪 Тестируем подключение к OpenAI...")
print("-" * 40)

try:
    # Тест простого запроса
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты - помощник по проверке заданий."},
            {"role": "user", "content": "Сколько будет 2+2?"}
        ],
        max_tokens=50
    )

    print(f"✅ Подключение успешно!")
    print(f"📝 Ответ GPT: {response.choices[0].message.content}")

    # Тест доступных моделей
    print("\n📊 Доступные модели:")
    models = client.models.list()
    gpt_models = [m.id for m in models.data if 'gpt' in m.id]
    for model in sorted(gpt_models)[:5]:  # Показываем первые 5
        print(f"  • {model}")

except Exception as e:
    print(f"❌ Ошибка подключения к OpenAI: {e}")
    print("\n🔧 Возможные причины:")
    print("1. Неверный API ключ")
    print("2. Нет баланса на счету")
    print("3. Проблемы с сетью")
    print("\n💡 Проверьте ключ на platform.openai.com")