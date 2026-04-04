import psycopg2
import os

import logging
import joblib
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram import Router, F

from aiogram.types import BotCommand

# Настройка логирования
logging.basicConfig(level=logging.INFO)

model = joblib.load("model.pkl")



API_TOKEN = 'REMOVED'
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)



DB_NAME = os.getenv("DB_NAME", "scoring_db")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "password")
DB_HOST = os.getenv("DB_HOST", "db")

def init_db():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id SERIAL PRIMARY KEY,
            telegram_id BIGINT UNIQUE,
            first_name TEXT,
            last_name TEXT,
            age INTEGER,
            income REAL,
            approved BOOLEAN
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()



class Form(StatesGroup):
    first_name = State()
    last_name = State()
    age = State()
    income = State()
    education = State()
    work = State()
    car = State()


@router.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    user_id = message.from_user.id

    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
        cursor = conn.cursor()

        cursor.execute("SELECT approved FROM applications WHERE telegram_id = %s", (user_id,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result:
            is_approved = result[0]
            if is_approved:
                await message.answer("Вы уже подавали заявку. Статус: ОДОБРЕНО! 🎉")
            else:
                await message.answer("Вы уже подавали заявку. Статус: Подобрана альтернативная карта. 😔")
            return

    except Exception as e:
        logging.error(f"Ошибка проверки БД: {e}")

    await message.answer("Добро пожаловать! Давайте проверим ваш кредитный лимит. Введите ваше Имя:")
    await state.set_state(Form.first_name)

@router.message(Form.first_name)
async def process_name(message: types.Message, state: FSMContext):
    await state.update_data(first_name=message.text)
    await message.answer("Введите вашу Фамилию:")
    await state.set_state(Form.last_name)

@router.message(Form.last_name)
async def process_surname(message: types.Message, state: FSMContext):
    await state.update_data(last_name=message.text)
    await message.answer("Сколько вам лет?")
    await state.set_state(Form.age)

# Обработка возраста
@router.message(Form.age)
async def process_age(message: types.Message, state: FSMContext):
    await state.update_data(age=int(message.text))
    await message.answer("Какой у вас доход в тысячах рублей?")
    await state.set_state(Form.income)


# Обработка дохода
@router.message(Form.income)
async def process_income(message: types.Message, state: FSMContext):
    await state.update_data(income=float(message.text))
    kb = [
        [types.KeyboardButton(text="Да"), types.KeyboardButton(text="Нет")]
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)
    await message.answer("Есть ли у вас высшее образование?", reply_markup=keyboard)
    await state.set_state(Form.education)


# Обработка образования
@router.message(Form.education)
async def process_edu(message: types.Message, state: FSMContext):
    val = 1 if message.text.lower() == "да" else 0
    await state.update_data(education=val)
    await message.answer("Есть ли у вас стабильная работа?")
    await state.set_state(Form.work)


# Обработка работы
@router.message(Form.work)
async def process_work(message: types.Message, state: FSMContext):
    val = 1 if message.text.lower() == "да" else 0
    await state.update_data(work=val)
    await message.answer("Есть ли у вас автомобиль?")
    await state.set_state(Form.car)


# Итоговый расчет
@router.message(Form.car)
async def process_car(message: types.Message, state: FSMContext):
    val = 1 if message.text.lower() == "да" else 0
    user_data = await state.get_data()

    features = [
        user_data['age'],
        user_data['income'],
        user_data['education'],
        user_data['work'],
        val
    ]

    prediction = model.predict([features])[0]
    approved = not prediction





    user_id = message.from_user.id
    user_data = await state.get_data()
    approved_status = True if approved else False

    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            host=os.getenv("DB_HOST")
        )
        cursor = conn.cursor()

        cursor.execute('''
                INSERT INTO applications (telegram_id, first_name, last_name, age, income, approved)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (
            user_id,
            user_data['first_name'],
            user_data['last_name'],
            user_data['age'],
            user_data['income'],
            approved_status
        ))

        conn.commit()
        cursor.close()
        conn.close()
        logging.info("Данные успешно сохранены в PostgreSQL")

    except Exception as e:
        logging.error(f"Ошибка при сохранении в БД: {e}")
        await message.answer("Произошла ошибка при сохранении данных, но результат готов.")





    if approved:
        await message.answer("🎉 Поздравляем! Ваша заявка одобрена!", reply_markup=types.ReplyKeyboardRemove())
    else:
        await message.answer("😔 К сожалению, по основной карте отказ, но мы подобрали альтернативу.",
                             reply_markup=types.ReplyKeyboardRemove())

    await state.clear()



async def set_main_menu(bot: Bot):
    main_menu_commands = [
        BotCommand(command='/start', description='Запустить проверку лимита'),
        BotCommand(command='/help', description='Как это работает?'),
        BotCommand(command='/reset', description='Сбросить мои данные')
    ]
    await bot.set_my_commands(main_menu_commands)

async def main():
    init_db()

    await set_main_menu(bot)

    await dp.start_polling(bot)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())