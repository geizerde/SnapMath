import logging
import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
import cv2
import numpy as np
from typing import List, Any
import joblib
import os

API_TOKEN = "7642880568:AAGYiiI_GoTTx-aq5V0iWRXrK-56xhp1m_Q"
MODELS = {
    "rfc": joblib.load('./models/rfc_model_2000_pca.pkl'),
    "svc": joblib.load('./models/svc_model_2000_pca.pkl'),
    "nb": joblib.load('./models/multinomial_nb_2000_model_without_weight.pkl')
}
SELECTED_MODEL = "rfc"

logging.basicConfig(level=logging.INFO)

def letters_extract(image_file: str, out_size=28) -> List[Any]:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]

            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:

                y_pos = size_max // 2 - h // 2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    letters.sort(key=lambda x: x[0], reverse=False)

    return letters

def predict_symbol(img, model):
    img_flatten = img.flatten()

    scaler = joblib.load('./models/scaler_2000_pca.pkl')
    pca = joblib.load('./models/pca_model_2000.pkl')

    img_flatten_scaled = scaler.transform([img_flatten])
    img_reduced = pca.transform(img_flatten_scaled)

    return model.predict(img_reduced)

def get_string_from_image(path_to_image, model, size=45):
    result = ''

    letters = letters_extract(path_to_image, size)

    for idx, img_data in enumerate(letters):
        result += predict_symbol(
            img_data[2],
            model
        ).item()

    return result

def recognize_formula(image_path, model, img_size=45):
    return get_string_from_image(image_path, model, img_size)

def get_choosed_model():
    return MODELS[SELECTED_MODEL]

async def start_handler(message: Message):
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="Выбрать RFC"),
                KeyboardButton(text="Выбрать SVC"),
                KeyboardButton(text="Выбрать NB"),
            ]
        ],
        resize_keyboard=True,
        one_time_keyboard=False
    )
    await message.answer(
        "Привет! Отправь мне фото формулы. Выберите модель для распознавания:",
        reply_markup=keyboard
    )

async def model_selection_handler(message: Message):
    global SELECTED_MODEL

    if message.text == "Выбрать RFC":
        SELECTED_MODEL = "rfc"
        await message.answer("Вы выбрали модель RFC.")
    elif message.text == "Выбрать SVC":
        SELECTED_MODEL = "svc"
        await message.answer("Вы выбрали модель SVC.")
    elif message.text == "Выбрать NB":
        SELECTED_MODEL = "nb"
        await message.answer("Вы выбрали модель Наивный Байес.")

async def handle_photo(message: Message, bot: Bot):
    photo = message.photo[-1]
    file = await bot.download(photo)
    file_path = f"./downloads/{photo.file_id}.jpg"

    try:
        with open(file_path, "wb") as f:
            f.write(file.read())
            model = get_choosed_model()
            formula = recognize_formula(file_path, model)
            await message.answer(f"Распознанная формула: {formula} (модель: {SELECTED_MODEL.upper()})")
    except Exception as e:
        logging.error(f"Ошибка при обработке изображения: {e}")
        await message.answer("Произошла ошибка при обработке фото.")
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Файл {file_path} успешно удалён.")
        elif file_path:
            print(f"Файл {file_path} не найден для удаления.")

async def main():
    bot = Bot(token=API_TOKEN)
    dp = Dispatcher()

    dp.message.register(start_handler, Command(commands=["start"]))
    dp.message.register(model_selection_handler, lambda message: message.text in ["Выбрать RFC", "Выбрать SVC", "Выбрать NB"])
    dp.message.register(handle_photo, lambda message: message.photo)

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())



