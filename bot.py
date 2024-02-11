from ultralytics import YOLO
import telebot
from telebot import types
from telebot.types import InputFile, InputMediaVideo
import os
import pandas as pd
from PIL import Image
import io
import numpy as np
import yaml
from helpers import *

clear_temp_dirs()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

bot = telebot.TeleBot(config['token'])
print("i'm awake")
strt = types.BotCommand('start', 'Вернуться в начало')
detect = types.BotCommand('detect', 'Определить породу')
metrcis = types.BotCommand('metrics', 'Метрики')
about = types.BotCommand('about', 'О проекте')
bot.set_my_commands(commands= [strt,detect,metrcis,about])

goodbyes = ['поки','пока', 'до связи']
alo = ['ало', 'оло','олу','алу','алло','олло']
thanks = ['спасибо', 'сенкс', 'спс', 'спасибки']
greeting = '\n'.join(["Привет, я котенок по имени Гав.",
        "Я знаю более 100 различных пород собак и помогу тебе определить породу по картинке или видео",
        "Выбирай 'Определить породу' или просто отправь мне фотографию или видео"
        ])

cl_map = read_class_map('class_mapping.csv')

path_to_weights = os.path.join(os.getcwd(),'weights/best_60.pt')
model = YOLO(path_to_weights)
print('model loaded')



@bot.message_handler(commands=['start'])
def start(message):
    get_msg_data(message)
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn0 = types.KeyboardButton("Определить породу")
    btn1 = types.KeyboardButton("Метрики")
    btn2 = types.KeyboardButton("О проекте")
    markup.add(btn0)
    markup.add(btn1)
    markup.add(btn2)
    bot.send_document(message.from_user.id, document="CAACAgIAAxkBAAElhJVk79KGyUEEP_9qejuYOc3JWIsLwwACjhcAAhspoEhGVp2TgPvWUTAE")
    bot.send_message(message.from_user.id, greeting, reply_markup=markup)



@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    get_msg_data(message)
    msg = bot.send_message(message.from_user.id, 'Секундочку...')
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    img = Image.open(io.BytesIO(downloaded_file))
    preds = model.predict(img,conf = 0.15)
    bot.edit_message_text('Обработка завершена',message.from_user.id,msg.message_id)
    if len(preds[0].boxes.cls) != 0:
        img = np.array(img)
        img_to_save = illustrate_boxes(preds[0],cl_map,0.8,img)
        # img_to_save = Image.fromarray(img_to_save)
        save_path = 'saved_temp/photo_from_{}.jpg'.format(message.from_user.username)
        img_to_save.save(save_path)
        bot.send_photo(message.from_user.id,InputFile(save_path))
        bot.delete_message(message.from_user.id,msg.message_id)
        if os.path.isfile(save_path):
            os.remove(save_path)
    else:
        bot.edit_message_text('К сожалению ничего не нашлось, попробуйте другое фото',message.from_user.id,msg.message_id)


@bot.message_handler(content_types=['video'])
def handle_video(message):
    get_msg_data(message)
    stick = bot.send_document(message.from_user.id, document="CAACAgQAAxkBAAEpWiRlxFdyBAdzL04Db6M3XBuRdyYwJQAC_wADD6r0B6U5nuntNbspNAQ")
    msg = bot.send_message(message.from_user.id, 'Секундочку...')
    file_info = bot.get_file(message.video.file_id)
    bot.edit_message_text('Загружаю видео',message.from_user.id,msg.message_id)
    downloaded_file = bot.download_file(file_info.file_path)
    print('downloaded video')
    output_file = 'saved_temp/video_from_{}.avi'.format(message.from_user.username)
    with open(output_file, "wb") as out_file: 
        out_file.write(downloaded_file)
    print('file written')
    bot.edit_message_text('Обрабатываю',message.from_user.id,msg.message_id)
    model.predict(output_file,project = 'video_preds',name = 'temp',save = True,exist_ok=True )
    os.remove(output_file)
    bot.edit_message_text('Конвертирую в mp4',message.from_user.id,msg.message_id)
    pred_path = 'video_preds/temp/video_from_{}'.format(message.from_user.username)
    convert_avi_to_mp4(pred_path+'.avi',pred_path+'.mp4')
    print('converted')
    bot.edit_message_text('Осталось совсем чуть-чуть',message.from_user.id,msg.message_id)
    bot.send_video(message.from_user.id,InputFile(pred_path+'.mp4'))
    bot.delete_message(message.from_user.id,stick.message_id)
    bot.delete_message(message.from_user.id,msg.message_id)
    print('sent')
    os.remove(pred_path+'.mp4')


@bot.message_handler(commands=['detect'])
def start_detection_dialog(message):
    get_msg_data(message)
    dtc_msg = bot.send_message(message.from_user.id, 'Отправь мне фото или видео и я определю породы собак на нем')
    bot.register_next_step_handler(dtc_msg,find_breeds)
def find_breeds(message):
    get_msg_data(message)
    if message.photo:
        handle_photo(message)
    elif message.video:
        handle_video(message)
    else:
        dtc_msg = bot.send_message(message.from_user.id, 'Это не фото и не видео')
        bot.register_next_step_handler(dtc_msg,find_breeds)


@bot.message_handler(func=lambda message: message.text == 'Определить породу')
def start_detection_dialog_(message):
    start_detection_dialog(message)


@bot.message_handler(commands=['metrcis'])
def send_metrics(message):
    get_msg_data(message)
    bot.send_message(message.from_user.id,'тут будут метрики')

@bot.message_handler(commands=['about'])
def send_about(message):
    get_msg_data(message)
    bot.send_message(message.from_user.id,'тут будет about')

@bot.message_handler(func=lambda message: message.text == 'Метрики')
def send_metrics_(message):
    send_metrics(message)

@bot.message_handler(func=lambda message: message.text == 'О проекте')
def send_about_(message):
    send_about(message)



@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    get_msg_data(message)
    if str(message.text).lower() in goodbyes:
        markup = types.InlineKeyboardMarkup()
        btn1 = types.InlineKeyboardButton(text='github', url='https://guthib.com/')
        markup.add(btn1)
        bot.send_message(message.from_user.id, 'Пока, если тебе понравилось, посмотри мои остальные проекты на github',reply_markup=markup)
        bot.send_document(message.from_user.id, document="CAACAgIAAxkBAAElhLVk7-Laa9CXFIWa6Tj1l8vB9-_WfQACGBkAAp4WoUhNBRp4s3ZdGzAE")

    elif str(message.text).lower() ==  'хуй':
        bot.send_document(message.from_user.id, document="CAACAgQAAxkBAAElhLdk7-PhZBJS1t5ejfNSAAHZq0HjkKwAAvsAAw-q9AdB7mhhzhiekTAE")

    elif str(message.text).lower() in thanks:
        bot.send_message(message.from_user.id, 'Всегда пожалуйста')
        bot.send_sticker(message.from_user.id, 'CAACAgIAAxkBAAElifBk8KwDUqQTS0au9nwo6hqSkyNCQgACoBwAAipooUjogwEq_q_PRzAE')

    elif str(message.text).lower() in alo:
        bot.send_message(message.from_user.id, 'Ало, это я, Пригожин Женя')
        bot.send_sticker(message.from_user.id, "CAACAgIAAxkBAAEligJk8K1OwDcyyDW-WW90RDyUZ7F_9wAC7RgAAuJ1oUhGn01X81OLfjAE")

    else:
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn0 = types.KeyboardButton("Определить породу")
        btn1 = types.KeyboardButton("Метрики")
        btn2 = types.KeyboardButton("О проекте")
        markup.add(btn0)
        markup.add(btn1)
        markup.add(btn2)
        bot.send_message(message.from_user.id, 'Не очень понимаю запрос, я еще не выучил все слова на планете, выбери кнопку, пожалуйста🙏',reply_markup=markup)
        bot.send_sticker(message.from_user.id, "CAACAgIAAxkBAAElicxk8KmBV9fzVoLzmGA_F7OVOrHR9AACoxcAAhfYoEhkgAYF4jHAQDAE")

@bot.message_handler(content_types=['audio', 'document', 'sticker', 'video_note', 'voice', 'location', 'contact'])
def get_other_messages(message):
    get_msg_data(message)
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn0 = types.KeyboardButton("Определить породу")
    btn1 = types.KeyboardButton("Метрики")
    btn2 = types.KeyboardButton("О проекте")
    markup.add(btn0)
    markup.add(btn1)
    markup.add(btn2)
    bot.send_message(message.from_user.id, 'Я не умею работать с такими форматами данных',reply_markup=markup)
    bot.send_sticker(message.from_user.id, "CAACAgIAAxkBAAElicxk8KmBV9fzVoLzmGA_F7OVOrHR9AACoxcAAhfYoEhkgAYF4jHAQDAE")

bot.polling(timeout = 90000)
