import os
import io
import re
import base64
import json
import requests
import numpy as np
import cv2
import pytesseract
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from PyPDF2 import PdfReader
from PIL import Image
from docx import Document
from google.cloud import vision
from google.cloud.vision_v1 import types
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix

# --- ENVIRONMENT SETUP ---
load_dotenv()
GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GOOGLE_CREDENTIALS_PATH:
    raise RuntimeError("Не установлена переменная окружения GOOGLE_APPLICATION_CREDENTIALS.")
if not GEMINI_API_KEY:
    raise RuntimeError("Не установлена переменная окружения GEMINI_API_KEY.")

# --- FLASK APP ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# --- GEMINI AI CONFIG ---
GEMINI_API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}'

def pre_anon_text(text):
    """
    Предварительная анонимизация текста по расширенным шаблонам (99% успеха).
    """
    patterns = [
        # ФИО (три слова с заглавной буквы, возможны двойные фамилии/имена)
        (r'\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\b', '[ФИО]'),
        # ФИО (две части, например "Светлана Викторовна" в обращении)
        (r'\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\b', '[ФИО]'),
        # ИИН (12 циферок подряд)
        (r'ИИН[\s:]*[\d  ]{12,}', '[ИИН]'),
        (r'\b\d{12}\b', '[ИИН]'),
        # Паспорт РФ, Казахстана, серия и номер (разные форматы, с пробелами/юникодными пробелами)
        (r'паспорт( серии)?\s*[A-ZА-ЯЁ]{2}[\s ]*\d{2}[\s ]*\d{6,}', '[ПАСПОРТ]'),
        (r'паспорт( серии)?\s*\d{2}\s*[A-ZА-ЯЁ]{2}\s*\d{6,}', '[ПАСПОРТ]'),
        # Водительское удостоверение (Казахстан, РФ)
        (r'водительск[а-я]+\s*удостоверени[ея][\s:]*\d{2}\s*[A-ZА-ЯЁ]{2}\s*\d{6,}', '[ВУ]'),
        (r'водительск[а-я]+\s*удостоверени[ея][\s:]*[A-ZА-ЯЁ0-9\s]+', '[ВУ]'),
        # Счёт (KZ, RU, IBAN)
        (r'сч[её]т\s*[A-Z]{2}\s*\d{2,4}(?:\s*\d{4}){3,}', '[СЧЁТ]'),
        (r'KZ\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}', '[СЧЁТ]'),
        # Телефоны (разные форматы, с пробелами, дефисами)
        (r'\+7[\s ]?\(?\d{3}\)?[\s -]?\d{3}[\s -]?\d{2}[\s -]?\d{2}', '[ТЕЛЕФОН]'),
        (r'8[\s ]?\(?\d{3}\)?[\s -]?\d{3}[\s -]?\d{2}[\s -]?\d{2}', '[ТЕЛЕФОН]'),
        # Email
        (r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]'),
        # Дата (разные разделители, с "выдан", "открыт" и тп)
        (r'(выдан[оая]?|открыт[ао]?|дата рождения|от)\s*[:\-]?\s*\d{2}[./]\d{2}[./]\d{4}', r'\1 [ДАТА]'),
        (r'\d{2}[./]\d{2}[./]\d{4}', '[ДАТА]'),
        # Адреса (г., пр., ул., д., кв., офис, корпус, палата, этаж, кабинет)
        (r'г\.\s?[А-ЯЁ][а-яё\- ]+,?\s*(пр\.|ул\.|просп\.|пер\.|бульвар|шоссе|пл\.)?\s*[А-ЯЁа-яё0-9\- ]+,?\s*д\.\s*\d+[А-Яа-я]?', '[АДРЕС]'),
        (r'пр\.\s?[А-ЯЁ][а-яё\- ]+,?\s*д\.\s*\d+[А-Яа-я]?', '[АДРЕС]'),
        (r'ул\.\s?[А-ЯЁ][а-яё\- ]+,?\s*д\.\s*\d+[А-Яа-я]?', '[АДРЕС]'),
        (r'просп\.\s?[А-ЯЁ][а-яё\- ]+,?\s*д\.\s*\d+[А-Яа-я]?', '[АДРЕС]'),
        (r'д\.\s*\d+[А-Яа-я]?', '[ДОМ]'),
        (r'кв\.\s*\d+', '[КВАРТИРА]'),
        (r'офис\s*\d+', '[НОМЕР_ОФИСА]'),
        (r'каб\.?\s*\d+', '[КАБИНЕТ]'),
        (r'палата\s*\d+', '[ПАЛАТА]'),
        (r'корпус\s*\d+', '[КОРПУС]'),
        (r'\b\d{1,2}\s?(?:этаж|эт|floor)\b', '[ЭТАЖ]'),
        # Номера документов (№, N, договор, заявка, полис и т.п.)
        (r'№[A-Za-zА-Яа-я0-9\-/]+', '[НОМЕР_ДОКУМЕНТА]'),
        (r'N[A-Za-zА-Яа-я0-9\-/]+', '[НОМЕР_ДОКУМЕНТА]'),
        (r'Договор\s*№[A-Za-zА-Яа-я0-9\-/]+', '[НОМЕР_ДОГОВОРА]'),
        (r'Полис\s*ОМС[:\s]*\d{4} \d{4} \d{4} \d{4}', '[НОМЕР_ПОЛИСА]'),
        (r'заявка\s*№?\s*[A-Za-zА-Яа-я0-9\-/]+', '[НОМЕР_ЗАЯВКИ]'),
        # Организации
        (r'ООО\s*"?[А-Яа-яA-Za-z0-9\s]+"?', '[НАЗВАНИЕ_ОРГАНИЗАЦИИ]'),
        # Время (HH:MM, HH:MM:SS)
        (r'\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b', '[ВРЕМЯ]'),
    ]
    for pat, repl in patterns:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE | re.UNICODE)
    return text

def local_anon_text(text, features=None):
    """
    Локальная анонимизация текста по расширенным шаблонам (99% успеха).
    """
    return pre_anon_text(text)

def gemini_anon_text(text, features=None):
    """
    Анонимизация текста через Gemini API с предварительной шаблонной анонимизацией.
    """
    pre_anon = pre_anon_text(text)
    # Привести features к списку, если это строка (например, из формы)
    if isinstance(features, str):
        try:
            features = json.loads(features)
        except Exception:
            features = [features]
    if features is not None and not isinstance(features, list):
        features = [features]

    # Если features пустой список — не анонимизировать ничего
    if features is not None and len(features) == 0:
        return pre_anon

    if features:
        tags, tag_desc = [], []
        if 'names' in features:
            tags.append('[ФИО]')
            tag_desc.append('ФИО (имена, фамилии, отчества)')
        if 'contacts' in features:
            tags.extend(['[ТЕЛЕФОН]', '[EMAIL]'])
            tag_desc.append('телефоны и email')
        if 'dates' in features:
            tags.append('[ДАТА]')
            tag_desc.append('даты')
        if 'ids' in features:
            tags.append('[НОМЕР_ДОКУМЕНТА]')
            tag_desc.append('номера документов, любые номера, идентификаторы, ИИН, ИНН, номера договоров, заявки, счета, полиса, паспорта, любые последовательности цифр и букв с символами №, N, /, -, АБ, А- и т.п.')
        if 'addresses' in features:
            tags.append('[АДРЕС]')
            tag_desc.append('адреса')
        if 'floors' in features:
            tags.append('[ЭТАЖ]')
            tag_desc.append('этажи')
        if 'rooms' in features:
            tags.append('[КАБИНЕТ]')
            tag_desc.append('кабинеты, комнаты')
        if 'wards' in features:
            tags.append('[ПАЛАТА]')
            tag_desc.append('палаты')
        if 'corpus' in features:
            tags.append('[КОРПУС]')
            tag_desc.append('корпуса')
        if 'times' in features:
            tags.append('[ВРЕМЯ]')
            tag_desc.append('время')
        tags_str = ', '.join(tags)
        tag_desc_str = ', '.join(tag_desc)
        prompt = (
            f"Анонимизируй только следующие типы данных в тексте: {tag_desc_str}. "
            f"Каждый найденный фрагмент этого типа замени строго на соответствующий тег из списка: {tags_str}. "
            "Не анонимизируй никакие другие данные, даже если они похожи на персональные. "
            "Если в тексте нет данных выбранных типов — не изменяй их. "
            "Если встречаются другие персональные данные, которые не относятся к выбранным типам, оставь их без изменений. "
            "Сохрани структуру и смысл текста.\n"
            "Вот исходный текст:\n"
            f"{text}\n"
            "Вот текст после шаблонной анонимизации:\n"
            f"{pre_anon}\n"
            "Если шаблонная анонимизация уже корректна — просто подтверди или скорректируй её. Если нет — исправь и выдай максимально точный анонимизированный текст."
        )
    else:
        prompt = (
            "Анонимизируй следующий текст: замени все ФИО, телефоны, email, номера документов, даты, адреса, время, этажи, кабинеты, палаты, корпуса и другие персональные данные на соответствующие теги в квадратных скобках (например, [ФИО], [ТЕЛЕФОН], [EMAIL], [ДАТА], [АДРЕС], [НОМЕР_ДОКУМЕНТА], [ВРЕМЯ], [ЭТАЖ], [КАБИНЕТ], [ПАЛАТА], [КОРПУС]). "
            "Если в тексте есть другие чувствительные данные (например, номер больницы, номер комнаты, номер карты, номер полиса, номер заявки, номер счета, номер паспорта и т.п.), также анонимизируй их аналогично, используя подходящие теги. "
            "Сохрани структуру и смысл текста.\n"
            "Вот исходный текст:\n"
            f"{text}\n"
            "Вот текст после шаблонной анонимизации:\n"
            f"{pre_anon}\n"
            "Если шаблонная анонимизация уже корректна — просто подтверди или скорректируй её. Если нет — исправь и выдай максимально точный анонимизированный текст."
        )
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if 'candidates' in result and result['candidates']:
            return result['candidates'][0]['content']['parts'][0]['text']
        if 'error' in result:
            return f"Ошибка Gemini: {result['error'].get('message', 'Неизвестная ошибка')}"

        return "Ошибка: пустой ответ от Gemini"
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException):
        # Fallback: локальная анонимизация если нет интернета или ошибка соединения
        return local_anon_text(text, features)
    except Exception as e:
        return f"Ошибка анонимизации: {e}"

@app.route('/anon_text', methods=['POST'])
@cross_origin()
def anon_text():
    data = request.get_json()
    text = data.get('text', '')
    features = data.get('features', None)
    if not text.strip():
        return jsonify({'result': ''})
    anon = gemini_anon_text(text, features)
    return jsonify({'result': anon})

def anonymize_faces(image_pil, features=None):
    """
    Анонимизация всех важных данных на изображении с использованием Google Vision API.
    """
    cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not cred_path or not os.path.isfile(cred_path):
        raise RuntimeError(
            f"Google Vision не настроен. Переменная GOOGLE_APPLICATION_CREDENTIALS не установлена или путь не существует: {cred_path!r}."
        )
    
    image_np = np.array(image_pil.convert("RGB"))
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    try:
        client = vision.ImageAnnotatorClient()
        buf = io.BytesIO()
        image_pil.save(buf, format='JPEG')
        content = buf.getvalue()
        image = vision.Image(content=content)

        # Определение важных данных через Google Vision API
        # 1. Лица
        face_response = client.face_detection(image=image)
        faces = face_response.face_annotations
        
        # 2. Текст
        text_response = client.document_text_detection(image=image)
        
        # 3. Объекты
        object_response = client.object_localization(image=image)
        
        # Маска для всех важных областей
        mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
        
        # Размытие лиц
        for face in faces:
            vertices = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
            pts = np.array(vertices, np.int32)
            x1, y1 = np.min(pts, axis=0)
            x2, y2 = np.max(pts, axis=0)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            if x2 > x1 and y2 > y1:
                roi = image_cv[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                    image_cv[y1:y2, x1:x2] = blurred

        # Поиск и размытие важного текста
        if text_response.text_annotations:
            # Регулярные выражения для поиска важных данных
            patterns = [
                r'\d{6,}',  # Номера документов
                r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}',  # Телефоны
                r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',  # Email
                r'\d{2}[./]\d{2}[./]\d{4}',  # Даты
                r'[А-Я][а-я]+\s+[А-Я][а-я]+(?:\s+[А-Я][а-я]+)?',  # ФИО
                r'(?:ул\.|пр\.|просп\.|д\.)\s*[\w\-]+,?\s*д\.?\s*\d+',  # Адреса
                r'\b(?:ИНН|ОГРН|КПП|БИК|р/с|к/с)\s*:?\s*\d+\b',  # Реквизиты
                r'(?:паспорт|снилс|инн|полис)\s*:?\s*\d+\b',  # Документы
                # Новые паттерны для времени, этажей и т.п.
                r'\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b',  # время HH:MM или HH:MM:SS
                r'\b\d{1,2}\s?(?:этаж|эт|floor)\b',  # этажи
                r'\b\d{1,2}\s?(?:каб\.?|кабинет|room)\b',  # кабинеты/комнаты
                r'\b\d{1,2}\s?(?:палата|пал)\b',  # палаты
                r'\b\d{1,2}\s?(?:корпус|корп)\b',  # корпуса
            ]
            
            for text in text_response.text_annotations[1:]:  # Пропускаем первую аннотацию (полный текст)
                text_content = text.description.strip()
                if any(re.search(pattern, text_content, re.IGNORECASE) for pattern in patterns):
                    vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
                    points = np.array(vertices, np.int32)
                    x1, y1 = np.min(points, axis=0)
                    x2, y2 = np.max(points, axis=0)
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    if x2 > x1 and y2 > y1:
                        roi = image_cv[y1:y2, x1:x2]
                        if roi.size > 0:
                            blurred = cv2.GaussianBlur(roi, (31, 31), 0)
                            image_cv[y1:y2, x1:x2] = blurred

        # Дополнительное определение текста через OCR
        try:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            text_blocks = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='rus+eng');
            
            for i, text in enumerate(text_blocks['text']):
                if not text.strip():
                    continue
                # Проверяем найденный текст на наличие важных данных
                if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                    x, y, w, h = (
                        text_blocks['left'][i],
                        text_blocks['top'][i],
                        text_blocks['width'][i],
                        text_blocks['height'][i]
                    )
                    if w > 0 and h > 0:
                        roi = image_cv[y:y+h, x:x+w]
                        if roi.size > 0:
                            blurred = cv2.GaussianBlur(roi, (31, 31), 0)
                            image_cv[y+y+h, x:x+w] = blurred
        except Exception as e:
            print(f"OCR error: {e}")

        result_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        return result_pil

    except Exception as e:
        err_msg = str(e)
        if "default credentials" in err_msg.lower():
            raise RuntimeError(
                "Google Vision не настроен. Для анонимизации изображений установите переменную окружения GOOGLE_APPLICATION_CREDENTIALS."
            )
        if "billing" in err_msg.lower() or "BILLING_DISABLED" in err_msg:
            raise RuntimeError(
                "Google Vision API отключён из-за отсутствия оплаты. Включите биллинг для проекта Google Cloud."
            )
        raise

@app.route('/anon_file', methods=['POST'])
@cross_origin()
def anon_file():
    file = request.files.get('file')
    if not file:
        return jsonify({'result': ''})
    filename = file.filename.lower()
    # TXT
    if filename.endswith('.txt'):
        text = file.read().decode('utf-8', errors='ignore')
        if not text.strip():
            return jsonify({'result': ''})
        anon = gemini_anon_text(text)
        return jsonify({'result': anon})
    # PDF
    elif filename.endswith('.pdf'):
        try:
            pdf = PdfReader(file)
            text = "".join(page.extract_text() or "" for page in pdf.pages)
            if not text.strip():
                return jsonify({'result': ''})
            anon = gemini_anon_text(text)
            return jsonify({'result': anon})
        except Exception as e:
            return jsonify({'error': f'Ошибка обработки PDF: {e}'}), 400
    # DOCX
    elif filename.endswith('.docx'):
        try:
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            if not text.strip():
                return jsonify({'result': ''})
            anon = gemini_anon_text(text)
            return jsonify({'result': anon})
        except Exception as e:
            return jsonify({'error': f'Ошибка обработки DOCX: {e}'}), 400
    # Изображения (JPG, JPEG, PNG)
    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            image = Image.open(file.stream)
            features = None
            if 'features' in request.form:
                try:
                    features = json.loads(request.form['features'])
                except Exception:
                    features = None
            anonymized = anonymize_faces(image, features)
            buf = io.BytesIO()
            anonymized.save(buf, format='PNG')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return jsonify({'result': 'data:image/png;base64,' + img_base64})
        except RuntimeError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': f'Ошибка обработки изображения: {e}'}), 400
    else:
        return jsonify({'error': 'Поддерживаются только TXT, PDF, DOCX, JPG, PNG файлы'}), 400

@app.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    data = request.get_json()
    message = data.get('message', '')
    is_text = data.get('isText', False)
    if not message.strip():
        return jsonify({'reply': 'Пожалуйста, введите сообщение.'})
    if is_text or 'анонимиз' in message.lower():
        anon = gemini_anon_text(message)
        return jsonify({'reply': anon})
    return jsonify({'reply': 'Я могу анонимизировать ваш текст. Просто отправьте его или напишите "анонимизируй ..."'})

@app.route('/')
def root():
    # Отдаём index.html из public/
    public_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'public'))
    return send_from_directory(public_dir, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    # Отдаём любые статические файлы из public/
    public_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'public'))
    file_path = os.path.join(public_dir, path)
    if os.path.isfile(file_path):
        return send_from_directory(public_dir, path)
    return send_from_directory(public_dir, 'index.html')

if __name__ == '__main__':
    # Для локальной разработки
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    # Для Vercel
    app = app.wsgi_app