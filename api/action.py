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

def gemini_anon_text(text, features=None):
    """
    Анонимизация текста через Gemini API.
    """
    feature_map = {
        'names': '[ФИО]',
        'contacts': '[ТЕЛЕФОН], [EMAIL]',
        'dates': '[ДАТА]',
        'ids': '[НОМЕР_ДОКУМЕНТА]',
        'addresses': '[АДРЕС]'
    }
    if features is not None and len(features) == 0:
        return text
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
            tag_desc.append('номера документов, любые номера, идентификаторы, ИИН, ИНН, номера договоров, заявки, счета, полиса, паспорта, любые последовательности цифр и букв с символами №, /, -, АБ, А- и т.п.')
        if 'addresses' in features:
            tags.append('[АДРЕС]')
            tag_desc.append('адреса')
        tags_str = ', '.join(tags)
        tag_desc_str = ', '.join(tag_desc)
        prompt = (
            f"Анонимизируй только следующие типы данных в тексте: {tag_desc_str}. "
            f"Каждый найденный фрагмент этого типа замени строго на соответствующий тег из списка: {tags_str}. "
            "Не анонимизируй никакие другие данные, даже если они похожи на персональные. "
            "Если в тексте нет данных выбранных типов — не изменяй их. "
            "Если встречаются другие персональные данные, которые не относятся к выбранным типам, оставь их без изменений. "
            "Сохрани структуру и смысл текста. Вот текст:\n" + text
        )
    else:
        prompt = (
            "Анонимизируй следующий текст: замени все ФИО, телефоны, email, номера документов, даты, адреса и другие персональные данные на соответствующие теги в квадратных скобках (например, [ФИО], [ТЕЛЕФОН], [EMAIL], [ДАТА], [АДРЕС], [НОМЕР_ДОКУМЕНТА]). "
            "Если в тексте есть другие чувствительные данные (например, номер больницы, номер комнаты, номер карты, номер полиса, номер заявки, номер счета, номер паспорта и т.п.), также анонимизируй их аналогично, используя подходящие теги. "
            "Сохрани структуру и смысл текста. Вот текст:\n" + text
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
    except requests.exceptions.Timeout:
        return "Ошибка: превышено время ожидания ответа от Gemini"
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
                            image_cv[y:y+h, x:x+w] = blurred

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
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    # Для локальной разработки
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    # Для Vercel
    app = app.wsgi_app