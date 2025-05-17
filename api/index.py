import os
import io
import re
import base64
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document

# --- ENVIRONMENT ---
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

app = Flask(__name__, static_folder="../public", static_url_path="")
CORS(app)

# --- Текстовая анонимизация (простая, без внешних API) ---
def simple_anon_text(text):
    # Примитивные паттерны для ФИО, даты, номера, телефона, email, адреса
    patterns = [
        (r'[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+', '[ФИО]'),
        (r'\d{2}\.\d{2}\.\d{4}', '[ДАТА]'),
        (r'\d{4} \d{4} \d{4} \d{4}', '[НОМЕР_ДОКУМЕНТА]'),
        (r'\+7\s?\(?\d{3}\)?\s?\d{3}-\d{2}-\d{2}', '[ТЕЛЕФОН]'),
        (r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]'),
        (r'г\.\s?[А-ЯЁ][а-яё]+,?\s+ул\.\s+[А-ЯЁа-яё\-]+,?\s+д\.\s*\d+', '[АДРЕС]')
    ]
    for pat, repl in patterns:
        text = re.sub(pat, repl, text)
    return text

@app.route('/anon_text', methods=['POST'])
def anon_text():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'result': ''})
    anon = simple_anon_text(text)
    return jsonify({'result': anon})

# --- Анонимизация изображений (блюр лиц и текста) ---
def blur_faces_and_text(image_pil):
    image_np = np.array(image_pil.convert("RGB"))
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # --- Лица ---
    try:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(haar_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        for (x, y, w, h) in faces:
            roi = image_cv[y:y+h, x:x+w]
            if roi.size > 0:
                blurred = cv2.GaussianBlur(roi, (99, 99), 30)
                image_cv[y:y+h, x:x+w] = blurred
    except Exception:
        pass
    # --- Текст (OCR pytesseract) ---
    try:
        import pytesseract
        from pytesseract import Output
        data = pytesseract.image_to_data(image_pil, lang='rus+eng', output_type=Output.DICT)
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            if data['text'][i].strip():
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                if w > 0 and h > 0:
                    roi = image_cv[y:y+h, x:x+w]
                    if roi.size > 0:
                        blurred = cv2.GaussianBlur(roi, (31, 31), 0)
                        image_cv[y:y+h, x:x+w] = blurred
    except Exception:
        pass
    result_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    return result_pil

@app.route('/anon_file', methods=['POST'])
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
        anon = simple_anon_text(text)
        return jsonify({'result': anon})
    # PDF
    elif filename.endswith('.pdf'):
        try:
            pdf = PdfReader(file)
            text = "".join(page.extract_text() or "" for page in pdf.pages)
            if not text.strip():
                return jsonify({'result': ''})
            anon = simple_anon_text(text)
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
            anon = simple_anon_text(text)
            return jsonify({'result': anon})
        except Exception as e:
            return jsonify({'error': f'Ошибка обработки DOCX: {e}'}), 400
    # Изображения (JPG, JPEG, PNG)
    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            image = Image.open(file.stream)
            anonymized = blur_faces_and_text(image)
            buf = io.BytesIO()
            anonymized.save(buf, format='PNG')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return jsonify({'result': 'data:image/png;base64,' + img_base64})
        except Exception as e:
            return jsonify({'error': f'Ошибка обработки изображения: {e}'}), 400
    else:
        return jsonify({'error': 'Поддерживаются только TXT, PDF, DOCX, JPG, PNG файлы'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    if not message.strip():
        return jsonify({'reply': 'Пожалуйста, введите сообщение.'})
    # Демоверсия: просто возвращаем текст
    return jsonify({'reply': f'Вы написали: {message}'})

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def home():
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
    # Если файл не найден — отдаём index.html (SPA fallback)
    return send_from_directory(public_dir, 'index.html')

# Для Vercel/Render: экспортируем Flask app
# (Vercel/Render ищет переменную "app" в этом файле)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
