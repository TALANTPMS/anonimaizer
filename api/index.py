import os
import io
import re
import base64
import json
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image, ImageFilter

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

def blur_jpg_pillow(image_pil):
    # Примитивный блюр всего изображения (или можно реализовать crop/box)
    return image_pil.filter(ImageFilter.GaussianBlur(radius=16))

@app.route('/anon_text', methods=['POST'])
def anon_text():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'result': ''})
    anon = simple_anon_text(text)
    return jsonify({'result': anon})

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
            anonymized = blur_jpg_pillow(image)
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
    # Если есть файл (base64) — блюрить и вернуть
    if 'file' in data:
        try:
            file_data = data['file']
            header, b64data = file_data.split(',', 1)
            img_bytes = base64.b64decode(b64data)
            image = Image.open(io.BytesIO(img_bytes))
            anonymized = blur_jpg_pillow(image)
            buf = io.BytesIO()
            anonymized.save(buf, format='PNG')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return jsonify({'reply': 'data:image/png;base64,' + img_base64})
        except Exception as e:
            return jsonify({'reply': f'Ошибка обработки изображения: {e}'})
    # Если текст — анонимизировать
    if message and message.strip():
        anon = simple_anon_text(message)
        return jsonify({'reply': anon})
    return jsonify({'reply': 'Пожалуйста, введите сообщение или отправьте файл.'})

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

# Для Vercel: экспортируем Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
