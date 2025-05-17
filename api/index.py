from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

@app.route('/api/anon_text', methods=['POST'])
def anon_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'result': ''})
        # Простая демо-анонимизация
        return jsonify({
            'result': text.replace('Петров Алексей Иванович', '[ФИО]')
                        .replace('+7 (916) 987-65-43', '[ТЕЛЕФОН]')
                        .replace('12.05.1978', '[ДАТА]')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/anon_file', methods=['POST'])
def anon_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не найден'}), 400
        return jsonify({'result': 'Файл успешно анонимизирован'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For Vercel serverless function
def handler(request):
    if request.method == "POST":
        if request.path == "/api/anon_text":
            return app.view_functions['anon_text']()
        elif request.path == "/api/anon_file":
            return app.view_functions['anon_file']()
    return jsonify({'error': 'Not found'}), 404
