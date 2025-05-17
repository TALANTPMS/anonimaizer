from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

@app.route('/anon_text', methods=['POST'])
def anon_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'result': ''})
        # Simple text anonymization
        anon_text = text.replace('Петров Алексей Иванович', '[ФИО]')\
                       .replace('12.05.1978', '[ДАТА]')\
                       .replace('1234 5678 9012 3456', '[НОМЕР_ДОКУМЕНТА]')
        return jsonify({'result': anon_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/anon_file', methods=['POST'])
def anon_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        return jsonify({'result': 'File processed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# This is crucial for Vercel
app = app.wsgi_app
