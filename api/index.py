from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/anon_text', methods=['POST'])
def anon_text():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'result': ''})
    # Simple anonymization for testing
    return jsonify({'result': '[Анонимизированный текст]'})

@app.route('/')
def home():
    return 'API is running'

# For Vercel serverless function
def handler(request):
    return app

# For local development
if __name__ == '__main__':
    app.run()
