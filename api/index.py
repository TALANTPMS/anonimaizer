from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return 'API is running'

@app.route('/anon_text', methods=['POST'])
def anon_text():
    data = request.get_json()
    return jsonify({'result': 'Test response'})

# Vercel требует именно эту функцию
def handler(request):
    with app.request_context(request):
        return app.full_dispatch_request()
