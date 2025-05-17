from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'API is running'

@app.route('/api/health')
def health():
    return jsonify({"status": "ok"})

# Export the Flask app for Vercel
app.debug = False
