import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from action import app as flask_app

def handler(request):
    """
    Handle Vercel serverless function requests
    """
    if request.method == "POST":
        return flask_app.view_functions['anon_text']()
    return flask_app.view_functions['home']()
