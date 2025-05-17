import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from action import app

# For Vercel Serverless Function
def handler(request, context):
    return app(request)
