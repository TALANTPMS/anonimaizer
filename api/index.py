from flask import Flask
from action import app

# For Vercel serverless function
def handler(request):
    return app

# For local development
if __name__ == '__main__':
    app.run()
