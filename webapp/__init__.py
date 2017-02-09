from flask import Flask
app = Flask(__name__)
app.config.from_object('config')
# Avoid circular references
from webapp import views