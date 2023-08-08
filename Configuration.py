import os
from flask import Flask

class Config:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = "secret_key"
        self.app.config['UPLOAD_FOLDER'] = self._get_upload_folder('static/uploads')
        self.app.config['UPLOAD_FOLDER1'] = self._get_upload_folder('static/Segments')
        self.app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

    def _get_upload_folder(self, folder_name):
        return os.path.join(os.path.dirname(__file__), folder_name)

    def get_app(self):
        return self.app