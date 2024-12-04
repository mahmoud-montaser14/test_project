import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_really_secure_key'
    UPLOAD_FOLDER = 'static/uploads/'
    MODEL_PATH = r'D:\api\model.tflite'  # Adjust path as needed
    # Configuration
    # Flask application factory function
    UPLOAD_FOLDER = 'uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

    
