from flask import Flask, request, render_template, jsonify
import os
import logging
from werkzeug.utils import secure_filename
from utils import preprocess_image, predict_and_format_result
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, filename='logs/app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Flask application factory function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config.from_object(Config)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_filename = None
    if request.method == 'POST':
        try:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                logging.info(f"File saved to {file_path}")

                # Predict and format result
                result, probability = predict_and_format_result(file_path)
                image_filename = filename
                return render_template('index.html', result=(result, probability), image=image_filename)
            else:
                raise ValueError("Invalid file type. Please upload an image.")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return render_template('index.html', result=f"Error: {str(e)}")
    return render_template('index.html', result=result, image=image_filename)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict and return result
            result, probability = predict_and_format_result(file_path)
            if result == "Anomalous":
                return jsonify({'error': 'Image is anomalous and cannot be classified.'}), 400
            return jsonify({'class': result, 'probability': probability})
        else:
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    except Exception as e:
        logging.error(f"API Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)