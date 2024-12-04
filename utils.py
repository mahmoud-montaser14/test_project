import logging
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import Config
import tensorflow as tf
# Load the model at the start of the app
# try:
#     model = load_model(Config.MODEL_PATH)
#     logging.info("Model loaded successfully.")
# except Exception as e:
#     logging.error(f"Failed to load model: {e}")
#     raise RuntimeError(f"Model could not be loaded: {e}")

# # Function to preprocess images for the model
# def preprocess_image(image_path, target_size=(128, 128)):
#     try:
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError(f"Could not read the image from {image_path}")

#         old_size = image.shape[:2]
#         ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
#         new_size = tuple([int(x * ratio) for x in old_size])
#         resized_image = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_LANCZOS4)

#         delta_w = target_size[1] - new_size[1]
#         delta_h = target_size[0] - new_size[0]
#         top, bottom = delta_h // 2, delta_h - (delta_h // 2)
#         left, right = delta_w // 2, delta_w - (delta_w // 2)

#         new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#         new_image = new_image.astype('float32')
#         new_image = new_image / 255.0
#         # new_image = preprocess_input(new_image)
#         return np.expand_dims(new_image, axis=0)
    
#     except Exception as e:
#         logging.error(f"Error during image preprocessing: {e}")
#         raise ValueError(f"Image preprocessing failed: {e}")

# # Function to handle predictions and result formatting
# def predict_and_format_result(image_path):
#     try:
#         processed_image = preprocess_image(image_path)
#         prediction = model.predict(processed_image)
#         predicted_probabilities = prediction.flatten()
#         predicted_class_index = np.argmax(predicted_probabilities)
#         print(predicted_class_index)
#         max_probability = predicted_probabilities[predicted_class_index]

#         class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
#         result_class = class_names[predicted_class_index]
#         logging.info(f"Prediction: {result_class}, Probabilities: {predicted_probabilities}")
#         return result_class, f"{max_probability:.4f}"
    
#     except Exception as e:
#         logging.error(f"Prediction failed: {e}")
#         raise ValueError(f"Prediction failed: {e}")

# Initialize and load the TFLite Interpreter
# Load the anomaly detection autoencoder model
try:
    autoencoder = load_model("autoencoder_model.h5")
    logging.info("Anomaly detection model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load anomaly detection model: {e}")
    raise RuntimeError(f"Anomaly detection model could not be loaded: {e}")

try:
    interpreter = tf.lite.Interpreter(model_path="D:/api/model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("TFLite model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load TFLite model: {e}")
    raise RuntimeError(f"TFLite model could not be loaded: {e}")

# Flask application factory function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image as before
def preprocess_image(image_path, target_size=(128, 128)):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read the image from {image_path}")
        
        # Resize with padding
        old_size = image.shape[:2]
        ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
        new_size = tuple([int(x * ratio) for x in old_size])
        resized_image = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_LANCZOS4)
        delta_w = target_size[1] - new_size[1]
        delta_h = target_size[0] - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        new_image = new_image.astype('float32')
        new_image /= 255.0  # Normalize to [0, 1]
        return np.expand_dims(new_image, axis=0)
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise ValueError(f"Image preprocessing failed: {e}")
def is_anomalous(image_path, threshold=0.02):
    try:
        processed_image = preprocess_image(image_path)
        reconstructed = autoencoder.predict(processed_image)
        reconstruction_error = np.mean((processed_image - reconstructed) ** 2)
        return reconstruction_error > threshold
    except Exception as e:
        logging.error(f"Anomaly detection failed: {e}")
        raise ValueError(f"Anomaly detection failed: {e}")
# Predict using the TFLite Interpreter
def predict_and_format_result(image_path):
    try:
        # Check for anomalies
        if is_anomalous(image_path):
            logging.info("Image detected as anomalous.")
            return "Anomalous", "N/A"

        # Preprocess for classification
        processed_image = preprocess_image(image_path)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)

        # Run inference
        interpreter.invoke()

        # Get classification result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data)
        class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
        result_class = class_names[predicted_class_index]
        max_probability = output_data[0][predicted_class_index]

        logging.info(f"Prediction: {result_class}, Probabilities: {output_data}")
        return result_class, f"{max_probability:.4f}"
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise ValueError(f"Prediction failed: {e}")
