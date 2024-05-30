import os
from flask import Flask, request, render_template
from io import BytesIO
from PIL import Image, ImageOps
import base64
import urllib
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from preprocessing import CropImages, PreprocessImages  # Import the custom transformers
import cv2

app = Flask(__name__)

# Load your trained model
model_path = 'FINAL.h5'

try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load the preprocessing pipeline
pipeline_path = 'preprocessing_pipeline.pkl'

try:
    preprocessing_pipeline = joblib.load(pipeline_path)
    print("Preprocessing pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading preprocessing pipeline: {e}")

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/chart")
def chart():
    return render_template('chart.html')

@app.route("/performance")
def performance():
    return render_template('performance.html')

@app.route("/index", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/upload", methods=['POST'])
def upload_file():
    if 'imagefile' not in request.files:
        error_msg = "No file part in the request"
        return render_template('index.html', error_msg=error_msg)
    
    file = request.files['imagefile']
    
    if file.filename == '':
        error_msg = "No selected file"
        return render_template('index.html', error_msg=error_msg)

    try:
        img = Image.open(BytesIO(file.read())).convert('RGB')
        img = ImageOps.fit(img, (224, 224), Image.LANCZOS)
        
        # Check if the image is grayscale
        grayscale_img = img.convert('L')
        img_array = np.array(img)
        grayscale_img_array = np.array(grayscale_img)
        
        if not np.array_equal(grayscale_img_array, img_array.mean(axis=2)):
            error_msg = "The uploaded image is not MRI. Please upload a MRI image."
            return render_template('index.html', error_msg=error_msg)
        
        # Additional validation checks
        if not validate_mri_image(grayscale_img_array):
            error_msg = "The uploaded image does not appear to be an MRI. Please upload a valid MRI image."
            return render_template('index.html', error_msg=error_msg)

    except Exception as e:
        error_msg = f"Please choose an image file! Error: {e}"
        return render_template('index.html', error_msg=error_msg)

    try:
        # Call Function to predict
        args = {'input': img}
        out_pred, out_prob, preprocessed_img_base64 = predict(args)
        out_prob = out_prob * 100

        danger = "danger"
        if out_pred == "Result: Normal":
            danger = "success"

        # Convert the processed image to base64
        img_io = BytesIO()
        img.save(img_io, 'PNG')
        png_output = base64.b64encode(img_io.getvalue()).decode('utf-8')
        processed_file = urllib.parse.quote(png_output)

        return render_template('result.html', out_pred=out_pred, out_prob=out_prob, processed_file=processed_file, danger=danger, preprocessed_file=preprocessed_img_base64)
    except Exception as e:
        error_msg = f"An error occurred during prediction: {e}"
        print(error_msg)
        return render_template('index.html', error_msg=error_msg)

def validate_mri_image(img_array):
    # Check the histogram for MRI-like intensity distribution
    histogram, bin_edges = np.histogram(img_array, bins=256, range=(0, 255))
    peaks = np.sort(histogram)[-2:]  # Check for two peaks
    if peaks[1] < 0.1 * np.sum(histogram):  # Simple check for bimodal distribution
        return False
    
    
    # Check texture features (e.g., Haralick features)
    texture_features = cv2.calcHist([img_array], [0], None, [256], [0, 256])
    if texture_features.mean() < 0.5:  # This threshold might need tuning
        return False
    
    return True

def predict(args):
    img = np.array(args['input'])

    # Apply preprocessing steps from the loaded pipeline
    img_preprocessed = preprocessing_pipeline.transform([img])

    # Debug: Print statistics of the preprocessed image
    print("Preprocessed image shape:", img_preprocessed.shape)
    print("Preprocessed image min value:", np.min(img_preprocessed))
    print("Preprocessed image max value:", np.max(img_preprocessed))
    print("Preprocessed image mean value:", np.mean(img_preprocessed))

    # Save the preprocessed image for inspection
    preprocessed_img_save_path = "preprocessed_image.png"
    cv2.imwrite(preprocessed_img_save_path, (img_preprocessed[0] * 255.0).astype(np.uint8))
    print(f"Preprocessed image saved to: {preprocessed_img_save_path}")

    # Convert the preprocessed image to base64
    _, img_encoded = cv2.imencode('.png', (img_preprocessed[0] * 255.0).astype(np.uint8))
    preprocessed_img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Predict using the loaded model
    pred = model.predict(img_preprocessed)
    print("Raw predictions:", pred)

    pred_label = [1 if x > 0.5 else 0 for x in pred]
    print("Predicted labels:", pred_label)

    if pred_label[0] == 1:
        out_pred = "Result: Brain Tumor Symptoms: unexplained weight loss, double vision or a loss of vision, increased pressure felt in the back of the head, dizziness and a loss of balance, sudden inability to speak, hearing loss, weakness or numbness that gradually worsens on one side of the body."
    else:
        out_pred = "Result: Normal"

    return out_pred, float(np.max(pred)), preprocessed_img_base64

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

