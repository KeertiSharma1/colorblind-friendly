from flask import Flask, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_pattern(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    blurred_image = cv2.GaussianBlur(adaptive_thresh, (5, 5), 0)
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pattern_mask = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.drawContours(pattern_mask, contours, -1, (255), thickness=cv2.FILLED)
    return pattern_mask

def detect_and_transform_color(image, pattern_mask, blindness_type):
    if np.count_nonzero(pattern_mask) == 0:
        raise ValueError("Pattern mask is empty. No pattern detected.")
    pattern_mask_3ch = cv2.merge([pattern_mask] * 3)
    pattern_area = cv2.bitwise_and(image, pattern_mask_3ch)
    hsv_image = cv2.cvtColor(pattern_area, cv2.COLOR_BGR2HSV)

    def transform_color(hsv_pixel):
        h, s, v = hsv_pixel
        if blindness_type == 'red-green':
            if h < 10 or h > 160:
                h = 30  # Yellow hue
        elif blindness_type == 'blue-yellow':
            if 35 < h < 85:
                h = 160  # Pink hue
        elif blindness_type == 'total':
            h = 0  # Black and white transformation for total color blindness
        return (h, s, v)
    
    hsv_image = hsv_image.astype(np.float32)
    transformed_hsv = np.apply_along_axis(transform_color, 2, hsv_image)
    transformed_hsv = np.clip(transformed_hsv, 0, 255).astype(np.uint8)
    transformed_pattern_area = cv2.cvtColor(transformed_hsv, cv2.COLOR_HSV2BGR)
    background = cv2.bitwise_and(image, cv2.bitwise_not(pattern_mask_3ch))
    result_image = cv2.add(background, transformed_pattern_area)
    return result_image

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'blindness_type' not in request.form:
        return jsonify({'error': 'No file or blindness type provided'}), 400
    
    file = request.files['file']
    blindness_type = request.form['blindness_type']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Load image
    image = cv2.imread(file_path)
    
    if image is None:
        return jsonify({'error': 'Error loading image'}), 500
    
    # Process image
    pattern_mask = detect_pattern(image)
    transformed_image = detect_and_transform_color(image, pattern_mask, blindness_type)
    
    # Convert to PIL Image and then to bytes
    _, buffer = cv2.imencode('.png', transformed_image)
    pil_image = Image.open(BytesIO(buffer))
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

