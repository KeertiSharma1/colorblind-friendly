from flask import Flask, request, jsonify, send_from_directory, render_template,jsonify
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
from image_transformer import detect_pattern, detect_and_transform_color

app = Flask(__name__, template_folder='acuas-1.0.0')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to serve static files from the 'acuas-1.0.0' folder
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('acuas-1.0.0', filename)

# Define a route to handle image uploads
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        filename = secure_filename(image.filename)
        image.save(os.path.join('uploads', filename))
        return jsonify({'imageUrl': f'/uploads/{filename}'})

# Define a route to handle image transformation
@app.route('/transform-image', methods=['POST'])
def transform_image():
    if request.method == 'POST':
        image = request.files['image']
        color_blindness_type = request.form.get('colorBlindnessType')

        print("Received image:", image)
        print("Received color blindness type:", color_blindness_type)

        filename = secure_filename(image.filename)
        image.save(os.path.join('uploads', filename))

        # Load the uploaded image using OpenCV
        image_path = os.path.join('uploads', filename)
        image_cv = cv2.imread(image_path)

        # Detect the pattern and transform the color
        pattern_mask = detect_pattern(image_cv)
        print("Transforming image...")
        transformed_image = detect_and_transform_color(image_cv, pattern_mask, color_blindness_type)

        print("Transformed image shape:", transformed_image.shape)
        print("Transformed image dtype:", transformed_image.dtype)
        print("Transformed image data:", transformed_image.data)

        transformed_image_path = os.path.join('transformed', f'transformed_{filename}')
        cv2.imwrite(transformed_image_path, transformed_image)

        transformed_image_url = f'/transformed/transformed_{filename}'

        return jsonify({'imageUrl': transformed_image_url})
@app.route('/transformed/<filename>')
def serve_transformed_image(filename):
    return send_from_directory('transformed', filename)
    # Run the application if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
