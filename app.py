from flask import Flask, request, render_template, jsonify
import os
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load model
model = joblib.load('waste_model.joblib')
img_size = (128, 128)
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save uploaded file temporarily
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Preprocess image
    img = load_img(file_path, target_size=img_size)
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Prediction
    pred = model.predict(x)
    pred_class = class_names[np.argmax(pred)]
    confidence = float(np.max(pred))

    os.remove(file_path)

    if pred_class not in ['trash']:
        return jsonify({'class': 'Recyclable', 'confidence': confidence})
    else :
        return jsonify({'class':'Non-Recyclable', 'confidence': confidence})
    
    # return jsonify({'class':pred_class, 'confidence':confidence})
if __name__ == '__main__':
    app.run(debug=True)




