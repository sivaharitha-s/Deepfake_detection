from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model('model/model_finetuned.h5')

# Prediction function
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    score = model.predict(img)[0][0]
    return ("REAL", score) if score >= 0.5 else ("FAKE", score)

# Home page: index.html
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction, score = predict_image(filepath)
            return render_template('index.html', filename=file.filename, prediction=prediction, score=score)
    return render_template('index.html', filename=None)

# Detection History page: history.html
@app.route('/history')
def history():
    upload_folder = app.config['UPLOAD_FOLDER']
    files = os.listdir(upload_folder)
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort(reverse=True)  # latest uploads first
    return render_template('history.html', images=images)

# About page: about.html
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
