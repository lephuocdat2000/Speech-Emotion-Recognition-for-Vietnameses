import os, glob
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
import numpy as np
import time
import librosa.display
from tensorflow import keras
from SER.test import test
from SER.test_CNN import predict_emotion_from_file

UPLOAD_FOLDER = 'static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
ALLOWED_EXTENSIONS = {'jpg', 'png', '.jpeg'}
app = Flask(__name__, static_url_path="/static")

# APP CONFIGURATIONS
app.config['SECRET_KEY'] = 'opencv'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 6mb
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024

#model_cnn1 = keras.models.load_model('./models/combined_model.h5')
#model_cnn2 = keras.models.load_model('./models/eng-vie-model.h5')
model_cnn3 = keras.models.load_model('./models/vie-vie-model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file attached in request')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        # if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # emotion = test(file_path)[0]
        emotion = predict_emotion_from_file(model_cnn3,file_path)[0]

        if emotion == 1:
            emotion = 4

        elif emotion == 5:
            emotion = 3

        elif emotion == 0:
            emotion = 1

        data = {
            "result_emotion": 'rating-' + str(emotion)
        }
        return render_template("index.html", data=data)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
