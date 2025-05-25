from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model("cnn_audio_classification.h5")

def process_audio(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_best')
    total_duration = librosa.get_duration(y=audio, sr=sample_rate)
    detections = []

    for start in np.arange(0, total_duration - 5, 2.5):
        end = start + 5
        audio_segment = audio[int(start * sample_rate):int(end * sample_rate)]
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        prediction = model.predict(np.expand_dims(np.expand_dims(mfccs_scaled, axis=-1), axis=0), verbose=0)

        if np.argmax(prediction) == 0:  # 0 = Capuchinbird
            detections.append(start)

    return len(detections)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "audio" not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    capuchin_calls = process_audio(file_path)
    return jsonify({"capuchin_calls": capuchin_calls})

if __name__ == "__main__":
    app.run(debug=True)
