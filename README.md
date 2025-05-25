# Deep-Audio-Classifier-for-Capuchin-Bird-Density-in-Amazon-Rainforest
# 🐦 Capuchinbird Call Detector

A deep learning-based Flask web application that automatically detects **Capuchinbird (Perissocephalus tricolor)** vocalizations from forest audio recordings using **MFCC** features and a trained **1D Convolutional Neural Network (CNN)**.

This work is part of the broader project:
**"Deep Audio Classifier for Capuchin Bird Density in Amazon Rainforest"**

---

## 📌 Objective

To assist wildlife researchers and conservationists in monitoring Capuchinbird populations by automating the detection of their calls from large, noisy rainforest recordings.

---

## 📂 Folder Structure

capuchinbird-call-detector/
├── app.py # Flask backend
├── train_model.py # CNN model training
├── requirements.txt # Python dependencies
├── templates/
│ └── index.html # Web interface
├── static/
│ └── capuchin1.jpg, etc. # Bird images for display
├── uploads/ # Uploaded audio (temp storage)
├── outputs/
│ └── demo1.png, demo2.png # Screenshots of the web app
├── cnn_audio_classification.h5 # Trained CNN model (user-provided)


---

## 🧠 Model Overview

- **Input Features:** MFCC (40 coefficients)
- **Architecture:** 1D CNN with:
  - Conv1D → ReLU → MaxPooling → Conv1D → ReLU → MaxPooling → Flatten → Dense → Dropout → Output
- **Accuracy:** 98.2% on validation
- **Inference:** Sliding 5-second window with 2.5-second stride

---

## 🌐 Web Interface

Upload `.wav`, `.mp3`, or `.ogg` audio files and view the number of Capuchinbird calls detected.

### 🔎 Interface Demo

![Interface Screenshot](outputs/demo1.png)
![Results Screenshot](outputs/demo2.png)

---

## 🛠 How to Run

```bash
# 1. Clone the repository and enter the project folder
git clone https://github.com/<your-username>/Deep-Audio-Classifier-for-Capuchin-Bird-Density-in-Amazon-Rainforest.git
cd capuchinbird-call-detector

# 2. Install required packages
pip install -r requirements.txt

# 3. Add the trained model file (cnn_audio_classification.h5) to the same folder

# 4. Run the Flask app
python app.py

# 5. Open your browser at
http://127.0.0.1:5000
