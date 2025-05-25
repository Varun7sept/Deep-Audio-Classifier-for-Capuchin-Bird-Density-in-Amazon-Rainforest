# 5.1: Importing Required Libraries
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

# 5.2: Defining file paths for audio dataset
CAPUCHIN_DIR = 'dataset/capuchin'
NON_CAPUCHIN_DIR = 'dataset/not_capuchin'

# 5.3, 5.5: Loading and displaying example audio waveforms
cap_file = os.path.join(CAPUCHIN_DIR, os.listdir(CAPUCHIN_DIR)[0])
noncap_file = os.path.join(NON_CAPUCHIN_DIR, os.listdir(NON_CAPUCHIN_DIR)[0])

y_cap, sr_cap = librosa.load(cap_file)
y_noncap, sr_noncap = librosa.load(noncap_file)

plt.figure(figsize=(10, 3))
librosa.display.waveshow(y_cap, sr=sr_cap)
plt.title("Capuchinbird Call")
ipd.Audio(y_cap, rate=sr_cap)

plt.figure(figsize=(10, 3))
librosa.display.waveshow(y_noncap, sr=sr_noncap)
plt.title("Non-Capuchinbird Sound")
ipd.Audio(y_noncap, rate=sr_noncap)

# 5.7: Feature extraction function
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_best')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 5.8: Extracting features from audio directory
extracted_features = []

for label, folder in [('capuchin', CAPUCHIN_DIR), ('not_capuchin', NON_CAPUCHIN_DIR)]:
    for file in tqdm(os.listdir(folder)):
        file_path = os.path.join(folder, file)
        data = features_extractor(file_path)
        extracted_features.append([data, label])

# 5.9: Creating DataFrame
extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class_label'])

# 5.11: Preparing data
X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class_label'].tolist())

le = LabelEncoder()
y = to_categorical(le.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# 5.12: CNN Model Architecture
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(40, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(32, 3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])

# 5.13: Compiling and Training Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('cnn_audio_classification.h5', monitor='val_accuracy', save_best_only=True)

start = datetime.now()
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=25, callbacks=[checkpoint])
end = datetime.now()

print("Training Time:", end - start)

# 5.14: Inference on Single File
def process_audio(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_best')
    total_duration = librosa.get_duration(y=audio, sr=sr)
    window = 5
    stride = 2.5
    calls = 0
    for start in np.arange(0, total_duration - window, stride):
        end = start + window
        segment = audio[int(start * sr):int(end * sr)]
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40)
        features = np.mean(mfcc.T, axis=0)
        features = features.reshape(1, 40, 1)
        pred = model.predict(features, verbose=0)
        if np.argmax(pred) == 0:
            calls += 1
    return calls

# 5.15: Processing All Test Files
def batch_process(folder):
    results = []
    for file in os.listdir(folder):
        if file.endswith((".mp3", ".wav", ".ogg")):
            path = os.path.join(folder, file)
            count = process_audio(path)
            results.append([file, count])
    return results

# Example usage
# print(batch_process("test_recordings"))

# 5.16: Saving Results to CSV
results = batch_process("test_audio")
df_results = pd.DataFrame(results, columns=["Filename", "Capuchin_Calls"])
df_results.to_csv("capuchin_detection_results.csv", index=False)
