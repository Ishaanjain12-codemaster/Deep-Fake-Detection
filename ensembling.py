import numpy as np
from tensorflow.keras.models import load_model

cnn_model = load_model('cnn_model.h5')         
vit_model = load_model('vit_model.h5')         
gating_model = load_model('gating_cnn.h5')     

def preprocess_input(image):
    # Resize and normalize for CNN and ViT
    cnn_input = cv2.resize(image, (224, 224)) / 255.0
    vit_input = cv2.resize(image, (224, 224)) / 255.0
    return np.expand_dims(cnn_input, axis=0), np.expand_dims(vit_input, axis=0)

def extract_features(image):
    # Example features: blur, brightness, face size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    face_size = min(image.shape[0], image.shape[1]) / 2
    return np.array([[blur, brightness, face_size]])

def ensemble_predict(image):
    cnn_input, vit_input = preprocess_input(image)
    features = extract_features(image)

    p_cnn = cnn_model.predict(cnn_input)[0][0]
    p_vit = vit_model.predict(vit_input)[0][0]

    g = gating_model.predict(features)[0][0]  # G ∈ [0, 1]

    p_final = g * p_cnn + (1 - g) * p_vit
    label = "Deepfake" if p_final > 0.5 else "Real"

    return {
        "P_CNN": p_cnn,
        "P_ViT": p_vit,
        "Gating_Score": g,
        "P_Final": p_final,
        "Prediction": label
    }

import cv2
image = cv2.imread("sample_frame.jpg")  # Replace with your input frame
result = ensemble_predict(image)
print(result)
