import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from preprocess import load_and_preprocess_images
import pandas as pd

def build_model():
   
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x) # Normal vs Diseased
    
    model = Model(inputs=base_model.input, outputs=output)
    return model

def predict_disease(model, images):
   
    preds = model.predict(images, verbose=0)
    
    
    noise = np.random.normal(loc=0.0, scale=0.05, size=preds.shape)
    preds = preds + noise
    preds = np.clip(preds, 0, 1)  

    predicted_labels = (preds > 0.5).astype(int).flatten()
    return predicted_labels, preds.flatten()

if __name__ == "__main__":
    print(" Loading preprocessed frames...")
    images, names = load_and_preprocess_images("data/frames")

    print(" Building EfficientNet model...")
    model = build_model()

    print(" Predicting diseases...")
    labels, probs = predict_disease(model, images)

    print(" Saving predictions to CSV...")
    df = pd.DataFrame({
        "Frame": names,
        "Probability": probs,
        "Prediction": ["Diseased" if l == 1 else "Normal" for l in labels]
    })
    df.to_csv("predictions.csv", index=False)

    print(" Done! Sample output:")
    for name, label, prob in zip(names, labels, probs):
        status = "Diseased" if label == 1 else "Normal"
        print(f"{name}: {status} ({prob:.2f})")
