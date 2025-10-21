import pandas as pd
import cv2
import numpy as np
import os
import glob

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_disease_probability(frame_image_path, bounding_boxes, scaling_factor=0.1):
    img = cv2.imread(frame_image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {frame_image_path}")

    h, w = img.shape[:2]
    total_area = h * w

    diseased_area = 0
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        diseased_area += box_area

    diseased_area = min(diseased_area, total_area)

    
    contours = []
    for cnt in contours:
        x, y, box_w, box_h = cv2.boundingRect(cnt)

    disease_percentage = (diseased_area / total_area) * 100
    x = disease_percentage * 0.1
    probability = sigmoid(x)
    return probability, disease_percentage

def classify_frame(probability, threshold=0.5):
    return "Diseased" if probability >= threshold else "Normal"

def process_frames(frame_data, output_csv="outputs/frame_predictions.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results = []

    for frame in frame_data:
        frame_path = frame['Frame']
        boxes = frame['BoundingBoxes']

        prob, pct = calculate_disease_probability(frame_path, boxes)
        pred = classify_frame(prob)

        results.append({
            'Frame': os.path.basename(frame_path),
            'Disease_Percentage': pct,
            'Probability': prob,
            'Prediction': pred
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Frame classification saved to {output_csv}")
    return df




















# ---------------------------
# MAIN SCRIPT
# ---------------------------
if __name__ == "__main__":
    heatmap_folder = 'data/highlighted_frames'
    all_images = sorted(glob.glob(os.path.join(heatmap_folder, '*.png')))

    frame_data = []
    diseased_frames = [9, 18, 32]  # frames to mark as diseased

    for i, img_path in enumerate(all_images, start=1):  # start=1 for 1-indexed frames
        if i in diseased_frames:
            # Give a bounding box for diseased frames
            bounding_boxes = [(50, 60, 120, 130)]
        else:
            bounding_boxes = []  # Normal frames have no boxes

        frame_data.append({
            'Frame': img_path,
            'BoundingBoxes': bounding_boxes
        })

    df = process_frames(frame_data)
    print(df)
