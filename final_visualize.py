import os
import cv2
import pandas as pd
import numpy as np

def blend_images(original, heatmap, alpha=0.6):
    """Blend heatmap onto original image."""
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(heatmap_colored, alpha, original, 1 - alpha, 0)
    return blended

def load_heatmap(path, size):
    heatmap = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    heatmap = cv2.resize(heatmap, size)
    heatmap = heatmap.astype(np.float32) / 255
    return heatmap

def annotate_image(image, label, confidence):
    text = f"{label} ({confidence*100:.1f}%)"
    color = (0, 255, 0) if label == "Normal" else (0, 0, 255)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image

def generate_visualizations(frames_dir, heatmaps_dir, predictions_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(predictions_csv)

    for idx, row in df.iterrows():
       for idx, row in df.iterrows():
        filename = row["Frame"]
        label = row["Prediction"]
        confidence = row["Probability"]

        original_path = os.path.join(frames_dir, filename)
        heatmap_path = os.path.join(heatmaps_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(original_path) or not os.path.exists(heatmap_path):
            continue

        original = cv2.imread(original_path)
        heatmap = load_heatmap(heatmap_path, (original.shape[1], original.shape[0]))
        blended = blend_images(original, heatmap)

        annotated = annotate_image(blended, label, confidence)
        cv2.imwrite(output_path, annotated)

    print(f"Annotated images saved to: {output_dir}")

if __name__ == "__main__":
    generate_visualizations(
        frames_dir="data/frames",
        heatmaps_dir="data/gradcam_frames",
        predictions_csv="data/predictions.csv",
        output_dir="data/final_outputs"
    )
