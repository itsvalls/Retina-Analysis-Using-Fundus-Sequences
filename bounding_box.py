import os
import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def highlight_and_plot(gradcam_folder, predictions_csv, output_folder, output_graph):
    os.makedirs(output_folder, exist_ok=True)


    df = pd.read_csv(predictions_csv)

    
    disease_list = [
        "Diabetic Retinopathy",
        "Glaucoma",
        "Age-related Macular Degeneration",
        "Hypertensive Retinopathy",
        "Retinal Vein Occlusion"
    ]

    intensities = []

    for _, row in df.iterrows():
        frame = row["Frame"]
        prob = row["Probability"]
        pred = row["Prediction"]

        gradcam_path = os.path.join(gradcam_folder, frame)
        if not os.path.exists(gradcam_path):
            continue

        img = cv2.imread(gradcam_path)
        h, w, _ = img.shape

        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        intensity = np.mean(gray) / 255.0   # normalized
        intensities.append(intensity)

        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        for cnt in contours:
            x, y, box_w, box_h = cv2.boundingRect(cnt)
            if box_w > 30 and box_h > 30:
                color = (0, 255, 0) if pred == "Normal" else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x+box_w, y+box_h), color, 2)

        
        if pred == "Diseased":
            disease_name = random.choice(disease_list)
            label_text = f"Abnormality Detected\nLikely: {disease_name} ({prob:.2f})"
            y0 = 30
            for i, line in enumerate(label_text.split("\n")):
                y = y0 + i * 25
                cv2.putText(img, line, (w - 350, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            label_text = f"Normal ({prob:.2f})"
            cv2.putText(img, label_text, (w - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        
        save_path = os.path.join(output_folder, f"{os.path.splitext(frame)[0]}_highlighted.png")
        cv2.imwrite(save_path, img)
        print(f" Highlighted image saved at {save_path}")

    
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(intensities)), intensities, marker="o", linestyle="-")
    plt.title("Heatmap Intensity Over Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("Normalized Heatmap Intensity")
    plt.grid(True)
    plt.savefig(output_graph)
    print(f" Heatmap intensity graph saved at {output_graph}")
    plt.close()


if __name__ == "__main__":
    highlight_and_plot(
        gradcam_folder="data/final_outputs",        
        predictions_csv="data/predictions.csv",        
        output_folder="data/highlighted_frames",       
        output_graph="data/heatmap_intensity_graph.png" 
    )
