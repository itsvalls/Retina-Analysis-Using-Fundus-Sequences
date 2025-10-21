import cv2
import os
import pandas as pd

def overlay_prediction(image_path, label, prob, output_path):
    img = cv2.imread(image_path)

    
    text = f"{label} ({prob*100:.1f}%)"

    
    color = (0, 0, 255) if label == "Diseased" else (0, 255, 0)

    
    cv2.rectangle(img, (10, 10), (360, 60), (0, 0, 0), -1)

    
    cv2.putText(img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    
    cv2.imwrite(output_path, img)

def visualize_all(input_folder, output_folder, csv_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        input_img = os.path.join(input_folder, row['Frame'])
        output_img = os.path.join(output_folder, row['Frame'])
        overlay_prediction(input_img, row['Prediction'], row['Probability'], output_img)

    print(f" Annotated frames saved in: {output_folder}")

if __name__ == "__main__":
    visualize_all("data/frames", "data/annotated_frames", "predictions.csv")
