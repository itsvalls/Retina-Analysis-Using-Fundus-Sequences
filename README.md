#  Automated Retinal Disease Detection using Deep Learning and Explainable AI  
 

---

##  Overview
This project presents a deep learning–based framework for **automated detection and explainable visualization of retinal diseases** from fundus video sequences.  
The system integrates **frame-level disease classification**, **Grad-CAM heatmap visualization**, **severity grading**, and **bilingual audio narration** for accessibility.  

It is designed as a clinical decision-support prototype that can assist ophthalmologists in screening for diabetic retinopathy and other retinal abnormalities.

---

##  <img width="1350" height="909" alt="Screenshot 2025-10-20 025457" src="https://github.com/user-attachments/assets/93b1eb34-c2de-4479-8c62-9264a6cfbd88" />


<img width="1735" height="896" alt="Screenshot 2025-10-16 101531" src="https://github.com/user-attachments/assets/107fea2b-e346-46f1-80b7-8d74cd5f96cf" />



<img width="1055" height="898" alt="Screenshot 2025-10-20 025205" src="https://github.com/user-attachments/assets/28ab173a-31d7-4241-9d66-a9787a14a295" />

<img width="1189" height="497" alt="Screenshot 2025-10-16 101509" src="https://github.com/user-attachments/assets/fcfa9a67-cdf9-4d22-b07b-29498799eb1a" />

---
ey Features
-  **Frame Extraction & Preprocessing:** Converts retinal fundus videos into individual frames.  
-  **Disease Classification:** Uses a lightweight EfficientNet-B0 model to classify each frame as *Normal* or *Diseased* based on bounding-box features.  
-  **Explainable AI (Grad-CAM):** Generates heatmaps highlighting regions of clinical concern such as microaneurysms and hemorrhages.  
-  **Severity Grading:** Automatically labels diseased frames as *Mild*, *Moderate*, *Severe*, or *Proliferative* based on probability scores.  
-  **Automated Report Generation:** Compiles frame-wise predictions into a structured report and visualization summary.  
-  **Bilingual Audio Narration:** Reads the diagnostic summary in **English and Hindi**, enhancing accessibility for both doctors and patients.  
-  **Decision Tree Visualization:** Depicts the model’s reasoning pathway for each frame and generates a frame-wise classification log.

---


##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/retina_project.git
   cd retina_project

2. Install additional tools:
    ```bash
   pip install opencv-python pyttsx3 pytesseract fitz googletrans==4.0.0-rc1

  ---
  ### Technologies Used

Python 3.10+
OpenCV – Frame extraction and visualization
TensorFlow / PyTorch – Model training and inference
Matplotlib – Decision flow and probability graph
PyPDF2 & ReportLab – Report generation
Pyttsx3 – Offline bilingual text-to-speech
Googletrans – English to Hindi translation
Pandas / NumPy – Data analysis and CSV handling
---
### Potential Applications

Early diabetic retinopathy screening in rural health camps
Automated analysis for ophthalmic telemedicine system
Integration with fundus camera devices for real-time screening
Training and educational tool for medical students
