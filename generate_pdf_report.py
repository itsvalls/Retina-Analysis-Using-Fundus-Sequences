from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from datetime import datetime
import pandas as pd
import os
import textwrap  

def generate_pdf_report(output_path, most_abnormal_frame, max_prob, disease_label,
                        student_name="Dr Sharma", frame_img_path=None):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin

    
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, y, "Retina Diagnostics")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, y, "Retinal Video Diagnostic Report")
    y -= 20

    c.line(margin, y, width - margin, y)
    y -= 30

    
    c.setFont("Helvetica", 10)
   
    c.drawRightString(width - margin, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 40

    
    image_width = 180
    image_height = 180
    text_start_x = margin + image_width + 20
    image_inserted = False

    if frame_img_path and os.path.exists(frame_img_path):
        try:
            c.drawImage(frame_img_path, margin, y - image_height + 10, width=image_width,
                        height=image_height, preserveAspectRatio=True)
            image_inserted = True
        except Exception as e:
            print(f" Could not insert image: {e}")

    
    y_summary_top = y
    y = y_summary_top

    c.setFont("Helvetica-Bold", 12)
    c.drawString(text_start_x if image_inserted else margin, y, "Diagnosis Summary:")
    y -= 25

    c.setFont("Helvetica", 11)
    lines = [
        "This result is based on spatio-temporal patterns analyzed.",
       
        "Please note:",
        "• This report provides a preliminary analysis.", 
        "• For any concerns or unusual symptoms, consult an ophthalmologist.",
        "• Follow-up tests may be recommended based on this result."
    ]

    
    max_line_chars = 90
    for line in lines:
        wrapped = textwrap.wrap(line, width=max_line_chars)
        for wline in wrapped:
            c.drawString(text_start_x if image_inserted else margin, y, wline)
            y -= 18

    if image_inserted:
        y = min(y, y_summary_top - image_height - 20)

    
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Recommended Follow-up Tests:")
    y -= 20

    tests = [
        ["Test", "Purpose"],
        ["OCT Scan", "Detailed retinal layers analysis"],
        ["Fluorescein Angiography", "Visualize blood vessels"],
        ["Visual Acuity Test", "Measure sharpness of vision"],
        ["Tonometry", "Measure eye pressure"]
    ]

    col_widths = [150, 300]
    row_height = 20
    for i, row in enumerate(tests):
        bg_color = colors.whitesmoke if i % 2 == 0 else colors.lightgrey
        c.setFillColor(bg_color)
        c.rect(margin, y - row_height, sum(col_widths), row_height, fill=True, stroke=False)
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 10)
        c.drawString(margin + 5, y - 15, row[0])
        c.drawString(margin + col_widths[0] + 5, y - 15, row[1])
        y -= row_height

    # General Precautions
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "⚠️ General Precautions:")
    y -= 20

    precautions = [
        "• Avoid excessive screen time.",
        "• Maintain blood sugar and blood pressure levels.",
        "• Schedule regular eye check-ups.",
        "• Report any sudden vision changes to a specialist."
    ]

    c.setFont("Helvetica", 10)
    for line in precautions:
        wrapped = textwrap.wrap(line, width=90)
        for wline in wrapped:
            c.drawString(margin + 10, y, wline)
            y -= 15

    # Signature
    y = 100
    c.setFont("Helvetica-Oblique", 16)
    c.drawRightString(width - margin, y, "Dr. Sharma")
    c.setFont("Helvetica", 9)
    c.drawRightString(width - margin, y - 15, "Consultant Ophthalmologist")

    # Save PDF
    c.save()
    print(f" Report saved to {output_path}")



csv_path = "data/predictions.csv"
df = pd.read_csv(csv_path)


df.columns = df.columns.str.strip().str.lower()
df.rename(columns={"frame": "filename", "prediction": "label", "probability": "prob"}, inplace=True)

df["prob"] = pd.to_numeric(df["prob"], errors='coerce')
idx = df["prob"].idxmax()
most_abnormal_frame = df.loc[idx, "filename"]
max_prob = df.loc[idx, "prob"]
disease_label = df.loc[idx, "label"]

os.makedirs("outputs", exist_ok=True)
frame_img_path = os.path.join("data", "frames", most_abnormal_frame)
if not os.path.exists(frame_img_path):
    frame_img_path = None

generate_pdf_report(
    output_path="outputs/final_report.pdf",
    most_abnormal_frame=most_abnormal_frame,
    max_prob=max_prob,
    disease_label=disease_label,
    student_name="Vallari Sharma",
    frame_img_path=frame_img_path
)
