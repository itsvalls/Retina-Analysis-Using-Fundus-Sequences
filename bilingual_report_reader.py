import pyttsx3
from PyPDF2 import PdfReader
from googletrans import Translator
import pytesseract
from PIL import Image
import fitz  

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(" PyPDF2 failed to extract text:", e)

    
    if not text.strip():
        print(" Using OCR fallback...")
        text = ocr_extract(pdf_path)

    return text.strip()

def ocr_extract(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img, lang="eng")
    return text

def bilingual_read_report(pdf_path):
    print(" Extracting text from report...")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print(" No readable text found in PDF.")
        return

    print("\n Extracted report text:\n", text[:500], "..." if len(text) > 500 else "")

    engine = pyttsx3.init()
    translator = Translator()

    
    print("\n🎧 Reading report in English...")
    engine.say(text)
    engine.runAndWait()

    
    print("\n Translating to Hindi...")
    translated = translator.translate(text, src="en", dest="hi").text

    print("\n Reading report in Hindi...")
    engine.say(translated)
    engine.runAndWait()

    print("\n Finished reading report in both English and Hindi!")


bilingual_read_report("outputs/final_report.pdf")
