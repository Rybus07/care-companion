# nlp/ocr_engine.py
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load your fine-tuned weights or the base model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

def transcribe_prescription_image(image: Image.Image) -> str:
    """Takes a PIL image and outputs transcribed clinical text."""
    pixel_values = processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)
        
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription