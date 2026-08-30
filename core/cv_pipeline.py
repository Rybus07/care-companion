# core/cv_pipeline.py
import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    RobertaTokenizer
)

def get_compute_device() -> torch.device:
    """Selects Apple Silicon (MPS), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_trocr_model(model_path: str, device: torch.device):
    """Loads the fine-tuned TrOCR weights and processor."""
    if not os.path.exists(model_path):
        print(f"Path '{model_path}' not found. Falling back to base checkpoint...")
        model_path = "microsoft/trocr-base-handwritten"

    image_processor = ViTImageProcessor.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path, use_fast=False)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    model.eval()
    return model, processor


def segment_lines(image: Image.Image, min_area: int = 300):
    """Segments text lines from full-document image using OpenCV."""
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area and h >= 12 and w >= 25:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[1])

    line_crops = []
    annotated_img = img_np.copy()

    for (x, y, w, h) in boxes:
        pad_y = max(0, y - 6)
        pad_h = min(img_np.shape[0] - pad_y, h + 12)
        pad_x = max(0, x - 6)
        pad_w = min(img_np.shape[1] - pad_x, w + 12)

        crop_np = img_np[pad_y:pad_y + pad_h, pad_x:pad_x + pad_w]
        if crop_np.size > 0:
            line_crops.append(Image.fromarray(crop_np))
            cv2.rectangle(annotated_img, (pad_x, pad_y), (pad_x + pad_w, pad_y + pad_h), (0, 180, 0), 2)

    return line_crops, annotated_img


def extract_prescription_text(line_crops, model, processor, device: torch.device) -> str:
    """Runs TrOCR inference over cropped image segments."""
    extracted_lines = []

    for crop in line_crops:
        pixel_values = processor(crop, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=40,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                length_penalty=1.0
            )

        pred_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if len(pred_text) > 1:
            extracted_lines.append(pred_text)

    return "\n".join(extracted_lines)


# This was used in the final model
def segment_words(image: Image.Image, min_area: int = 150):
    """
    Segments individual word-level bounding boxes instead of full text lines.
    Optimized for word-level fine-tuned TrOCR checkpoints.
    """
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Adaptive thresholding
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 8
    )

    # Use a smaller horizontal kernel (12x2 or 8x2) to keep words separate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding boxes
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area and h >= 10 and w >= 15:
            boxes.append((x, y, w, h))
            
    # Sort boxes in standard reading order (line-by-line: top-to-bottom, left-to-right)
    def sort_reading_order(b_list, line_threshold=18):
        # Sort primarily by Y
        b_sorted = sorted(b_list, key=lambda b: b[1])
        lines = []
        current_line = []
        last_y = -1
        
        for b in b_sorted:
            if last_y == -1 or abs(b[1] - last_y) < line_threshold:
                current_line.append(b)
                last_y = b[1]
            else:
                lines.append(sorted(current_line, key=lambda b: b[0]))
                current_line = [b]
                last_y = b[1]
        if current_line:
            lines.append(sorted(current_line, key=lambda b: b[0]))
            
        return [box for line in lines for box in line]

    sorted_boxes = sort_reading_order(boxes)
    
    word_crops = []
    annotated_img = img_np.copy()
    
    for (x, y, w, h) in sorted_boxes:
        pad_y = max(0, y - 4)
        pad_h = min(img_np.shape[0] - pad_y, h + 8)
        pad_x = max(0, x - 4)
        pad_w = min(img_np.shape[1] - pad_x, w + 8)
        
        crop_np = img_np[pad_y:pad_y + pad_h, pad_x:pad_x + pad_w]
        if crop_np.size > 0:
            word_crops.append(Image.fromarray(crop_np))
            cv2.rectangle(annotated_img, (pad_x, pad_y), (pad_x + pad_w, pad_y + pad_h), (0, 180, 0), 2)
            
    return word_crops, annotated_img




# This was our utilized method as it was easier to have the model look at the writing word for word
def recognize_text_words(word_crops, model, processor, device: torch.device) -> str:
    """
    Recognizes individual word crops and reconstructs the full prescription string.
    """
    extracted_words = []

    for crop in word_crops:
        # Preprocess visual tensor
        pixel_values = processor(crop, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            # Constrained generation
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=40,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.5,
                length_penalty=1.0
            )

        word_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Filter out empty or low-confidence single-character artifacts
        if len(word_text) > 0 and word_text not in ["<s>", "</s>", "<pad>"]:
            extracted_words.append(word_text)

    return extracted_words