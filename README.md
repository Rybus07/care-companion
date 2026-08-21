# RxGuard: Handwritten Prescription Understanding & Clinical Safety Verification

RxGuard is an open-source, end-to-end clinical AI application designed to transcribe handwritten prescriptions, verify medications against the NIH RxNorm ontology, and translate complex clinical directions into clear, patient-friendly schedules. **Built for Final Project in CS 614 (Applications of Machine Learning) class @ Drexel University**

## 📌 Overview & Motivation
Handwritten clinical notes and prescriptions are notorious for ambiguous handwriting, dense medical shorthand (e.g., PO TID x 10d), and high transcription risk. Misreading a single dosage or drug name can lead to severe adverse drug events.

RxGuard tackles this problem through a neuro-symbolic approach:

**Vision**: Extracts text from handwritten prescription slips using Transformer-based OCR.

**Clinical Verification**: Validates extracted drug names and strengths against standardized medical ontologies (RxNorm) to prevent Look-Alike Sound-Alike (LASA) hallucinations.

**Plain-Language Scheduling**: Translates clinical abbreviations into an actionable, patient-accessible daily timetable.

**Interactive UI**: Provides a web-based inspector for visual grounding and human-in-the-loop verification.

## 🏗️ System Architecture

[ Prescription Image ]
         │
         ▼
[ Hugging Face TrOCR / HTR ] (Vision Feature Extractor)
         │
         ▼
[ Clinical Regex / NER Parser ] (Extracts Drug, Strength, Sig)
         │
         ▼
[ NIH RxNorm REST API ] ──► Validates Concept Unique Identifier (RxCUI)
         │              ──► Normalizes Generics / Active Ingredients
         ▼
[ Plain-Language LLM Engine ] ──► Generates 6th-Grade Daily Regimen
         │
         ▼
[ Django + Tailwind Web App ] (Interactive Split-Screen Inspector)

##c✨ Key Features

**Handwritten OCR Extraction**: Uses pre-trained/fine-tuned Transformer OCR (microsoft/trocr-base-handwritten) to transcribe cursive medical text.

**Ontology Grounding (RxNorm)**: Queries the NIH RxNav API to verify drug names, flag unknown entities, and catch potential OCR misreadings.

**Patient-Friendly Translation**: Converts latin sig abbreviations (e.g., q.h.s., b.i.d., p.r.n.) into accessible language.

**Human-in-the-Loop Interface**: Lightweight Django dashboard displaying confidence scores and editable entity fields before generating the final schedule.

## 📂 Repository Structure
Plaintext
RxGuard/
├── core/
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── engine.py          # Model loader & inference pipeline
│   │   └── rxnorm.py          # NIH RxNav API wrapper & normalization
│   ├── templates/
│   │   └── core/
│   │       ├── index.html     # Main dashboard
│   │       └── partials/      # HTMX dynamic response components
│   ├── models.py              # Upload tracking & JSON results schema
│   ├── views.py               # Request handlers & processing pipeline
│   └── apps.py                # Model weight singleton initialization
├── sample_data/               # Public benchmark samples (RxHandBD)
├── config/                    # Django project settings
├── manage.py
├── requirements.txt
└── README.md

## 🚀 Quickstart Guide

### Prerequisites
* Python 3.10+
* *(Optional)* CUDA-compatible GPU for accelerated local inference

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/RxGuard.git](https://github.com/your-username/RxGuard.git)
   cd RxGuard
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run database migrations:**
   ```bash
   python manage.py migrate
   ```

5. **Start the local server:**
   ```bash
   python manage.py runserver
   ```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## 📊 Datasets & Open Resources

**Handwriting Data**: RxHandBD Dataset (Mendeley Data / Kaggle)

**Drug Ontology & API**: NIH RxNav / RxNorm REST API

**Base Vision Model**: Microsoft TrOCR (Hugging Face)

## ⚠️ Medical Disclaimer
**This project is strictly for academic, research, and educational purposes. It is not certified for clinical use, automated dispensing, or direct medical diagnosis. Always consult a qualified healthcare professional regarding prescription medications.**

## 📄 License
This project is open-source and available under the MIT License.
