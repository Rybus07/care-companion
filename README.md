# CareCompanion: AI-Powered Medical Appointment Preparation & Follow-Up Assistant

CareCompanion is an open-source clinical communication tool designed to help patients organize their symptoms and questions before a medical visit and transform complex doctor notes or discharge summaries into clear, actionable follow-up plans afterward.

---

## 📌 Overview & Problem Statement
Navigating healthcare appointments is often overwhelming:
* **Before the visit:** Patients frequently struggle to structure their symptoms chronologically, forget key concerns under stress, or leave without asking critical questions.
* **After the visit:** Clinical instructions, medication changes, diagnostic referrals, and red-flag warning signs are frequently lost in medical jargon or forgotten.

**CareCompanion** bridges this communication gap through a two-sided natural language processing pipeline focused strictly on **organization, clarity, and patient empowerment**—without providing autonomous medical diagnoses or treatment recommendations.

---

## 🏗️ System Workflow & Architecture

[ PRE-APPOINTMENT WORKFLOW ]
User raw notes & concerns ──► [ Structured Extraction LLM ] ──► • Chief Complaint & History (HPI)
• Symptom Timeline
• Prioritized Doctor Questions
• 1-Page Printable Doctor Agenda

[ POST-APPOINTMENT WORKFLOW ]
Doctor notes / visit summary ──► [ Action Plan Extractor LLM ] ──► • Prescriptions & Schedule
• Lab / Imaging Orders to Book
• Lifestyle Modifications
• Emergency Red-Flag Warnings
• Follow-up Timeline Checklist

---

## ✨ Key Features
* **Pre-Visit Agenda Builder:** Converts unstructured thoughts and worry lists into a concise, clinical-style summary sheet for the clinician.
* **Question Prioritizer:** Helps patients formulate the top 3–5 targeted questions to ask during limited appointment time slots.
* **Post-Visit Action Item Extraction:** Automatically parses doctor notes, discharge sheets, or visit transcripts into a structured, step-by-step checklist.
* **Medication & Referral Tracker:** Catalogs new prescriptions, dosage instructions, and required specialist referrals.
* **Red-Flag Warning Highlighter:** Pinpoints specific symptoms noted by the clinician that warrant immediate emergency attention.
* **Non-Diagnostic Safety Design:** Structured explicitly to assist communication and administrative follow-through rather than generating diagnostic assertions.

---

## 📂 Repository Structure

```text
CareCompanion/
├── core/
│   ├── nlp/
│   │   ├── engine.py                            # LLM model scripting
│   │   ├── markdown_display.py                  # JSON to Markdown display code
│   │   ├── ocr_engine.py                        # Pre-appointment structuring engine
│   │   └── schemas.py                           # Pydantic / JSON output contracts
│   ├── image-to-text/
│   │   └── train_trocr.py
│   ├── app.py/                                  # Testing of Front End Rendering
│   └── cv_pipeline.py                           # Pipeline for TrOCR model
├── notebooks/ 
│   ├── 01_Training Image-to-Text Model.ipynb    # Training TrOCR model
│   ├── 02_Testing Image-to-Text Model.ipynb     # Testing TrOCR on curated Images
│   ├── 03_App Functionality Demo.ipynb          # Testing complete app capabilities
│   ├── Medical Prescription Reader.ipynb
│   ├── Training Dataset Demo.ipynb
│   └── appointment_test.ipynb 
├── .gitignore/                                   
├── CareCompanion_Pitch_Dexk_CS614.pdf           # PDF of Powerpoint Slides
├── README.md
└── requirements.txt
```
---

## 🚀 Quickstart Guide
**Prerequisites**
- Python 3.10+

- Local LLM runner (e.g., Ollama with llama3.2:3b or mistral) OR an API key for any standard inference provider

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/CareCompanion.git](https://github.com/your-username/CareCompanion.git)
   cd CareCompanion
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

4. **Ensure your local LLM engine is active (e.g., using Ollama):**
   ```bash
   ollama run llama3.2:3b
   ```

5. **Run the following notebooks in this order to recreate project:**
   ```bash
   1. 01_Training Image-to-Text Model.ipynb
   2. 02_Testing Image-to-Text Model.ipynb
   3. 03_App Functionality Demo.ipynb
   ```


## ⚠️ Medical & Ethical Disclaimer
CareCompanion is strictly an organizational and communication support tool. It is not a certified medical device, does not provide medical advice, and is not designed to diagnose, treat, cure, or prevent any health condition. Users should always consult a licensed healthcare professional for medical concerns.

## 📄 License
Distributed under the MIT License.
