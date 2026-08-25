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
│   │   ├── __init__.py
│   │   ├── schemas.py         # Pydantic / JSON output contracts
│   │   ├── pre_visit.py       # Pre-appointment structuring engine
│   │   └── post_visit.py      # Post-appointment action plan extractor
│   ├── templates/
│   │   └── core/
│   │       ├── base.html
│   │       ├── pre_visit.html # Intake form & generated agenda
│   │       ├── post_visit.html# Note parser & action checklist
│   │       └── partials/      # Dynamic HTMX response cards
│   ├── models.py              # Appointment, Agenda, and ActionPlan models
│   ├── views.py               # View endpoints for processing notes
│   ├── urls.py
│   └── apps.py
├── sample_data/               # Sample patient concerns & doctor notes
├── config/                    # Django project configuration
├── manage.py
├── requirements.txt
└── README.md
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

5. **Run database migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Start the local server:**
   ```bash
   python manage.py runserver
   ```

7. **Open the application:**
   Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## ⚠️ Medical & Ethical Disclaimer
CareCompanion is strictly an organizational and communication support tool. It is not a certified medical device, does not provide medical advice, and is not designed to diagnose, treat, cure, or prevent any health condition. Users should always consult a licensed healthcare professional for medical concerns.

## 📄 License
Distributed under the MIT License.
