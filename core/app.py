import json
import requests
import streamlit as st
from pydantic import BaseModel, Field
from typing import List

# --- 1. SCHEMAS ---
class SymptomItem(BaseModel):
    symptom: str
    duration_or_onset: str

class PreVisitAgenda(BaseModel):
    chief_complaint: str = Field(description="Primary reason for visit in one concise sentence")
    symptom_timeline: List[SymptomItem]
    current_medications_mentioned: List[str]
    prioritized_questions_for_doctor: List[str]

# --- 2. NLP ENGINE ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

def call_ollama(prompt: str, system: str, format_json: bool = False) -> str:
    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "system": system,
        "stream": False
    }
    if format_json:
        payload["format"] = "json"
    
    res = requests.post(OLLAMA_ENDPOINT, json=payload).json()
    return res.get("response", "")

def generate_agenda(notes: str) -> dict:
    schema = json.dumps(PreVisitAgenda.model_json_schema(), indent=2)
    system = f"You are a clinical communication assistant. Summarize notes strictly into this JSON schema:\n{schema}"
    raw = call_ollama(notes, system, format_json=True)
    return json.loads(raw)

def answer_user_qa(notes: str, agenda: dict, question: str) -> str:
    system = "Explain your reasoning based STRICTLY on the original patient notes. Quote relevant phrases."
    context = f"NOTES:\n{notes}\n\nAGENDA:\n{json.dumps(agenda, indent=2)}\n\nQUESTION:\n{question}"
    return call_ollama(context, system, format_json=False)

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="CareCompanion", page_icon="🩺", layout="wide")

st.title("🩺 CareCompanion")
st.caption("AI-supported appointment preparation and grounded Q&A")

# Initialize Session State
if "agenda" not in st.session_state:
    st.session_state.agenda = None
if "notes" not in st.session_state:
    st.session_state.notes = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Input
with st.sidebar:
    st.header("1. Your Appointment Notes")
    user_input = st.text_area(
        "Enter symptoms, worries, or medications:",
        value=st.session_state.notes,
        height=220,
        placeholder="e.g., Lower back pain for 3 weeks, worse in the morning. Taking Advil 200mg. Worried about a disc issue..."
    )
    if st.button("Generate Agenda", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing and structuring your agenda..."):
                st.session_state.notes = user_input
                st.session_state.agenda = generate_agenda(user_input)
                st.session_state.messages = [
                    {"role": "assistant", "content": "I've generated your appointment summary. Ask me anything about why specific items or questions were included!"}
                ]
        else:
            st.warning("Please enter your concerns first.")

# Main Display Area
if st.session_state.agenda:
    agenda = st.session_state.agenda
    
    col1, col2 = st.columns([1.2, 1], gap="large")

    # Column 1: The Generated Document
    with col1:
        st.subheader("📋 Generated Appointment Agenda")
        
        st.info(f"**Chief Complaint:** {agenda.get('chief_complaint')}")
        
        st.markdown("#### ⏳ Symptom Timeline")
        for item in agenda.get("symptom_timeline", []):
            st.markdown(f"- **{item.get('symptom')}** ({item.get('duration_or_onset')})")
            
        st.markdown("#### ❓ Prioritized Questions for Doctor")
        for idx, q in enumerate(agenda.get("prioritized_questions_for_doctor", []), 1):
            st.markdown(f"**{idx}.** {q}")
            
        if agenda.get("current_medications_mentioned"):
            st.markdown("#### 💊 Current Medications")
            st.write(", ".join(agenda.get("current_medications_mentioned", [])))

    # Column 2: Grounded Q&A Chat
    with col2:
        st.subheader("💬 Ask About This Agenda")
        
        # Chat container
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # Chat input
        if prompt := st.chat_input("Why was question #2 added?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)

            # Generate grounded answer
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Checking your original notes..."):
                        response = answer_user_qa(st.session_state.notes, agenda, prompt)
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👈 Enter your symptoms in the sidebar and click **Generate Agenda** to start.")