import streamlit as st
from PIL import Image
from nlp.ocr_engine import transcribe_clinical_note
from nlp.engine import (
    generate_pre_visit_agenda, 
    generate_post_visit_actions, 
    answer_grounded_qa
)

st.set_page_config(page_title="CareCompanion", page_icon="🩺", layout="wide")

st.title("🩺 CareCompanion")
st.caption("AI-Powered Clinical Preparation & Care Plan Assistant")

# --- Session State Initialization ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Pre-Appointment Prep"
if "extracted_result" not in st.session_state:
    st.session_state.extracted_result = None
if "context_text" not in st.session_state:
    st.session_state.context_text = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Workflow Mode")
    
    # Toggle button group for switching LLM approach
    mode = st.radio(
        "Select Appointment Stage:",
        options=["Pre-Appointment Prep", "Post-Appointment Care"],
        horizontal=True
    )
    
    # Reset state if mode changes
    if mode != st.session_state.app_mode:
        st.session_state.app_mode = mode
        st.session_state.extracted_result = None
        st.session_state.context_text = ""
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Dynamic Inputs Based on Selected Mode
    if mode == "Pre-Appointment Prep":
        st.subheader("1. Your Concerns & Symptoms")
        input_label = "Describe your symptoms, timeline, or concerns:"
        input_placeholder = "e.g., Lower back pain for 3 weeks, worse in the morning. Taking Advil 200mg..."
        img_label = "Upload previous slip / handwritten notes (Optional):"
        button_label = "Generate Doctor Agenda"
    else:
        st.subheader("1. Doctor's Clinical Notes / Slip")
        input_label = "Paste typed discharge instructions or physician notes:"
        input_placeholder = "e.g., Dx: Lumbar strain. PT 2x/wk. Naproxen 500mg BID with food. Avoid heavy lifting..."
        img_label = "Upload handwritten prescription or doctor note:"
        button_label = "Extract Action & Care Plan"

    user_text = st.text_area(input_label, height=160, placeholder=input_placeholder)
    uploaded_img = st.file_uploader(img_label, type=["png", "jpg", "jpeg"])

    if uploaded_img:
        st.image(uploaded_img, caption="Document Preview", use_container_width=True)

    if st.button(button_label, type="primary", use_container_width=True):
        if user_text.strip() or uploaded_img:
            ocr_text = ""
            
            # Stage 1: Vision Transformer Transcription
            if uploaded_img:
                with st.spinner("Running TrOCR Vision Transformer on handwriting..."):
                    try:
                        img = Image.open(uploaded_img)
                        ocr_text = transcribe_clinical_note(img)
                        st.info(f"**TrOCR Transcription:** {ocr_text}")
                    except Exception as e:
                        st.warning(f"OCR failed to process image: {e}")

            # Combine typed text + OCR output
            combined_input = user_text
            if ocr_text:
                combined_input = f"{user_text}\n[Handwritten Note Extracted]: {ocr_text}".strip()

            st.session_state.context_text = combined_input

            # Stage 2: Route to Appropriate LLM Approach
            with st.spinner(f"Analyzing via {mode} approach..."):
                try:
                    if mode == "Pre-Appointment Prep":
                        result = generate_pre_visit_agenda(combined_input)
                        initial_msg = "I've structured your doctor appointment agenda. Ask me why any item was included!"
                    else:
                        result = generate_post_visit_actions(combined_input)
                        initial_msg = "I've extracted your medication schedule and lifestyle guidance. Ask me about any instructions!"

                    st.session_state.extracted_result = result
                    st.session_state.messages = [{"role": "assistant", "content": initial_msg}]
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing error: {e}")
        else:
            st.warning("Please provide notes or upload an image first.")

# --- Main Dashboard Display ---
if st.session_state.extracted_result:
    result = st.session_state.extracted_result
    col_doc, col_chat = st.columns([1.2, 1], gap="large")

    with col_doc:
        # Render Pre-Appointment View
        if st.session_state.app_mode == "Pre-Appointment Prep":
            st.subheader("📋 Pre-Visit Doctor Agenda")
            st.info(f"**Chief Complaint:** {result.get('chief_complaint', 'N/A')}")
            
            st.markdown("#### ⏳ Symptom Timeline")
            for item in result.get("symptom_timeline", []):
                st.markdown(f"- **{item.get('symptom')}** ({item.get('duration_or_onset')})")
                
            st.markdown("#### ❓ Prioritized Questions for Doctor")
            for idx, q in enumerate(result.get("prioritized_questions_for_doctor", []), 1):
                st.markdown(f"**{idx}.** {q}")
                
            if result.get("current_medications_mentioned"):
                st.markdown("#### 💊 Mentioned Medications")
                st.write(", ".join(result.get("current_medications_mentioned", [])))

        # Render Post-Appointment View
        else:
            st.subheader("📋 Post-Visit Action Plan & Lifestyle Care")
            st.info(f"**Assessment Summary:** {result.get('summary_of_visit', 'N/A')}")

            st.markdown("#### 🏃 Lifestyle & Home Care Instructions")
            for care in result.get("lifestyle_and_home_care", []):
                st.markdown(f"- {care}")

            st.markdown("#### 💊 Prescribed Medications")
            for med in result.get("medications", []):
                st.markdown(f"- **{med.get('name')}** ({med.get('dosage')}): {med.get('instructions')}")

            if result.get("tests_or_referrals"):
                st.markdown("#### 🧪 Tests & Follow-up Orders")
                for test in result.get("tests_or_referrals", []):
                    st.markdown(f"- {test}")

            if result.get("red_flag_warnings"):
                st.error(f"🚨 **Red Flag Warnings:** {', '.join(result.get('red_flag_warnings'))}")
            
            st.markdown(f"**Follow-Up Timeline:** {result.get('follow_up_timeline', 'As needed')}")

    # Right Column: Grounded Q&A Chat (Works for Both Modes)
    with col_chat:
        st.subheader("💬 Ask Context Q&A")
        chat_box = st.container(height=420)
        
        with chat_box:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        if prompt := st.chat_input("Ask a question about this summary..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_box:
                with st.chat_message("user"):
                    st.write(prompt)

            with chat_box:
                with st.chat_message("assistant"):
                    with st.spinner("Referencing original context..."):
                        reply = answer_grounded_qa(
                            st.session_state.context_text, 
                            st.session_state.extracted_result, 
                            prompt
                        )
                        st.write(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
else:
    st.info("👈 Select a mode, enter notes or upload an image in the sidebar, and click generate.")