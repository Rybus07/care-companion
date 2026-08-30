from IPython.display import display, Markdown

def display_pre_visit_agenda(data: dict):
    """Formats and displays a PreVisitAgenda dictionary output as clean Markdown."""
    
    # 1. Format Symptoms Table or List
    symptoms_md = ""
    if data.get("symptom_timeline"):
        symptoms_md = "\n".join([
            f"- **{item.get('symptom', item)}**: {item.get('duration', 'Duration not specified')}"
            if isinstance(item, dict) else f"- {item}"
            for item in data["symptom_timeline"]
        ])
    else:
        symptoms_md = "_None documented._"

    # 2. Format Medications List
    meds_md = "\n".join([f"- {med}" for med in data.get("current_medications_mentioned", [])]) or "_None documented._"

    # 3. Format Questions List
    questions_md = "\n".join([f"{i+1}. {q}" for i, q in enumerate(data.get("prioritized_questions_for_doctor", []))]) or "_None generated._"

    markdown_content = f"""
## Pre-Visit Agenda Summary

**Chief Complaint:**  
> {data.get('chief_complaint', 'N/A')}

---

###  Symptom Timeline
{symptoms_md}

---

### Current Medications & Supplements
{meds_md}

---

### Prioritized Questions for the Physician
{questions_md}
"""
    display(Markdown(markdown_content))


def display_post_visit_plan(data: dict):
    """Formats and displays a PostVisitPlan dictionary output as clean Markdown."""
    
    # Adjust field names below to match your PostVisitPlan Pydantic fields
    med_instructions = "\n".join([f"- {m}" for m in data.get("medications", data.get("medication_changes", []))]) or "_No changes noted._"
    action_items = "\n".join([f"- [ ] {a}" for a in data.get("action_items", data.get("to_do_list", []))]) or "_No action items._"
    red_flags = "\n".join([f"- ⚠️ {f}" for f in data.get("warning_signs", data.get("red_flags", []))]) or "_Standard clinic contact guidance applies._"
    follow_up = data.get("follow_up_schedule", data.get("follow_up", "Follow up as needed."))

    markdown_content = f"""
## 🩺 Post-Visit Action Plan & Summary

### 💊 Medication Instructions
{med_instructions}

---

### ✅ Patient Action Checklist
{action_items}

---

### 🚨 Warning Signs & When to Call the Clinic
{red_flags}

---

### 🗓️ Follow-Up Schedule
> **{follow_up}**
"""
    display(Markdown(markdown_content))