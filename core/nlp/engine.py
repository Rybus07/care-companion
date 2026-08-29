import json
import requests
from nlp.schemas import PreVisitAgenda, PostVisitPlan

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"

def call_ollama(prompt: str, system: str, format_json: bool = False, model: str = DEFAULT_MODEL) -> str:
    """Core wrapper to communicate with the local Ollama daemon."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False
    }
    if format_json:
        payload["format"] = "json"
    
    try:
        res = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data.get("response", "")
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot connect to Ollama. Make sure the Ollama app or daemon is running.")
    except Exception as e:
        raise RuntimeError(f"Ollama inference error: {str(e)}")

def generate_pre_visit_agenda(notes: str) -> dict:
    """Extracts structured appointment agenda from raw user concerns."""
    schema_json = json.dumps(PreVisitAgenda.model_json_schema(), indent=2)
    system_prompt = (
        "You are an organizational clinical communication assistant. "
        "Your task is strictly organizational (never diagnose or recommend treatments). "
        "Convert the patient's unstructured notes into JSON conforming strictly to this schema:\n"
        f"{schema_json}"
    )
    raw_response = call_ollama(notes, system_prompt, format_json=True)
    parsed = json.loads(raw_response)
    validated = PreVisitAgenda.model_validate(parsed)
    return validated.model_dump()

def generate_post_visit_actions(doctor_notes: str) -> dict:
    """Extracts actionable checklists from doctor visit or discharge notes."""
    schema_json = json.dumps(PostVisitPlan.model_json_schema(), indent=2)
    system_prompt = (
        "You are an appointment follow-up assistant. Extract actionable instructions from clinical notes. "
        "Conform strictly to this JSON schema:\n"
        f"{schema_json}"
    )
    raw_response = call_ollama(doctor_notes, system_prompt, format_json=True)
    parsed = json.loads(raw_response)
    validated = PostVisitPlan.model_validate(parsed)
    return validated.model_dump()

def answer_grounded_qa(patient_notes: str, structured_agenda: dict, user_question: str) -> str:
    """Answers patient questions by citing specific context from their original notes."""
    system_prompt = (
        "You are a transparent assistant explaining why specific items were added to the patient's agenda. "
        "Rules:\n"
        "1. Base your answer STRICTLY on the provided Original Notes and Generated Agenda.\n"
        "2. Directly quote the relevant phrase/sentence from the notes that justified the item.\n"
        "3. Keep the tone helpful, concise, and non-diagnostic."
    )
    context_prompt = (
        f"--- ORIGINAL PATIENT NOTES ---\n{patient_notes}\n\n"
        f"--- GENERATED AGENDA ---\n{json.dumps(structured_agenda, indent=2)}\n\n"
        f"--- USER QUESTION ---\n{user_question}"
    )
    return call_ollama(context_prompt, system_prompt, format_json=False)