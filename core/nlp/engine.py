import json
import os
import requests
from typing import Any, Dict, Optional, Union
from core.nlp.schemas import PreVisitAgenda, PostVisitPlan

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

def call_ollama(
    prompt: str,
    system: str,
    format_spec: Optional[Union[str, Dict[str, Any]]] = None,
    model: str = DEFAULT_MODEL
) -> str:
    """Core wrapper to communicate with the local Ollama daemon."""
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.0  # Set to 0 for deterministic schema adherence
        }
    }
    
    # Pass JSON Schema or "json" format directly to Ollama's constrained decoding engine
    if format_spec is not None:
        payload["format"] = format_spec
    
    try:
        res = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data.get("response", "").strip()
    except requests.exceptions.HTTPError as e:
        try:
            err_msg = res.json().get("error", res.text)
        except Exception:
            err_msg = str(e)
        raise RuntimeError(f"Ollama API error: {err_msg}")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Cannot connect to Ollama at {OLLAMA_ENDPOINT}. Make sure Ollama is running.")
    except Exception as e:
        raise RuntimeError(f"Ollama inference error: {str(e)}")

def generate_pre_visit_agenda(notes: str) -> dict:
    """Extracts structured appointment agenda from raw user concerns."""
    # 1. Provide a concise instructional prompt WITHOUT dumping the JSON schema text
    system_prompt = (
        "You are an organizational clinical communication assistant. "
        "Your task is strictly organizational (never diagnose or recommend treatments). "
        "Extract the patient's concerns into the required schema fields."
    )
    
    # 2. Pass the JSON schema dict directly to format_spec
    schema = PreVisitAgenda.model_json_schema()
    raw_response = call_ollama(notes, system_prompt, format_spec=schema)
    
    parsed = json.loads(raw_response)
    validated = PreVisitAgenda.model_validate(parsed)
    return validated.model_dump()

def generate_post_visit_actions(doctor_notes: str) -> dict:
    """Extracts actionable checklists from doctor visit or discharge notes."""
    system_prompt = (
        "You are an appointment follow-up assistant. Extract actionable instructions from clinical notes "
        "into the required schema fields."
    )
    
    # Pass the JSON schema dict directly to format_spec
    schema = PostVisitPlan.model_json_schema()
    raw_response = call_ollama(doctor_notes, system_prompt, format_spec=schema)
    
    parsed = json.loads(raw_response)
    validated = PostVisitPlan.model_validate(parsed)
    return validated.model_dump()

def answer_grounded_qa(patient_notes: str, structured_agenda: dict, user_question: str) -> str:
    """Answers patient questions by citing specific context from their original notes."""
    system_prompt = (
        "You are a transparent assistant explaining why specific items were added to the patient's agenda.\n"
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
    return call_ollama(context_prompt, system_prompt, format_spec=None)