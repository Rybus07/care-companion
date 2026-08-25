import json
import requests
from core.nlp.schemas import PreVisitAgenda  # <-- Using schemas.py here

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

def generate_pre_visit_agenda(user_notes: str) -> dict:
    # 1. Automatically convert the schema into a JSON contract for the prompt
    schema_definition = json.dumps(PreVisitAgenda.model_json_schema(), indent=2)
    
    system_prompt = f"""
    You are a clinical communication assistant. Your role is purely organizational.
    Convert the patient's unstructured notes into structured JSON conforming strictly to this schema:
    {schema_definition}
    """
    
    payload = {
        "model": "llama3.2:3b",
        "prompt": user_notes,
        "system": system_prompt,
        "format": "json",
        "stream": False
    }
    
    response = requests.post(OLLAMA_ENDPOINT, json=payload).json()
    raw_dict = json.loads(response["response"])
    
    # 2. Validate against schema (ensures no missing keys or type errors)
    validated_data = PreVisitAgenda.model_validate(raw_dict)
    
    # 3. Return clean dictionary for Django
    return validated_data.model_dump()