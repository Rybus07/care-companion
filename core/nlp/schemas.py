from pydantic import BaseModel, Field
from typing import List, Optional

class SymptomItem(BaseModel):
    symptom: str
    duration_or_onset: str

class PreVisitAgenda(BaseModel):
    chief_complaint: str = Field(description="Primary reason for visit in one concise sentence")
    symptom_timeline: List[SymptomItem]
    current_medications: List[str]
    questions_for_doctor: List[str] = Field(description="Top 3-4 prioritized questions")

class MedicationInstruction(BaseModel):
    name: str
    dosage: str
    instructions: str

class PostVisitPlan(BaseModel):
    medications: List[MedicationInstruction]
    tests_to_schedule: List[str]
    lifestyle_modifications: List[str]
    red_flag_warnings: List[str]
    follow_up_timeframe: str