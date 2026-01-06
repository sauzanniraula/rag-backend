from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class ChatRequest(BaseModel):
    """
    Schema for the Conversational RAG API.
    Handles multi-turn queries using a unique session ID.
    """
    session_id: str = Field(..., description="Unique ID for the chat session to maintain multi-turn memory.")
    query: str = Field(..., description="The user's question or request.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "user_12345",
                    "query": "Can you explain the main points of the document?"
                }
            ]
        }
    }

class BookingRequest(BaseModel):
    """
    Schema for interview booking extraction.
    Used by the LLM tool-calling logic to structure booking data.
    """
    name: str = Field(..., min_length=2, description="Full name of the interviewee.")
    email: EmailStr = Field(..., description="Valid email address for confirmation.")
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date in YYYY-MM-DD format.")
    time: str = Field(..., pattern=r"^\d{2}:\d{2}$", description="Time in HH:MM (24-hour) format.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Saujan Prakash",
                    "email": "saujan@example.com",
                    "date": "2026-01-15",
                    "time": "14:30"
                }
            ]
        }
    }