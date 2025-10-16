from typing import Optional
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    user_prompt: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GenerateResponse(BaseModel):
    job_id: str
    status: str = "queued"


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None


class ModeRequest(BaseModel):
    mode: str
    custom_system_prompt: Optional[str] = None


class ReloadRequest(BaseModel):
    model_id: Optional[str] = None


class ShutdownRequest(BaseModel):
    reason: Optional[str] = "shutdown requested"
