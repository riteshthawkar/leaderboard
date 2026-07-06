"""
Pydantic models for input validation and data contracts.
"""

from pydantic import BaseModel, validator, Field, conint, confloat
from typing import Optional, Dict, Any, List
from enum import Enum
import re

from constants import (
    MAX_MODEL_NAME_LENGTH,
    MIN_MODEL_NAME_LENGTH,
    DEFAULT_LEADERBOARD_LIMIT,
    MAX_LEADERBOARD_LIMIT,
    MIN_LEADERBOARD_LIMIT,
)


class BenchmarkEnum(str, Enum):
    """Valid benchmark types."""
    MINDS_EYE = "minds_eye"
    DO_YOU_SEE_ME = "do_you_see_me"


class SubmissionRequest(BaseModel):
    """Validated submission request."""
    model_name: str = Field(..., min_length=MIN_MODEL_NAME_LENGTH, max_length=MAX_MODEL_NAME_LENGTH)
    benchmark: BenchmarkEnum
    task_name: Optional[str] = Field(None, min_length=1, max_length=255)
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Validate model name format."""
        # Allow alphanumeric, hyphens, underscores, dots, slashes, and spaces
        if not re.match(r'^[a-zA-Z0-9\-_./ ]+$', v):
            raise ValueError('Model name can only contain alphanumeric characters, hyphens, underscores, dots, slashes, and spaces')
        return v.strip()
    
    @validator('task_name')
    def validate_task_name(cls, v):
        """Validate task name format."""
        if v and not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError('Task name can only contain alphanumeric characters, hyphens, and underscores')
        return v.strip() if v else None
    
    class Config:
        use_enum_values = False
        protected_namespaces = ()


class LeaderboardRequest(BaseModel):
    """Validated leaderboard request."""
    benchmark: Optional[BenchmarkEnum] = None
    task: Optional[str] = Field(None, min_length=1, max_length=255)
    limit: conint(ge=MIN_LEADERBOARD_LIMIT, le=MAX_LEADERBOARD_LIMIT) = DEFAULT_LEADERBOARD_LIMIT
    
    @validator('task')
    def validate_task(cls, v):
        """Validate task name format."""
        if v and not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError('Invalid task name format')
        return v.strip() if v else None
    
    class Config:
        use_enum_values = False


class PredictionItem(BaseModel):
    """Single prediction in submission."""
    image_name: str = Field(..., min_length=1, max_length=1024)
    prediction: str = Field(..., min_length=1, max_length=10000)
    task_name: Optional[str] = Field(None, min_length=1, max_length=255)


class SubmissionMetadata(BaseModel):
    """Metadata for submission reproducibility."""
    model_version: Optional[str] = Field(None, max_length=255)
    model_architecture: Optional[str] = Field(None, max_length=255)
    model_parameters: Dict[str, Any] = Field(default_factory=dict)
    dataset_version: Optional[str] = Field(None, max_length=255)
    preprocessing_details: Dict[str, Any] = Field(default_factory=dict)
    code_version: Optional[str] = Field(None, max_length=255)
    framework_versions: Dict[str, str] = Field(default_factory=dict)
    notes: Optional[str] = Field(None, max_length=2000)

    class Config:
        protected_namespaces = ()


class AccuracyMetrics(BaseModel):
    """Accuracy metrics with confidence intervals."""
    accuracy: confloat(ge=0.0, le=1.0)
    confidence_interval_lower: confloat(ge=0.0, le=1.0)
    confidence_interval_upper: confloat(ge=0.0, le=1.0)
    total_samples: int = Field(..., ge=1)
    correct_samples: int = Field(..., ge=0)
    
    @validator('correct_samples')
    def validate_correct_samples(cls, v, values):
        """Ensure correct_samples <= total_samples."""
        if 'total_samples' in values and v > values['total_samples']:
            raise ValueError('correct_samples cannot exceed total_samples')
        return v
    
    class Config:
        use_enum_values = False


class SubmissionResponse(BaseModel):
    """Response model for submission."""
    success: bool
    submission_id: str
    model_name: str
    benchmark: str
    overall_accuracy: confloat(ge=0.0, le=1.0)
    total_samples: int
    correct_samples: int
    task_results: Dict[str, Dict[str, Any]]
    message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

    class Config:
        protected_namespaces = ()


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    details: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    request_id: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str  # "healthy" or "unhealthy"
    timestamp: str
    components: Dict[str, str] = Field(default_factory=dict)
    grading: Optional[str] = None  # "openai" (LLM judge) or "deterministic" (fallback)
    details: Optional[str] = None
