from pydantic import BaseModel, Field


class HeartInput(BaseModel):
    age: int = Field(..., json_schema_extra={"example": 52})
    sex: int = Field(..., json_schema_extra={"example": 1})
    cp: int = Field(..., json_schema_extra={"example": 0})
    trestbps: int = Field(..., json_schema_extra={"example": 125})
    chol: int = Field(..., json_schema_extra={"example": 212})
    fbs: int = Field(..., json_schema_extra={"example": 0})
    restecg: int = Field(..., json_schema_extra={"example": 1})
    thalach: int = Field(..., json_schema_extra={"example": 168})
    exang: int = Field(..., json_schema_extra={"example": 0})
    oldpeak: float = Field(..., json_schema_extra={"example": 1.0})
    slope: int = Field(..., json_schema_extra={"example": 2})
    ca: int = Field(..., json_schema_extra={"example": 2})
    thal: int = Field(..., json_schema_extra={"example": 3})


class PredictionResponse(BaseModel):
    heart_disease: bool
    prediction_code: int
