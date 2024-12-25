from pydantic import BaseModel
from typing import List, Tuple


class DentalMeasurements(BaseModel):
    side_id: int
    CEJ: Tuple[int, int]
    ALC: Tuple[int, int]
    APEX: Tuple[int, int]
    CAL: float
    TRL: float
    ABLD: float
    stage: str

class Measurements(BaseModel):
    teeth_id: int
    pair_measurements: List[DentalMeasurements]
    teeth_center: Tuple[int, int]

class InferenceResponse(BaseModel):
    request_id: int
    measurements: List[Measurements]
    message: str

