"""Pydantic v2 request/response schemas.

`GuestFeatures` lists the most important, human-meaningful inputs with validation; any other model
feature (see outputs/models/<scenario>/schema.json) may be passed by name as an extra field.
Omitted fields are treated as missing (count-like fields default to 0), exactly as in training.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Channel = Literal["OTA", "Direct Web", "Mobile App", "Corporate"]


class GuestFeatures(BaseModel):
    model_config = ConfigDict(extra="allow", json_schema_extra={"example": {
        "guest_id": "G-1001", "loyalty_tier": "Gold", "is_loyalty_member": 1, "cobrand_card": 1,
        "n_stays": 6, "stays_12m": 2, "recency_days": 140, "inter_stay_mean_days": 180, "avg_adr": 310.0,
        "ota_share": 0.35, "ota_share_12m": 0.5, "n_ota_12m": 1, "direct_share": 0.55, "cancel_rate": 0.1,
        "nps_mean": 8.0, "n_complaints": 0, "email_open_rate_3m": 0.45, "email_click_rate_3m": 0.1,
        "sessions_30d": 3, "checkout_reach_90d": 1,
        "channel": "OTA", "rate_plan": "Refundable", "room_type": "Deluxe", "property_type": "Coastal Vacation",
        "lead_time": 45, "los": 4, "adults": 2, "children": 1, "adr": 340.0, "days_until_checkin": 21, "checkin_month": 7}})

    guest_id: Optional[str] = None
    # ---- profile
    loyalty_tier: Optional[Literal["Non-Member", "Silver", "Gold", "Platinum"]] = None
    is_loyalty_member: Optional[int] = Field(None, ge=0, le=1)
    cobrand_card: Optional[int] = Field(None, ge=0, le=1)
    corporate_account: Optional[int] = Field(None, ge=0, le=1)
    app_installed: Optional[int] = Field(None, ge=0, le=1)
    age_band: Optional[Literal["18-29", "30-44", "45-59", "60+"]] = None
    market_type: Optional[Literal["drive", "fly"]] = None
    avg_distance_km: Optional[float] = Field(None, ge=0)
    points_gap_to_free_night: Optional[float] = Field(None, ge=0)
    account_age_days: Optional[float] = Field(None, ge=0)
    # ---- RFM / inter-stay
    n_stays: Optional[float] = Field(None, ge=0)
    stays_12m: Optional[float] = Field(None, ge=0)
    stays_24m: Optional[float] = Field(None, ge=0)
    recency_days: Optional[float] = Field(None, ge=0)
    frequency_per_year: Optional[float] = Field(None, ge=0)
    lifetime_spend: Optional[float] = Field(None, ge=0)
    avg_adr: Optional[float] = Field(None, ge=0)
    avg_los: Optional[float] = Field(None, ge=0)
    inter_stay_mean_days: Optional[float] = Field(None, ge=0)
    inter_stay_std_days: Optional[float] = Field(None, ge=0)
    inter_stay_decay_ratio: Optional[float] = Field(None, ge=0)
    # ---- channel / price sensitivity / affinity
    ota_share: Optional[float] = Field(None, ge=0, le=1)
    ota_share_12m: Optional[float] = Field(None, ge=0, le=1)
    n_ota_12m: Optional[float] = Field(None, ge=0)
    direct_share: Optional[float] = Field(None, ge=0, le=1)
    app_share: Optional[float] = Field(None, ge=0, le=1)
    nonref_share: Optional[float] = Field(None, ge=0, le=1)
    discount_dependency: Optional[float] = Field(None, ge=0, le=1)
    adr_vs_season_mean: Optional[float] = Field(None, ge=0)
    n_distinct_properties: Optional[float] = Field(None, ge=0)
    property_entropy: Optional[float] = Field(None, ge=0)
    top_property_share: Optional[float] = Field(None, ge=0, le=1)
    share_at_last_property: Optional[float] = Field(None, ge=0, le=1)
    last_property_type: Optional[str] = None
    # ---- service quality, ancillary, engagement
    cancel_rate: Optional[float] = Field(None, ge=0, le=1)
    n_noshow: Optional[float] = Field(None, ge=0)
    nps_mean: Optional[float] = Field(None, ge=0, le=10)
    nps_last: Optional[float] = Field(None, ge=0, le=10)
    n_complaints: Optional[float] = Field(None, ge=0)
    days_since_complaint: Optional[float] = Field(None, ge=0)
    ancillary_per_stay: Optional[float] = Field(None, ge=0)
    upgrade_rate: Optional[float] = Field(None, ge=0, le=1)
    spa_rate: Optional[float] = Field(None, ge=0, le=1)
    email_open_rate_3m: Optional[float] = Field(None, ge=0, le=1)
    email_click_rate_3m: Optional[float] = Field(None, ge=0, le=1)
    email_open_trend: Optional[float] = Field(None, ge=-1, le=1)
    sessions_30d: Optional[float] = Field(None, ge=0)
    sessions_90d: Optional[float] = Field(None, ge=0)
    checkout_reach_90d: Optional[float] = Field(None, ge=0)
    days_since_last_session: Optional[float] = Field(None, ge=0)
    # ---- current / upcoming booking context (scenarios 3, 4, 8) and open web session (scenario 2)
    channel: Optional[Channel] = None
    rate_plan: Optional[Literal["Refundable", "Non-Refundable"]] = None
    room_type: Optional[Literal["Standard", "Deluxe", "Club", "Suite"]] = None
    property_type: Optional[Literal["Luxury Resort", "Urban Business Hotel", "Coastal Vacation"]] = None
    lead_time: Optional[float] = Field(None, ge=0)
    los: Optional[float] = Field(None, ge=1)
    adults: Optional[int] = Field(None, ge=1, le=10)
    children: Optional[int] = Field(None, ge=0, le=10)
    adr: Optional[float] = Field(None, ge=0)
    days_until_checkin: Optional[float] = Field(None, ge=0)
    checkin_month: Optional[int] = Field(None, ge=1, le=12)
    n_searches: Optional[float] = Field(None, ge=0)
    pages_viewed: Optional[float] = Field(None, ge=0)
    reached_checkout: Optional[int] = Field(None, ge=0, le=1)
    device: Optional[Literal["mobile", "desktop", "tablet", "app"]] = None
    days_since_session: Optional[float] = Field(None, ge=0)


class PredictGuestRequest(BaseModel):
    guest: GuestFeatures
    explain: bool = Field(False, description="include top SHAP drivers per scenario")


class PredictBatchRequest(BaseModel):
    guests: list[GuestFeatures] = Field(min_length=1, max_length=20000)


class Driver(BaseModel):
    feature: str
    shap: float
    value: Optional[str] = None
    direction: str


class ScenarioScore(BaseModel):
    propensity: float = Field(ge=0, le=1, description="calibrated probability")
    decile: int = Field(ge=1, le=10, description="10 = top-propensity decile")
    tier: str
    base_rate: float
    model: str
    uplift_segment: Optional[str] = None
    uplift: Optional[float] = None
    next_best_action: str
    top_drivers: Optional[list[Driver]] = None


class PredictGuestResponse(BaseModel):
    guest_id: Optional[str] = None
    scores: dict[str, ScenarioScore]
    latency_ms: float


class PredictBatchResponse(BaseModel):
    n: int
    results: list[PredictGuestResponse]
    latency_ms: float
    guests_per_second: float


class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    gpu_device: Optional[str] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    models_loaded: dict[str, str]
    uplift_model_loaded: bool
