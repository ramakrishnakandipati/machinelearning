"""Pydantic request/response models for the cold-start recommender API."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class UserContext(BaseModel):
    """Onboarding / real-time context. All fields optional so anonymous users work."""

    region: Optional[str] = Field(None, examples=["north"])
    device: Optional[str] = Field(None, examples=["ios"])
    acquisition_channel: Optional[str] = Field(None, examples=["paid_social"])
    age_band: Optional[str] = Field(None, examples=["25-34"])
    onboarding_categories: List[str] = Field(default_factory=list, examples=[["Running Shoes", "Audio"]])
    channel: Optional[Literal["web", "app", "pos"]] = Field(None, description="Touchpoint of this request")
    season: Optional[str] = None
    hour: Optional[int] = Field(None, ge=0, le=23)


class RecommendRequest(BaseModel):
    user_id: Optional[str] = Field(None, examples=["U00042"])
    user_context: UserContext = Field(default_factory=UserContext)
    session_item_ids: List[str] = Field(default_factory=list, description="Real-time clickstream item ids")
    k: int = Field(10, ge=1, le=100)
    include_new_items_boost: bool = False


class RecommendedItem(BaseModel):
    rank: int
    item_id: str
    title: str
    brand: str
    dept: str
    subcat: str
    price: float
    score: float
    is_new_item: bool
    explain: List[str] = Field(description="Human-readable reasons this item was recommended")


class RecommendResponse(BaseModel):
    strategy: str = Field(description="cold_zero_history | semi_cold_session | warm_personalized")
    n_interactions: int
    weights: Dict[str, float]
    items: List[RecommendedItem]
    latency_ms: float
    cached: bool
    warnings: List[str] = Field(default_factory=list)


class OnboardItemRequest(BaseModel):
    item_id: str = Field(examples=["SKU_NEW_001"])
    title: str = Field(examples=["Aero Trail Runner Pro"])
    description: str = Field(examples=["Lightweight trail running shoe with grippy sole and breathable mesh."])
    brand: str
    dept: str = Field(examples=["Footwear"])
    subcat: str = Field(examples=["Running Shoes"])
    price: float = Field(gt=0)
    leaf: str = ""
    color: str = ""
    material: str = ""
    style: str = ""


class OnboardItemResponse(BaseModel):
    item_id: str
    position: int
    index_size: int
    embed_and_inject_ms: float
    message: str


class FeedbackRequest(BaseModel):
    item_id: str
    event: Literal["view", "skip", "click", "cart", "purchase"]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    user_context: UserContext = Field(default_factory=UserContext)


class FeedbackResponse(BaseModel):
    accepted: bool
    reward: float
    bandit_total_updates: int
    session_length: int


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: Optional[str]
    cuda_available: bool
    amp_trained: bool
    models_loaded: Dict[str, bool]
    faiss_backend: str
    faiss_index_size: int
    n_items: int
    n_known_users: int
    redis: str
    cache: Dict[str, int]
    embedding_backend: str
