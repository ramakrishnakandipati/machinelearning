"""Strict Pydantic v2 request / response schemas with latency & strategy telemetry."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

STRICT = ConfigDict(extra="forbid", strict=True)

Strategy = Literal["content+graph", "meta_adapter+graph", "multimodal+contrastive_ensemble"]
ColdStartType = Literal["warm", "user_cold_zero_shot", "user_cold_few_shot", "system_cold"]


class UserProfile(BaseModel):
    model_config = STRICT
    acquisition_channel: Optional[str] = Field(None, examples=["social"])
    loyalty_tier: Optional[str] = Field(None, examples=["silver"])
    home_region: Optional[str] = Field(None, examples=["north"])
    home_store: Optional[str] = Field(None, examples=["S03"])
    device_os: Optional[str] = Field(None, examples=["ios"])
    quiz_tags: list[str] = Field(default_factory=list, max_length=10, examples=[["likes_audio"]])


class RequestContext(BaseModel):
    model_config = STRICT
    hour: Optional[int] = Field(None, ge=0, le=23)
    device: Optional[str] = Field(None, examples=["ios"])
    store_id: Optional[str] = Field(None, examples=["S03"])
    channel: Optional[Literal["web", "app", "store"]] = None


class Filters(BaseModel):
    model_config = STRICT
    l1: Optional[str] = None
    max_price_tier: Optional[int] = Field(None, ge=1, le=4)
    exclude_cold_items: bool = False
    only_cold_items: bool = False

    @model_validator(mode="after")
    def _exclusive(self):
        if self.exclude_cold_items and self.only_cold_items:
            raise ValueError("exclude_cold_items and only_cold_items are mutually exclusive")
        return self


class RecommendRequest(BaseModel):
    model_config = STRICT
    user_id: Optional[str] = Field(None, examples=["U0001"])
    profile: Optional[UserProfile] = None
    context: Optional[RequestContext] = None
    recent_item_ids: list[str] = Field(default_factory=list, max_length=50, description="Items clicked in the current session")
    k: int = Field(10, ge=1, le=100)
    filters: Optional[Filters] = None
    use_cache: bool = True

    @model_validator(mode="after")
    def _need_identity(self):
        if self.user_id is None and self.profile is None and not self.recent_item_ids:
            raise ValueError("provide at least one of user_id, profile or recent_item_ids")
        return self


class ItemOut(BaseModel):
    item_id: str
    title: str
    l1: str
    l2: str
    l3: str
    brand: str
    price: float
    price_tier: int
    score: float
    is_cold_item: bool


class Telemetry(BaseModel):
    total_ms: float
    embed_ms: float
    ann_ms: float
    cache_hit: bool
    cache_backend: str
    index: str


class RecommendResponse(BaseModel):
    user_id: Optional[str]
    strategy: Strategy
    cold_start_type: ColdStartType
    n_history_items: int
    unknown_item_ids: list[str]
    n_cold_items_in_results: int
    items: list[ItemOut]
    telemetry: Telemetry


class NewItemRequest(BaseModel):
    model_config = STRICT
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    l1: str
    l2: str
    l3: str
    brand: str
    price: float = Field(gt=0)
    price_tier: int = Field(ge=1, le=4)
    margin: float = Field(0.3, ge=0, le=1)
    color: str = "black"
    material: str = "cotton"
    visual_embedding: Optional[list[float]] = Field(None, description="64-d CLIP-like vector; omitted => modality treated as missing")
    register: bool = Field(True, description="add the item to the live ANN index so it is recommendable immediately")


class NewItemResponse(BaseModel):
    item_id: str
    registered: bool
    embedding_dim: int
    used_visual: bool
    used_graph_neighbours: bool
    nearest_items: list[ItemOut]
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    n_items: int
    n_users: int
    embedding_dim: int
    cache_backend: str
    device: str
    index_size: int
