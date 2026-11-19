"""Pydantic v2 request / response models."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Channel = Literal["Web", "Mobile", "In-Store"]


class UserRecRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=64, examples=["42"])
    top_k: int = Field(default=10, ge=1, le=100)
    channel: Optional[Channel] = None
    exclude_purchased: bool = True
    category_filter: Optional[str] = Field(default=None, max_length=64, examples=["Apparel"])
    in_stock_only: bool = True


class SimilarItemsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: int = Field(ge=0)
    top_k: int = Field(default=10, ge=1, le=100)
    category_filter: Optional[str] = Field(default=None, max_length=64)
    in_stock_only: bool = True


class RecItem(BaseModel):
    item_id: int
    score: float
    category_l1: str
    category_l2: str
    brand: str
    price: float
    in_stock: bool


class Telemetry(BaseModel):
    inference_ms: float
    ann_search_ms: float
    total_ms: float


class RecResponse(BaseModel):
    strategy: Literal["personalized", "cold_start_segment", "item_similarity"]
    cold_start: bool
    model_version: str
    items: list[RecItem]
    telemetry: Telemetry


class ReloadResponse(BaseModel):
    status: str
    model_version: str
    n_items: int
    reload_ms: float
