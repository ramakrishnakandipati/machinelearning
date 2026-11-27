from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Raw (possibly misspelled / colloquial) shopper query")
    user_id: str | None = Field(None, description="Known user for personalisation (e.g. U00012); omit for anonymous")
    store_id: str | None = Field(None, description="Store for BOPIS / regional stock (S1..S5); overrides zip_code")
    zip_code: str | None = Field(None, description="Resolves to the matching store (e.g. 98101 -> S1)")
    fulfillment: Literal["any", "ship", "bopis"] = "any"
    brand_allow: list[str] = Field(default_factory=list)
    brand_deny: list[str] = Field(default_factory=list)
    min_price: float = 0.0
    max_price: float = Field(0.0, description="Hard price cap; also parsed from text such as 'under $100'")
    ranker: str | None = Field(None, description="xgb_pointwise | lgbm_pointwise | xgb_pairwise | lgbm_lambdamart | "
                                                 "cat_yetirank | xgb_listwise | torch_listnet | dcnv2 | ensemble")
    fusion: Literal["alpha", "rrf", "dense", "sparse"] | None = None
    alpha: float | None = Field(None, ge=0, le=1)
    top_k: int = Field(10, ge=1, le=50)
    margin_weight: float | None = Field(None, ge=0, le=1, description="Weight of expected profit in the final blend")
    apply_rules: bool = True
    margin_opt: bool = True
    diversify: bool = True

    model_config = {"json_schema_extra": {"examples": [
        {"query": "waterproof trail kicks", "user_id": "U00012", "zip_code": "98101", "top_k": 10},
        {"query": "gortex snekers under $100", "fulfillment": "bopis", "store_id": "S2", "ranker": "dcnv2"},
        {"query": "warm winter puffer", "user_id": "U00100", "margin_weight": 0.5, "fusion": "rrf"},
    ]}}


class ScoreBreakdown(BaseModel):
    final: float
    ranker: float
    relevance_norm: float
    dense_cos: float
    bm25: float
    p_click: float
    p_add_to_cart: float
    p_purchase_given_cart: float
    p_return: float
    expected_revenue: float
    expected_profit: float


class ResultItem(BaseModel):
    rank: int
    sku_id: str
    title: str
    brand: str
    category: str
    subcategory: str
    color: str
    price: float
    gross_margin_pct: float
    return_rate: float
    is_cold_start: bool
    stock_online: int
    stock_store: int
    fulfillment: str
    scores: ScoreBreakdown
    flags: list[str]


class Funnel(BaseModel):
    retrieved: int
    after_hard_filters: int
    returned: int
    dropped_reasons: dict[str, int]
    cold_start_in_results: int


class SearchResponse(BaseModel):
    query: str
    corrected_query: str
    spell_edits: list[list[str]]
    understanding: dict
    ranker: str
    fusion: str
    store_id: str | None
    funnel: Funnel
    latency_ms: dict[str, float]
    results: list[ResultItem]
