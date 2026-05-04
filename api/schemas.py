from pydantic import BaseModel, Field
from typing import List, Optional


class SearchRequest(BaseModel):
    query:   str             = Field(..., min_length=1, max_length=500)
    user_id: str             = Field(default="anonymous")
    method:  str             = Field(default="hybrid")  # hybrid | dense | bm25
    top_k:   int             = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    rank:           int
    doc_id:         str
    doc_idx:        int = Field(default=-1, description="Corpus index for click feedback")
    title:          str
    snippet:        str
    url:            str
    colbert_score:  float
    semantic_score: float
    user_score:     float
    final_score:    float
    personalized:   bool
    source:         str


class SearchResponse(BaseModel):
    query:          str
    user_id:        str
    method:         str
    results:        List[SearchResult]
    alpha:          float
    alpha_label:    str
    colbert_variance: float
    has_user_profile: bool
    user_click_count: int
    latency_ms:     float
    stage1_count:   int
    stage2_count:   int
    token_weights:  List[dict]
    stage1_all_scores: List[float] = Field(default_factory=list, description="ColBERT/MaxSim scores for all stage-1 candidates for score distribution chart")


class ClickRequest(BaseModel):
    user_id: str
    doc_id:  str
    doc_idx: int = -1
    query:   str = ""


class ClickResponse(BaseModel):
    status:      str
    user_id:     str
    click_count: int


class AnalyticsResponse(BaseModel):
    total_searches:  int
    total_clicks:    int
    avg_alpha:       float
    avg_latency_ms:  float
    top_queries:     List[dict]
