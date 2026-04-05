"""
POST /api/click — logs a click and updates the user interest model live.
"""

from fastapi import APIRouter
from api.schemas import ClickRequest, ClickResponse
from api.state import get_state
from src.database.user_store import log_click, save_user_profile

router = APIRouter()


@router.post("/click", response_model=ClickResponse)
async def record_click(req: ClickRequest):
    state      = get_state()
    user_model = state["user_model"]
    embeddings = state["embeddings"]

    # Update in-memory EMA model
    idx = req.doc_idx
    if 0 <= idx < len(embeddings):
        user_model.update(req.user_id, embeddings[idx])

    # Persist click to DB
    await log_click(
        user_id=req.user_id,
        doc_id=req.doc_id,
        doc_idx=req.doc_idx,
        query=req.query,
    )

    # Persist updated interest vector
    profile = user_model.get_profile(req.user_id)
    if profile is not None:
        await save_user_profile(
            user_id=req.user_id,
            interest_vector=profile.tolist(),
            click_count=user_model.get_click_count(req.user_id),
        )

    return ClickResponse(
        status="ok",
        user_id=req.user_id,
        click_count=user_model.get_click_count(req.user_id),
    )
