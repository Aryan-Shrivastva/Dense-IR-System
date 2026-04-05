"""
GET /api/analytics — system-wide analytics for the dashboard charts.
GET /api/user/{user_id}/profile — per-user profile info.
GET /api/user/{user_id}/history — click history.
"""

from fastapi import APIRouter
from api.state import get_state
from src.database.user_store import get_analytics, get_click_history, load_user_profile

router = APIRouter()


@router.get("/analytics")
async def analytics():
    return await get_analytics()


@router.get("/user/{user_id}/profile")
async def user_profile(user_id: str):
    state      = get_state()
    user_model = state["user_model"]

    if not user_model.has_profile(user_id):
        profile = await load_user_profile(user_id)
        if profile:
            user_model.load_from_db(
                user_id,
                profile["interest_vector"],
                profile.get("click_count"),
            )

    return {
        "user_id":     user_id,
        "has_profile": user_model.has_profile(user_id),
        "click_count": user_model.get_click_count(user_id),
    }


@router.get("/user/{user_id}/history")
async def user_history(user_id: str, limit: int = 20):
    history = await get_click_history(user_id, limit)
    return {"user_id": user_id, "history": history}
