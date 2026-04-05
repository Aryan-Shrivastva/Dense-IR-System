"""
SQLite-based persistence for user profiles and click history.
Uses aiosqlite for async FastAPI compatibility.
"""

import json
import os
import time
import aiosqlite
from typing import List, Dict, Optional


DB_PATH = os.getenv("DB_PATH", "data/userdata.db")


async def init_db(db_path: str = DB_PATH):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS clicks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                doc_id      TEXT NOT NULL,
                doc_idx     INTEGER DEFAULT -1,
                query       TEXT,
                timestamp   REAL NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id          TEXT PRIMARY KEY,
                interest_vector  TEXT,
                click_count      INTEGER DEFAULT 0,
                updated_at       REAL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS search_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT,
                query       TEXT,
                alpha       REAL,
                method      TEXT,
                result_count INTEGER,
                latency_ms  REAL,
                timestamp   REAL
            )
        """)
        await db.commit()


async def log_click(
    user_id: str, doc_id: str, doc_idx: int, query: str,
    db_path: str = DB_PATH
):
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO clicks (user_id, doc_id, doc_idx, query, timestamp)"
            " VALUES (?, ?, ?, ?, ?)",
            (user_id, doc_id, doc_idx, query, time.time())
        )
        await db.commit()


async def save_user_profile(
    user_id: str, interest_vector: List[float], click_count: int,
    db_path: str = DB_PATH
):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            INSERT INTO user_profiles (user_id, interest_vector, click_count, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                interest_vector=excluded.interest_vector,
                click_count=excluded.click_count,
                updated_at=excluded.updated_at
        """, (user_id, json.dumps(interest_vector), click_count, time.time()))
        await db.commit()


async def load_user_profile(
    user_id: str, db_path: str = DB_PATH
) -> Optional[Dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT interest_vector, click_count FROM user_profiles WHERE user_id=?",
            (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row and row["interest_vector"]:
                return {
                    "interest_vector": json.loads(row["interest_vector"]),
                    "click_count": row["click_count"],
                }
    return None


async def get_click_history(
    user_id: str, limit: int = 20, db_path: str = DB_PATH
) -> List[Dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT doc_id, query, timestamp FROM clicks"
            " WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def log_search(
    user_id: str, query: str, alpha: float, method: str,
    result_count: int, latency_ms: float, db_path: str = DB_PATH
):
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO search_logs"
            " (user_id, query, alpha, method, result_count, latency_ms, timestamp)"
            " VALUES (?,?,?,?,?,?,?)",
            (user_id, query, alpha, method, result_count, latency_ms, time.time())
        )
        await db.commit()


async def get_analytics(db_path: str = DB_PATH) -> Dict:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("SELECT COUNT(*) as n FROM search_logs") as c:
            total_searches = (await c.fetchone())["n"]

        async with db.execute("SELECT COUNT(*) as n FROM clicks") as c:
            total_clicks = (await c.fetchone())["n"]

        async with db.execute(
            "SELECT AVG(alpha) as avg_alpha FROM search_logs WHERE alpha IS NOT NULL"
        ) as c:
            avg_alpha = (await c.fetchone())["avg_alpha"] or 0

        async with db.execute(
            "SELECT AVG(latency_ms) as avg_lat FROM search_logs"
        ) as c:
            avg_latency = (await c.fetchone())["avg_lat"] or 0

        async with db.execute(
            "SELECT query, COUNT(*) as cnt FROM search_logs"
            " GROUP BY query ORDER BY cnt DESC LIMIT 10"
        ) as c:
            top_queries = [dict(r) for r in await c.fetchall()]

        return {
            "total_searches": total_searches,
            "total_clicks": total_clicks,
            "avg_alpha": round(avg_alpha, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "top_queries": top_queries,
        }
