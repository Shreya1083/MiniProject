"""SQLite-backed storage for users and detection search history.

Tables
------
users        : id, email, username, password_hash, created_at
search_history : id, user_id, filename, label, deepfake_score,
                 architecture, timestamp, fft_available
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "app.db"


def _ensure_db_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _conn():
    _ensure_db_dir()
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def init_db() -> None:
    """Create tables if they don't exist."""
    with _conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                email         TEXT    NOT NULL UNIQUE,
                username      TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        INTEGER NOT NULL,
                filename       TEXT    NOT NULL,
                label          TEXT    NOT NULL,
                deepfake_score REAL    NOT NULL,
                architecture   TEXT    NOT NULL,
                preprocessing  TEXT    NOT NULL,
                score_note     TEXT    NOT NULL,
                timestamp      TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )


# --------------------------------------------------------------------------- #
# Password helpers
# --------------------------------------------------------------------------- #
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _check_password(password: str, stored_hash: str) -> bool:
    return _hash_password(password) == stored_hash


# --------------------------------------------------------------------------- #
# User management
# --------------------------------------------------------------------------- #
def create_user(email: str, username: str, password: str) -> dict | None:
    """Create a new user. Returns the user dict or None if email/username taken."""
    init_db()
    pw_hash = _hash_password(password)
    try:
        with _conn() as con:
            con.execute(
                "INSERT INTO users (email, username, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (email.lower().strip(), username.strip(), pw_hash,
                 datetime.now().isoformat()),
            )
        return get_user_by_username(username)
    except sqlite3.IntegrityError:
        return None  # duplicate email or username


def get_user_by_username(username: str) -> dict | None:
    """Return user dict by username or None."""
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT id, email, username, created_at FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    return dict(row) if row else None


def authenticate(username: str, password: str) -> dict | None:
    """Return user dict if credentials are valid, else None."""
    init_db()
    with _conn() as con:
        row = con.execute(
            "SELECT id, email, username, password_hash, created_at "
            "FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row and _check_password(password, row["password_hash"]):
        return {
            "id": row["id"],
            "email": row["email"],
            "username": row["username"],
            "created_at": row["created_at"],
        }
    return None


# --------------------------------------------------------------------------- #
# Search history
# --------------------------------------------------------------------------- #
def save_search(
    user_id: int,
    filename: str,
    label: str,
    deepfake_score: float,
    architecture: str,
    preprocessing: str,
    score_note: str,
) -> int:
    """Persist one detection result. Returns the new row id."""
    init_db()
    with _conn() as con:
        cur = con.execute(
            """
            INSERT INTO search_history
                (user_id, filename, label, deepfake_score,
                 architecture, preprocessing, score_note, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, filename, label, deepfake_score,
                architecture, preprocessing, score_note,
                datetime.now().isoformat(),
            ),
        )
        return cur.lastrowid


def get_user_history(user_id: int) -> list[dict]:
    """Return all searches for a user, newest first."""
    init_db()
    with _conn() as con:
        rows = con.execute(
            """
            SELECT id, filename, label, deepfake_score,
                   architecture, preprocessing, score_note, timestamp
            FROM search_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_history_entry(entry_id: int, user_id: int) -> bool:
    """Delete a single history entry (only if it belongs to user_id)."""
    init_db()
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM search_history WHERE id = ? AND user_id = ?",
            (entry_id, user_id),
        )
    return cur.rowcount > 0


def clear_user_history(user_id: int) -> int:
    """Delete all history for a user. Returns number of rows deleted."""
    init_db()
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM search_history WHERE user_id = ?", (user_id,)
        )
    return cur.rowcount
