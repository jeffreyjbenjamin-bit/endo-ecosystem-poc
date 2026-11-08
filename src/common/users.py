"""App users: the start of user accounts for Evidence Gap.

Users live in users.sqlite in the app state directory. Each has a name, a unique
email, a user type and an active flag. The first time the table is created it is
seeded from the email mailing list, with every recipient as an Admin. Settings
(⚙️) has the screens to add, edit, deactivate and delete users.

This is a directory of users and their types only; it does not yet control
sign-in or permissions.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.paths import state_path

USERS_DB = state_path("users.sqlite")
MAILING_LIST_DB = state_path("mailing_list.sqlite")
USER_TYPES = ["Admin", "Physician", "Payer", "Patient", "Researcher"]
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE COLLATE NOCASE,
    user_type   TEXT NOT NULL DEFAULT 'Researcher',
    active      INTEGER NOT NULL DEFAULT 1,
    source      TEXT NOT NULL DEFAULT 'manual',
    created_ts  TEXT NOT NULL,
    updated_ts  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class UserError(ValueError):
    """A user could not be saved (bad email, duplicate, unknown type)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or USERS_DB
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def mailing_list_recipients(mailing_db: Optional[str] = None) -> List[Dict[str, str]]:
    path = mailing_db or MAILING_LIST_DB
    if not Path(path).exists():
        return []
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("SELECT name, email FROM recipients ORDER BY id").fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        conn.close()
    return [{"name": n or e, "email": e} for n, e in rows if e]


def _validate(name: str, email: str, user_type: str) -> tuple:
    name, email = (name or "").strip(), (email or "").strip().lower()
    if not name:
        raise UserError("Name is required.")
    if not _EMAIL_RE.match(email):
        raise UserError(f"'{email}' is not a valid email address.")
    if user_type not in USER_TYPES:
        raise UserError(f"User type must be one of: {', '.join(USER_TYPES)}.")
    return name, email


def ensure_users(
    db_path: Optional[str] = None, mailing_db: Optional[str] = None
) -> int:
    """Create the users table; on first creation seed it from the mailing list
    (everyone as Admin). Returns how many users were seeded (0 after the first time).
    """
    conn = _connect(db_path)
    try:
        if conn.execute("SELECT value FROM users_meta WHERE key='seeded'").fetchone():
            return 0
        n = 0
        ts = _now()
        for r in mailing_list_recipients(mailing_db):
            cur = conn.execute(
                "INSERT OR IGNORE INTO users "
                "(name, email, user_type, active, source, created_ts, updated_ts) "
                "VALUES (?,?,?,?,?,?,?)",
                (r["name"], r["email"].lower(), "Admin", 1, "mailing list", ts, ts),
            )
            n += cur.rowcount
        conn.execute(
            "INSERT OR REPLACE INTO users_meta (key, value) VALUES ('seeded', ?)", (ts,)
        )
        conn.commit()
        return n
    finally:
        conn.close()


def list_users(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY active DESC, user_type, name COLLATE NOCASE"
        ).fetchall()
        return [dict(r) | {"active": bool(r["active"])} for r in rows]
    finally:
        conn.close()


def add_user(
    name: str,
    email: str,
    user_type: str,
    active: bool = True,
    source: str = "manual",
    db_path: Optional[str] = None,
) -> int:
    name, email = _validate(name, email, user_type)
    conn = _connect(db_path)
    try:
        ts = _now()
        cur = conn.execute(
            "INSERT INTO users (name, email, user_type, active, source, created_ts, "
            "updated_ts) VALUES (?,?,?,?,?,?,?)",
            (name, email, user_type, int(active), source, ts, ts),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        raise UserError(f"A user with email {email} already exists.") from None
    finally:
        conn.close()


def update_user(
    user_id: int,
    name: str,
    email: str,
    user_type: str,
    active: bool,
    db_path: Optional[str] = None,
) -> None:
    name, email = _validate(name, email, user_type)
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE users SET name=?, email=?, user_type=?, active=?, updated_ts=? "
            "WHERE id=?",
            (name, email, user_type, int(bool(active)), _now(), user_id),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise UserError(f"A user with email {email} already exists.") from None
    finally:
        conn.close()


def delete_user(user_id: int, db_path: Optional[str] = None) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def import_mailing_list(
    user_type: str, db_path: Optional[str] = None, mailing_db: Optional[str] = None
) -> int:
    """Add mailing-list recipients who are not users yet. Returns how many."""
    if user_type not in USER_TYPES:
        raise UserError(f"User type must be one of: {', '.join(USER_TYPES)}.")
    conn = _connect(db_path)
    try:
        ts, n = _now(), 0
        for r in mailing_list_recipients(mailing_db):
            cur = conn.execute(
                "INSERT OR IGNORE INTO users "
                "(name, email, user_type, active, source, created_ts, updated_ts) "
                "VALUES (?,?,?,?,?,?,?)",
                (r["name"], r["email"].lower(), user_type, 1, "mailing list", ts, ts),
            )
            n += cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()
