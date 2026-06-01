from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["admin", "user"]


@dataclass(frozen=True)
class UserRecord:
    username: str
    role: Role


class UserStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(
                {
                    "users": [
                        _make_user("admin", "admin123", "admin"),
                        _make_user("user", "user123", "user"),
                    ]
                }
            )

    def list_users(self) -> list[UserRecord]:
        raw = self._read()
        return [
            UserRecord(username=user["username"], role=user["role"])
            for user in raw["users"]
        ]

    def authenticate(self, username: str, password: str) -> UserRecord | None:
        raw = self._read()
        for user in raw["users"]:
            if user["username"] == username and _verify_password(password, user["password_hash"]):
                return UserRecord(username=user["username"], role=user["role"])
        return None

    def get_user(self, username: str) -> UserRecord | None:
        raw = self._read()
        for user in raw["users"]:
            if user["username"] == username:
                return UserRecord(username=user["username"], role=user["role"])
        return None

    def add_user(self, username: str, password: str, role: Role) -> UserRecord:
        username = username.strip()
        if not username:
            raise ValueError("Username is required.")
        if role not in {"admin", "user"}:
            raise ValueError("Role must be admin or user.")
        if len(password) < 6:
            raise ValueError("Password must contain at least 6 characters.")

        raw = self._read()
        if any(user["username"] == username for user in raw["users"]):
            raise ValueError("Username already exists.")

        raw["users"].append(_make_user(username, password, role))
        self._write(raw)
        return UserRecord(username=username, role=role)

    def delete_user(self, username: str) -> None:
        raw = self._read()
        users = raw["users"]
        target = next((user for user in users if user["username"] == username), None)
        if target is None:
            raise ValueError("User not found.")
        if target["role"] == "admin" and sum(1 for user in users if user["role"] == "admin") <= 1:
            raise ValueError("Cannot delete the last admin user.")

        raw["users"] = [user for user in users if user["username"] != username]
        self._write(raw)

    def _read(self) -> dict:
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write(self, payload: dict) -> None:
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, UserRecord] = {}

    def create(self, user: UserRecord) -> str:
        token = secrets.token_urlsafe(32)
        self._sessions[token] = user
        return token

    def get(self, token: str | None) -> UserRecord | None:
        if not token:
            return None
        return self._sessions.get(token)

    def delete(self, token: str | None) -> None:
        if token:
            self._sessions.pop(token, None)


def _make_user(username: str, password: str, role: Role) -> dict:
    return {
        "username": username,
        "role": role,
        "password_hash": _hash_password(password),
    }


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"pbkdf2_sha256${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    algorithm, encoded_salt, encoded_digest = stored_hash.split("$", 2)
    if algorithm != "pbkdf2_sha256":
        return False
    salt = base64.b64decode(encoded_salt)
    expected = base64.b64decode(encoded_digest)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return hmac.compare_digest(actual, expected)
