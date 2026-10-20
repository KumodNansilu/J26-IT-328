"""MongoDB connection manager for MindSync HR users (Atlas).

Reads credentials from the MONGODB_URI environment variable (see .env.example).
Database: MindSync_DB, collection: users.
"""

import hashlib
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except Exception:
    pass

try:
    from pymongo import MongoClient
    from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError

    _HAS_PYMONGO = True
except Exception:
    _HAS_PYMONGO = False

DB_NAME = os.getenv("MONGODB_DB", "MindSync_DB")
COLLECTION = os.getenv("MONGODB_COLLECTION", "users")

# Direct shard URI (no SRV lookup) for networks whose DNS blocks
# mongodb+srv:// TXT/SRV records (e.g. campus WiFi). Password is
# injected from MONGODB_PW / the srv URI at runtime — never hardcoded.
DIRECT_URI_TEMPLATE = (
    "mongodb://admin:{pw}@ac-6ulfiwt-shard-00-00.bnqdvpr.mongodb.net:27017,"
    "ac-6ulfiwt-shard-00-01.bnqdvpr.mongodb.net:27017,"
    "ac-6ulfiwt-shard-00-02.bnqdvpr.mongodb.net:27017/"
    "?ssl=true&replicaSet=atlas-lzd5x7-shard-0&authSource=admin"
    "&retryWrites=true&w=majority"
)


def resolve_uri(explicit: str | None = None) -> str:
    """Return a working URI: explicit > MONGODB_URI, with SRV fallback to direct."""
    from urllib.parse import quote_plus, urlparse

    raw = (explicit or os.getenv("MONGODB_URI", "")).strip()
    if not raw:
        raise RuntimeError("MONGODB_URI is not set. Add it to your .env file.")
    if raw.startswith("mongodb://"):
        return raw
    # Try the srv URI first; fall back to direct shards on DNS failure.
    try:
        host = urlparse(raw).hostname or ""
        __import__("socket").getaddrinfo(f"_mongodb._tcp.{host}", 0)
        return raw
    except Exception:
        pw = os.getenv("MONGODB_PW", "")
        if not pw and "@" in raw:
            creds = raw.split("@")[0].split("//", 1)[-1]
            if ":" in creds:
                pw = creds.split(":", 1)[1]
        return DIRECT_URI_TEMPLATE.format(pw=quote_plus(pw))


def hash_password(password: str, salt: str | None = None) -> str:
    """PBKDF2-SHA256 password hash (stdlib only). Format: salt$hash."""
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a salt$hash value (constant-time)."""
    try:
        salt, h = stored.split("$", 1)
        return secrets.compare_digest(hash_password(password, salt), stored)
    except Exception:
        return False


class MongoUserManager:
    """CRUD + auth for HR users stored in MongoDB Atlas."""

    def __init__(self, uri: str | None = None, timeout_ms: int = 10_000) -> None:
        if not _HAS_PYMONGO:
            raise RuntimeError("pymongo is not installed. Run: pip install pymongo dnspython")
        self.uri = resolve_uri(uri)
        self.client = MongoClient(self.uri, serverSelectionTimeoutMS=timeout_ms)
        self.col = self.client[DB_NAME][COLLECTION]
        # Unique indexes for user_id + username
        try:
            self.col.create_index("user_id", unique=True)
            self.col.create_index("username", unique=True)
        except Exception:
            pass

    # -- health ---------------------------------------------------------
    def ping(self) -> Dict[str, Any]:
        """Check server reachability. Returns server info dict."""
        try:
            info = self.client.server_info()
            return {"ok": True, "version": info.get("version", "?")}
        except ServerSelectionTimeoutError as e:
            return {"ok": False, "error": f"Cannot reach MongoDB: {e}"}

    # -- create ---------------------------------------------------------
    def add_user(
        self,
        user_id: str,
        username: str,
        password: str,
        full_name: str = "",
        role: str = "HR",
    ) -> Dict[str, Any]:
        """Insert a new HR user. Raises ValueError on bad input/duplicates."""
        user_id = (user_id or "").strip()
        username = (username or "").strip().lower()
        if not user_id:
            raise ValueError("User ID is required.")
        if not username:
            raise ValueError("Username is required.")
        if len(password or "") < 4:
            raise ValueError("Password must be at least 4 characters.")
        doc = {
            "user_id": user_id,
            "username": username,
            "password_hash": hash_password(password),
            "full_name": full_name.strip(),
            "role": role,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            res = self.col.insert_one(doc)
        except DuplicateKeyError:
            raise ValueError("User ID or username already exists.")
        doc["_id"] = str(res.inserted_id)
        return doc

    # -- auth -----------------------------------------------------------
    def validate_login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Return user dict (without hash) if credentials match, else None."""
        if not username or not password:
            return None
        doc = self.col.find_one({"username": username.strip().lower()})
        if not doc:
            return None
        if verify_password(password, doc.get("password_hash", "")):
            doc.pop("password_hash", None)
            doc["_id"] = str(doc.get("_id"))
            return doc
        return None

    # -- read -----------------------------------------------------------
    def list_users(self, limit: int = 200) -> List[Dict[str, Any]]:
        """List users newest-first (hashes excluded)."""
        out = []
        for d in self.col.find({}, {"password_hash": 0}).sort("created_at", -1).limit(limit):
            d["_id"] = str(d.get("_id"))
            out.append(d)
        return out
