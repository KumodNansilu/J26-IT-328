"""MongoDB Atlas database connection."""

import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "mindsync_enterprise",
)

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is missing from the .env file.")

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=5000,
)

db = client[MONGODB_DATABASE]


def test_connection() -> bool:
    """Check whether MongoDB is reachable."""
    client.admin.command("ping")
    return True