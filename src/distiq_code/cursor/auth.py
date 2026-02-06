"""Cursor authentication token extraction from SQLite database.

Reads Bearer token and machine ID from Cursor's local state database.
"""

import os
import platform
import sqlite3
from pathlib import Path
from typing import Dict, Optional

from loguru import logger
from pydantic import BaseModel


class CursorTokens(BaseModel):
    """Cursor authentication tokens."""

    access_token: str
    refresh_token: Optional[str] = None
    email: Optional[str] = None
    machine_id: str
    membership_type: Optional[str] = None


class CursorAuth:
    """Extract Cursor authentication tokens from SQLite database."""

    @staticmethod
    def get_db_path() -> Path:
        """
        Get platform-specific path to Cursor state database.

        Returns:
            Path to state.vscdb SQLite file

        Raises:
            FileNotFoundError: If database file doesn't exist
        """
        system = platform.system()

        if system == "Windows":
            base = Path(os.environ.get("APPDATA", ""))
            db_path = base / "Cursor" / "User" / "globalStorage" / "state.vscdb"
        elif system == "Darwin":  # macOS
            base = Path.home() / "Library" / "Application Support"
            db_path = base / "Cursor" / "User" / "globalStorage" / "state.vscdb"
        elif system == "Linux":
            base = Path.home() / ".config"
            db_path = base / "Cursor" / "User" / "globalStorage" / "state.vscdb"
        else:
            raise OSError(f"Unsupported platform: {system}")

        if not db_path.exists():
            raise FileNotFoundError(
                f"Cursor database not found at {db_path}. "
                "Make sure Cursor IDE is installed and you've logged in."
            )

        return db_path

    @staticmethod
    def extract_tokens() -> CursorTokens:
        """
        Extract authentication tokens from Cursor database.

        Returns:
            CursorTokens with access token, machine ID, and optional metadata

        Raises:
            FileNotFoundError: If Cursor database doesn't exist
            ValueError: If required tokens are missing
            sqlite3.Error: If database read fails
        """
        db_path = CursorAuth.get_db_path()
        logger.info(f"Reading Cursor tokens from {db_path}")

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Keys to extract
            keys = {
                "cursorAuth/accessToken": "access_token",
                "cursorAuth/refreshToken": "refresh_token",
                "cursorAuth/cachedEmail": "email",
                "storage.serviceMachineId": "machine_id",
                "cursorAuth/stripeMembershipType": "membership_type",
            }

            tokens = {}

            for db_key, field_name in keys.items():
                cursor.execute("SELECT value FROM ItemTable WHERE key = ?", (db_key,))
                row = cursor.fetchone()

                if row:
                    value = row[0]
                    # Handle both string and bytes
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    tokens[field_name] = value
                else:
                    logger.debug(f"Key not found: {db_key}")

            conn.close()

            # Validate required fields
            if "access_token" not in tokens:
                raise ValueError(
                    "Access token not found in Cursor database. "
                    "Please log in to Cursor IDE first."
                )

            if "machine_id" not in tokens:
                raise ValueError(
                    "Machine ID not found in Cursor database. "
                    "This is required for authentication."
                )

            cursor_tokens = CursorTokens(**tokens)

            logger.success(
                f"Extracted Cursor tokens (email: {cursor_tokens.email or 'unknown'}, "
                f"tier: {cursor_tokens.membership_type or 'unknown'})"
            )

            return cursor_tokens

        except sqlite3.Error as e:
            logger.error(f"Failed to read Cursor database: {e}")
            raise
