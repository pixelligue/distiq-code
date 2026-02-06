"""Cursor IDE proxy support.

Provides integration with Cursor IDE for cost optimization through smart model routing.
"""

from distiq_code.cursor.auth import CursorAuth
from distiq_code.cursor.client import CursorClient
from distiq_code.cursor.proxy import CursorMITMProxy

__all__ = ["CursorAuth", "CursorClient", "CursorMITMProxy"]
