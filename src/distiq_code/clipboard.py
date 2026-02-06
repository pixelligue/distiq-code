"""Clipboard image detection, saving, and background watching."""

from datetime import datetime
from pathlib import Path


def get_clipboard_hash() -> int | None:
    """Return a fast hash of the current clipboard image, or None."""
    try:
        from PIL import ImageGrab, Image

        img = ImageGrab.grabclipboard()
        if img is None or not isinstance(img, Image.Image):
            return None
        # Hash first 2KB of pixel data for speed
        return hash(img.tobytes()[:2048])
    except Exception:
        return None


def check_clipboard_image(project_dir: str | None = None) -> str | None:
    """Check clipboard for an image, save it, return the absolute path (or None)."""
    try:
        from PIL import ImageGrab, Image
    except ImportError:
        return None

    try:
        img = ImageGrab.grabclipboard()
    except Exception:
        return None

    if img is None or not isinstance(img, Image.Image):
        return None

    base = Path(project_dir) if project_dir else Path.cwd()
    img_dir = base / ".distiq-images"
    img_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"paste_{timestamp}.png"
    filepath = img_dir / filename

    img.save(str(filepath), "PNG")

    return str(filepath)
