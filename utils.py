from datetime import datetime
import uuid
import os
from fastapi import UploadFile
from typing import Optional

# ----------------
# Helper utilities
# ----------------

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def save_upload(upload: UploadFile, out_dir: str, suffix: Optional[str] = None) -> str:
    ext = os.path.splitext(upload.filename or "")[1].lower() or (suffix or "")
    uid = str(uuid.uuid4())
    out_path = os.path.join(out_dir, f"{uid}{ext}")
    with open(out_path, "wb") as f:
        f.write(upload.file.read())
    upload.file.seek(0)
    return out_path