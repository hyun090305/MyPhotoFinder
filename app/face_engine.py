from __future__ import annotations

import hashlib
from pathlib import Path

try:
    import face_recognition  # type: ignore
except Exception:  # pragma: no cover
    face_recognition = None


class FaceEngine:
    """Detects and embeds faces. Falls back to deterministic mock in limited envs."""

    def __init__(self) -> None:
        self.real_mode = face_recognition is not None

    def detect_and_embed(self, image_path: Path) -> list[dict]:
        if self.real_mode:
            img = face_recognition.load_image_file(str(image_path))
            locations = face_recognition.face_locations(img)
            encodings = face_recognition.face_encodings(img, locations)
            faces = []
            for (top, right, bottom, left), encoding in zip(locations, encodings):
                faces.append(
                    {
                        "bbox": [left, top, right - left, bottom - top],
                        "embedding": [float(x) for x in encoding],
                    }
                )
            return faces

        # fallback: treat entire image as one pseudo-face with stable embedding
        blob = image_path.read_bytes()
        digest = hashlib.sha256(blob).digest()
        embedding = []
        for i in range(0, 128):
            b = digest[i % len(digest)]
            embedding.append((b / 255.0) * 2 - 1)
        return [{"bbox": [0, 0, 0, 0], "embedding": embedding}]
