from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .clustering import best_group
from .face_engine import FaceEngine
from .models import FaceRecord, GroupRecord, PhotoRecord
from .storage import JsonStore

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "db.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
store = JsonStore(DB_PATH)
engine = FaceEngine()

app = FastAPI(title="Seoul Sci Graduation Photo Finder")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/upload")
async def upload_photos(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    created = []
    for file in files:
        photo_id = str(uuid.uuid4())
        ext = Path(file.filename or "upload.jpg").suffix or ".jpg"
        stored_name = f"{photo_id}{ext}"
        stored_path = UPLOAD_DIR / stored_name
        with stored_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        faces = engine.detect_and_embed(stored_path)

        def save(payload):
            payload["photos"][photo_id] = PhotoRecord(
                id=photo_id,
                filename=file.filename or stored_name,
                stored_path=f"/api/photos/{photo_id}",
                faces=[],
            ).to_dict()

            group_vectors: dict[str, list[list[float]]] = {}
            for gid, group in payload["groups"].items():
                group_vectors[gid] = [payload["faces"][fid]["embedding"] for fid in group["face_ids"]]

            for face in faces:
                face_id = str(uuid.uuid4())
                gid = best_group(face["embedding"], group_vectors, threshold=0.86)
                if gid is None:
                    gid = str(uuid.uuid4())
                    payload["groups"][gid] = GroupRecord(id=gid, face_ids=[]).to_dict()
                    group_vectors[gid] = []

                payload["groups"][gid]["face_ids"].append(face_id)
                payload["photos"][photo_id]["faces"].append(face_id)
                payload["faces"][face_id] = FaceRecord(
                    id=face_id,
                    photo_id=photo_id,
                    group_id=gid,
                    bbox=face["bbox"],
                    embedding=face["embedding"],
                ).to_dict()
                group_vectors[gid].append(face["embedding"])

        store.transaction(save)
        created.append({"photo_id": photo_id, "filename": file.filename, "detected_faces": len(faces)})
    return {"uploaded": created}


@app.get("/api/photos/{photo_id}")
def get_photo(photo_id: str):
    snapshot = store.get_snapshot()
    photo = snapshot["photos"].get(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="photo not found")

    ext = Path(photo["filename"]).suffix or ".jpg"
    path = UPLOAD_DIR / f"{photo_id}{ext}"
    if not path.exists():
        candidates = list(UPLOAD_DIR.glob(f"{photo_id}.*"))
        if not candidates:
            raise HTTPException(status_code=404, detail="file missing")
        path = candidates[0]
    return FileResponse(path)


@app.get("/api/groups")
def groups():
    snapshot = store.get_snapshot()
    data = []
    for group in snapshot["groups"].values():
        photo_ids = sorted({snapshot["faces"][fid]["photo_id"] for fid in group["face_ids"]})
        data.append(
            {
                "id": group["id"],
                "face_count": len(group["face_ids"]),
                "photo_count": len(photo_ids),
                "photo_ids": photo_ids,
            }
        )
    data.sort(key=lambda x: x["photo_count"], reverse=True)
    return {"groups": data}


@app.get("/api/groups/{group_id}/photos")
def group_photos(group_id: str):
    snapshot = store.get_snapshot()
    group = snapshot["groups"].get(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="group not found")

    photo_ids = sorted({snapshot["faces"][fid]["photo_id"] for fid in group["face_ids"]})
    photos = [snapshot["photos"][pid] for pid in photo_ids]
    return {"group_id": group_id, "photos": photos, "face_ids": group["face_ids"]}


@app.post("/api/admin/move-face")
def move_face(face_id: str, target_group_id: str):
    def mutate(payload):
        face = payload["faces"].get(face_id)
        if not face:
            raise HTTPException(status_code=404, detail="face not found")
        old_gid = face["group_id"]
        if target_group_id not in payload["groups"]:
            payload["groups"][target_group_id] = GroupRecord(id=target_group_id, face_ids=[]).to_dict()

        payload["groups"][old_gid]["face_ids"] = [fid for fid in payload["groups"][old_gid]["face_ids"] if fid != face_id]
        payload["groups"][target_group_id]["face_ids"].append(face_id)
        face["group_id"] = target_group_id

        if not payload["groups"][old_gid]["face_ids"]:
            del payload["groups"][old_gid]

    store.transaction(mutate)
    return {"ok": True}


@app.post("/api/admin/merge-groups")
def merge_groups(source_group_id: str, target_group_id: str):
    if source_group_id == target_group_id:
        raise HTTPException(status_code=400, detail="same group")

    def mutate(payload):
        source = payload["groups"].get(source_group_id)
        target = payload["groups"].get(target_group_id)
        if not source or not target:
            raise HTTPException(status_code=404, detail="group not found")

        for fid in source["face_ids"]:
            payload["faces"][fid]["group_id"] = target_group_id
        target["face_ids"].extend(source["face_ids"])
        del payload["groups"][source_group_id]

    store.transaction(mutate)
    return {"ok": True}


@app.post("/api/reset")
def reset_data():
    store.clear()
    for path in UPLOAD_DIR.glob("*"):
        if path.is_file():
            path.unlink()
    return {"ok": True}
