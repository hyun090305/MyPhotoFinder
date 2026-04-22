from pathlib import Path

from app.storage import JsonStore


def test_store_init_and_clear(tmp_path: Path):
    store = JsonStore(tmp_path / "db.json")
    snap = store.get_snapshot()
    assert snap == {"photos": {}, "faces": {}, "groups": {}}
    store.clear()
    assert store.get_snapshot() == {"photos": {}, "faces": {}, "groups": {}}
