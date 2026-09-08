import base64
import hashlib
import zlib
from pathlib import Path

import artifacts


def test_same_sized_replacement_is_not_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts.tempfile, "gettempdir", lambda: str(tmp_path))
    secrets = {"artifacts": {}}
    monkeypatch.setattr(artifacts.st, "secrets", secrets)
    for raw in [b"model-one", b"model-two"]:
        secrets["artifacts"]["test.cbm"] = base64.b64encode(zlib.compress(raw)).decode()
        artifacts.artifact_path.cache_clear()
        path = Path(artifacts.artifact_path("test.cbm"))
        assert path.read_bytes() == raw
        assert path.parent.name == hashlib.sha256(raw).hexdigest()
    artifacts.artifact_path.cache_clear()
