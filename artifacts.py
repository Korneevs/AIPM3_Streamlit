import base64
import functools
import os
import tempfile
import zlib

import streamlit as st


@functools.lru_cache(maxsize=None)
def artifact_path(name: str) -> str:
    """Materialize a frozen model artifact from Streamlit secrets."""
    try:
        encoded = st.secrets["artifacts"][name]
    except Exception:
        st.error(
            f"Не найден артефакт `{name}`. Добавьте его в секцию "
            "`[artifacts]` в Streamlit Secrets."
        )
        st.stop()

    try:
        raw = zlib.decompress(base64.b64decode(encoded))
    except zlib.error:
        # Backward compatible with an uncompressed artifact secret.
        raw = base64.b64decode(encoded)
    cache_dir = os.path.join(tempfile.gettempdir(), "aipm3_artifacts")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path) or os.path.getsize(path) != len(raw):
        with open(path, "wb") as stream:
            stream.write(raw)
    return path
