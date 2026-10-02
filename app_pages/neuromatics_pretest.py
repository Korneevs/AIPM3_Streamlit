"""Neuromatics pretest uses its own scorer and isolated page state."""
from app_pages.latest_pretest import main


if __name__ in {"__main__", "__page__"}:
    main(material_kind="neuromatics")
