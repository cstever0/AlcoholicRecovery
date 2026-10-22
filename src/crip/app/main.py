"""Cardiomyopathy Recovery Intelligence Platform (CRIP) - Main Streamlit Entrypoint."""

import sys
from pathlib import Path

# Ensure src directory is in sys.path when run via `streamlit run`
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st
from crip.app.components.disclaimer import render_clinical_disclaimer

APP_DIR = Path(__file__).resolve().parent

# Clean page configuration definitions
PAGE_CONFIGS = [
    {
        "file": APP_DIR / "pages" / "1_Patient_Profile.py",
        "title": "Patient Profile",
        "icon": "👤",
        "default": True,
    },
    {
        "file": APP_DIR / "pages" / "2_MRI_Analysis.py",
        "title": "MRI Analysis",
        "icon": "🫀",
        "default": False,
    },
    {
        "file": APP_DIR / "pages" / "3_Recovery_Estimate.py",
        "title": "Recovery Estimate",
        "icon": "📈",
        "default": False,
    },
    {
        "file": APP_DIR / "pages" / "4_Evaluation.py",
        "title": "Evaluation",
        "icon": "🔬",
        "default": False,
    },
]

pages = [
    st.Page(
        str(cfg["file"]),
        title=cfg["title"],
        icon=cfg["icon"],
        default=cfg["default"],
    )
    for cfg in PAGE_CONFIGS
]

navigation = st.navigation(
    {
        "Clinical Exploration": pages
    }
)


def main() -> None:
    navigation.run()


if __name__ == "__main__":
    main()
