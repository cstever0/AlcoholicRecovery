"""Clinical research disclaimer component for Streamlit."""

import streamlit as st


def render_clinical_disclaimer(location: str = "main") -> None:
    """Render a prominent clinical research disclaimer banner.
    
    Args:
        location: 'main' to render on the main page canvas, or 'sidebar' for the sidebar.
    """
    disclaimer_text = (
        "⚠️ **RESEARCH PROOF OF CONCEPT — NOT FOR CLINICAL USE**\n\n"
        "This software is an experimental prototype created exclusively for clinical AI research. "
        "It integrates synthetic patient markers, simulated genetics, and exploratory MRI measurements. "
        "**This system is NOT a medical device and is NOT intended for clinical diagnosis, prognosis, "
        "or treatment planning.** Synthetic predictions must never be represented as clinically validated "
        "patient probabilities."
    )

    if location == "sidebar":
        st.sidebar.warning(disclaimer_text, icon="⚠️")
    else:
        st.warning(disclaimer_text, icon="⚠️")
