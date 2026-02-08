def get_card_css() -> str:
    """Return the full CSS block for recommendation cards."""
    return """
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* Container for each card as rendered by Streamlit */
.recommend-card {
    width: 100%;
    margin: 8px 0;
}

/* Main card */
.clean-card {
    display: block;
    position: relative;
    width: 100%;
    max-width: 100%;
    min-width: 0;
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
    padding: 14px 16px 12px 16px;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\",
                 sans-serif;
    color: #0f172a;
}

/* Core vs Spec border accent */
"""
