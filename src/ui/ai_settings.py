"""Settings → AI models & usage: which model each feature uses, tokens and cost."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.services import ai_service as ai


def render_ai_usage() -> None:
    st.markdown("#### 🤖 AI models & usage")
    status = {f: ai.provider_status(f) for f in ai.FEATURES}
    st.caption(
        " · ".join(f"{ai.FEATURES[f]['label']}: **{s}**" for f, s in status.items())
        + " · Everything else: **Azure OpenAI GPT-5.1**. Models are set with the "
        "ANTHROPIC_MODEL and NEWSLETTER_AI_PROVIDER server settings."
    )
    if not ai.is_configured():
        st.info(
            "Claude is not set up yet: add ANTHROPIC_API_KEY to the server settings "
            "(.env locally, App Service application settings on Azure). Until then the "
            "newsletter features use Azure OpenAI."
        )
    days = (
        st.segmented_control(
            "Period",
            [7, 30, 90],
            default=30,
            format_func=lambda d: f"Last {d} days",
            key="ai_usage_days",
            label_visibility="collapsed",
        )
        or 30
    )
    rows = ai.usage_summary(days)
    if not rows:
        st.caption("No Claude calls recorded in this period.")
        return
    df = pd.DataFrame(rows)
    df["feature"] = df["feature"].map(lambda f: ai.FEATURES.get(f, {}).get("label", f))
    df["model"] = df["model"].map(ai.model_name)
    total = float(df["cost_usd"].sum())
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "feature": "Feature",
            "model": "Model",
            "calls": st.column_config.NumberColumn("Calls"),
            "errors": st.column_config.NumberColumn("Failed"),
            "input_tokens": st.column_config.NumberColumn(
                "Input tokens", format="localized"
            ),
            "output_tokens": st.column_config.NumberColumn(
                "Output tokens", format="localized"
            ),
            "cache_write_tokens": st.column_config.NumberColumn(
                "Cache write", format="localized"
            ),
            "cache_read_tokens": st.column_config.NumberColumn(
                "Cache read", format="localized"
            ),
            "cost_usd": st.column_config.NumberColumn("Est. cost", format="$%.4f"),
        },
    )
    st.caption(
        f"Estimated total: **${total:,.2f}** over the last {days} days, from list "
        "prices. Your Anthropic Console invoice is the source of truth."
    )
