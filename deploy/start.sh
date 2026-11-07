#!/bin/sh
# Container entrypoint: daily-send scheduler in the background, Streamlit in the foreground.
python scheduler.py &
exec streamlit run src/ui/rag_panel.py --server.port=8501 --server.address=0.0.0.0
