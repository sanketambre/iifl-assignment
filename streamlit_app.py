#!/usr/bin/env python3
"""Entry point for the support agent UI.

    streamlit run streamlit_app.py

This is the composition root: the only file that knows about both the agent and
the interface. The backend has no idea a UI exists, so the batch runner and the
tests use it unchanged.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Streamlit Cloud supplies the key through st.secrets rather than a .env file.
# This has to happen before src.backend.config is imported, because config reads
# the environment once at import time.
try:
    for name in ("GEMINI_API_KEY", "GEMINI_MODEL"):
        if name in st.secrets:
            os.environ.setdefault(name, str(st.secrets[name]))
except Exception:
    pass  # no secrets file locally, which is fine: .env covers it

from src.frontend.ui import render

render()
