import streamlit as st
import yaml
import streamlit_authenticator as stauth
from pathlib import Path

st.set_page_config(layout="wide")

# Load auth.yaml
auth_path = Path("auth.yaml")
with auth_path.open() as f:
    cfg = yaml.safe_load(f)

authenticator = stauth.Authenticate(
    credentials=cfg["credentials"],
    cookie_name=cfg["cookie"]["name"],
    cookie_key=cfg["cookie"]["key"],
    cookie_expiry_days=cfg["cookie"]["expiry_days"],
)

st.write("Rendering login block now…")

login_result = authenticator.login(key="test_login")

st.write("DEBUG:", login_result)

if login_result:
    name, auth_status, username = login_result
    st.success(f"Logged in as {name}")
    authenticator.logout("Logout", "sidebar")
