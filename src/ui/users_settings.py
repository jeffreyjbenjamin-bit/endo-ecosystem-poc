"""Settings → Users: view, add, edit, deactivate and delete app users."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from src.common.users import (
    USER_TYPES,
    UserError,
    add_user,
    delete_user,
    ensure_users,
    import_mailing_list,
    list_users,
    update_user,
)

_EDITOR_KEY = "users_editor"
_COLS = ["id", "name", "email", "user_type", "active", "source", "created_ts"]


def _reset_editor() -> None:
    st.session_state.pop(_EDITOR_KEY, None)


def _save_changes(df: pd.DataFrame, original: pd.DataFrame) -> None:
    errors, n_upd, n_del = [], 0, 0
    orig = original.set_index("id")
    for _, row in df.iterrows():
        uid = int(row["id"])
        if bool(row.get("delete")):
            delete_user(uid)
            n_del += 1
            continue
        before = orig.loc[uid]
        if any(row[c] != before[c] for c in ("name", "email", "user_type", "active")):
            try:
                update_user(
                    uid,
                    row["name"],
                    row["email"],
                    row["user_type"],
                    bool(row["active"]),
                )
                n_upd += 1
            except UserError as exc:
                errors.append(f"{row['email']}: {exc}")
    st.session_state["users_msg"] = (n_upd, n_del, errors)
    _reset_editor()


def render_users() -> None:
    seeded = ensure_users()
    if seeded:
        st.toast(f"Users created from the mailing list: {seeded} Admin(s).")
    users = list_users()

    st.markdown("#### 👥 Users")
    counts = Counter(u["user_type"] for u in users if u["active"])
    st.caption(
        f"{len(users)} user{'s' if len(users) != 1 else ''} · "
        + " · ".join(
            f"{counts.get(t, 0)} {t}{'s' if counts.get(t, 0) != 1 else ''}"
            for t in USER_TYPES
        )
        + ". User types: Admin, Physician, Payer, Patient, Researcher. Sign-in and "
        "permissions by type are not switched on yet."
    )

    msg = st.session_state.pop("users_msg", None)
    if msg:
        n_upd, n_del, errors = msg
        if n_upd or n_del:
            st.success(f"Saved: {n_upd} updated, {n_del} deleted.")
        for e in errors:
            st.error(e)

    # ── Add a user / import ─────────────────────────────────────
    with st.form("users_add_form", clear_on_submit=True, border=True):
        st.markdown("**Add a user**")
        c1, c2, c3, c4 = st.columns([1.3, 1.6, 1, 0.7], vertical_alignment="bottom")
        name = c1.text_input("Name", placeholder="Jane Smith")
        email = c2.text_input("Email", placeholder="jane@example.com")
        utype = c3.selectbox(
            "User type", USER_TYPES, index=USER_TYPES.index("Physician")
        )
        submitted = c4.form_submit_button("➕ Add", use_container_width=True)
    if submitted:
        try:
            add_user(name, email, utype)
            st.session_state["users_flash"] = f"Added {name.strip()} as {utype}."
            _reset_editor()
            st.rerun()  # redraw the counts and the list with the new user
        except UserError as exc:
            st.error(str(exc))
    flash = st.session_state.pop("users_flash", None)
    if flash:
        st.success(flash)

    with st.popover("📥 Import from mailing list"):
        st.caption(
            "Adds everyone on the email mailing list who is not a user yet. "
            "Existing users are not changed."
        )
        imp_type = st.selectbox("Add them as", USER_TYPES, key="users_import_type")
        if st.button("Import", key="users_import_btn", type="primary"):
            n = import_mailing_list(imp_type)
            st.session_state["users_flash"] = (
                f"Imported {n} user(s) as {imp_type}."
                if n
                else "Everyone on the mailing list is already a user."
            )
            _reset_editor()
            st.rerun()

    # ── Edit the list ───────────────────────────────────────────
    if not users:
        st.info("No users yet. Add one above.")
        return
    type_filter = st.pills(
        "Show",
        USER_TYPES,
        selection_mode="multi",
        key="users_type_filter",
        on_change=_reset_editor,
        help="Filter by user type (none selected = all)",
    )
    original = pd.DataFrame(users)[_COLS]
    shown = (
        original[original["user_type"].isin(type_filter)] if type_filter else original
    )
    shown = shown.assign(delete=False).reset_index(drop=True)
    edited = st.data_editor(
        shown,
        key=_EDITOR_KEY,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_order=[
            "name",
            "email",
            "user_type",
            "active",
            "source",
            "created_ts",
            "delete",
        ],
        column_config={
            "name": st.column_config.TextColumn("Name", required=True),
            "email": st.column_config.TextColumn("Email", required=True),
            "user_type": st.column_config.SelectboxColumn(
                "User type", options=USER_TYPES, required=True
            ),
            "active": st.column_config.CheckboxColumn(
                "Active", help="Inactive users are kept but marked as not current"
            ),
            "source": st.column_config.TextColumn("Added from", disabled=True),
            "created_ts": st.column_config.DatetimeColumn(
                "Added", disabled=True, format="MMM D, YYYY"
            ),
            "delete": st.column_config.CheckboxColumn(
                "Delete", help="Tick, then Save changes, to delete the user"
            ),
        },
    )
    changed = not edited.drop(columns=["delete"]).equals(
        shown.drop(columns=["delete"])
    ) or bool(edited["delete"].any())
    n_del = int(edited["delete"].sum())
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    c1.button(
        "💾 Save changes",
        key="users_save_btn",
        type="primary",
        disabled=not changed,
        on_click=_save_changes,
        args=(edited, original),
        use_container_width=True,
    )
    if n_del:
        c2.warning(f"{n_del} user(s) will be deleted when you save.")
    elif changed:
        c2.caption("● Unsaved changes")
