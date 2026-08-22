from __future__ import annotations

from typing import Any, Optional

import streamlit as st

try:
    from supabase import Client, create_client
except Exception:
    Client = Any  # type: ignore
    create_client = None

st.set_page_config(page_title="D&D Companion App · V2", page_icon="🎲", layout="wide")

APP_MODULE_ORDER = ["crafting_gathering", "bastions", "relations"]


def _base_client() -> Optional[Client]:
    if create_client is None:
        return None
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_ANON_KEY") or st.secrets.get("SUPABASE_PUBLISHABLE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def _clear_auth_state() -> None:
    for key in ("access_token", "refresh_token", "auth_user_id", "auth_email", "selected_campaign_id"):
        st.session_state.pop(key, None)


def _authed_client() -> Optional[Client]:
    sb = _base_client()
    if sb is None:
        return None
    access = st.session_state.get("access_token")
    refresh = st.session_state.get("refresh_token")
    if access and refresh:
        try:
            response = sb.auth.set_session(access, refresh)
            if getattr(response, "session", None):
                st.session_state.access_token = response.session.access_token
                st.session_state.refresh_token = response.session.refresh_token
        except Exception:
            _clear_auth_state()
            return _base_client()
    return sb


def _remember_session(response: Any) -> bool:
    session = getattr(response, "session", None)
    user = getattr(response, "user", None)
    if not session or not user:
        return False
    st.session_state.access_token = session.access_token
    st.session_state.refresh_token = session.refresh_token
    st.session_state.auth_user_id = str(user.id)
    st.session_state.auth_email = getattr(user, "email", "") or ""
    return True


def _query_data(response: Any) -> list[dict[str, Any]]:
    return list(getattr(response, "data", None) or [])


def _friendly_error(exc: Exception) -> str:
    message = str(exc).replace("AuthApiError", "").strip("() :")
    return message or "Something went wrong."


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
        div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: .8rem 1rem;}
        .v2-card {border: 1px solid rgba(128,128,128,.22); border-radius: 18px; padding: 1.1rem 1.2rem; margin-bottom: .8rem;}
        .v2-muted {opacity: .72;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_auth(sb: Client) -> None:
    st.title("🎲 D&D Companion App")
    st.caption("V2 sandbox · Accounts, campaigns and characters")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown(
            """
            ### Your campaign systems, one table away
            This V2 sandbox is rebuilding the app around real accounts, campaigns,
            permissions and reusable modules.

            **Available modules**
            - ⚒️ Crafting & Gathering
            - 🏰 Bastions
            - 🤝 Relations
            """
        )
        st.info("This is a development sandbox. Production campaign data is not connected to this interface.")
    with right:
        sign_in_tab, sign_up_tab = st.tabs(["Log in", "Create account"])
        with sign_in_tab:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Log in", use_container_width=True)
            if submitted:
                try:
                    response = sb.auth.sign_in_with_password({"email": email.strip(), "password": password})
                    if _remember_session(response):
                        st.success("Welcome back.")
                        st.rerun()
                    else:
                        st.error("Login succeeded without a usable session.")
                except Exception as exc:
                    st.error(_friendly_error(exc))
        with sign_up_tab:
            with st.form("signup_form"):
                display_name = st.text_input("Display name")
                email = st.text_input("Email", key="signup_email")
                password = st.text_input("Password", type="password", help="Use at least 8 characters.", key="signup_password")
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                submitted = st.form_submit_button("Create account", use_container_width=True)
            if submitted:
                if len(password) < 8:
                    st.error("Password must contain at least 8 characters.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif not display_name.strip():
                    st.error("Choose a display name.")
                else:
                    try:
                        response = sb.auth.sign_up({"email": email.strip(), "password": password, "options": {"data": {"display_name": display_name.strip()}}})
                        if _remember_session(response):
                            st.success("Account created.")
                            st.rerun()
                        else:
                            st.success("Account created. If email confirmation is enabled, check your inbox, confirm the address, then log in.")
                    except Exception as exc:
                        st.error(_friendly_error(exc))


def _load_profile(sb: Client, user_id: str) -> dict[str, Any]:
    rows = _query_data(sb.table("profiles").select("user_id,display_name,avatar_path,created_at").eq("user_id", user_id).limit(1).execute())
    return rows[0] if rows else {"user_id": user_id, "display_name": None}


def _load_campaigns(sb: Client) -> list[dict[str, Any]]:
    return _query_data(sb.table("campaigns").select("id,name,description,status,owner_user_id,created_at,updated_at").order("updated_at", desc=True).execute())


def _load_membership(sb: Client, campaign_id: str, user_id: str) -> Optional[dict[str, Any]]:
    rows = _query_data(sb.table("campaign_members").select("id,campaign_id,user_id,role,status,joined_at").eq("campaign_id", campaign_id).eq("user_id", user_id).limit(1).execute())
    return rows[0] if rows else None


def _load_modules(sb: Client, campaign_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_modules = _query_data(sb.table("app_modules").select("id,code,name,description").eq("is_active", True).execute())
    enabled = _query_data(sb.table("campaign_modules").select("campaign_id,module_id,enabled,settings").eq("campaign_id", campaign_id).execute())
    return all_modules, enabled


def _load_characters(sb: Client, campaign_id: str) -> list[dict[str, Any]]:
    return _query_data(sb.table("characters").select("id,campaign_id,owner_membership_id,name,avatar_path,status,created_at").eq("campaign_id", campaign_id).order("created_at").execute())


def render_sidebar(sb: Client, profile: dict[str, Any]) -> None:
    with st.sidebar:
        st.markdown("### 🎲 D&D Companion")
        display = profile.get("display_name") or st.session_state.get("auth_email") or "Account"
        st.write(f"**{display}**")
        st.caption(st.session_state.get("auth_email", ""))
        st.divider()
        if st.button("🏠 My campaigns", use_container_width=True):
            st.session_state.pop("selected_campaign_id", None)
            st.rerun()
        if st.button("Log out", use_container_width=True):
            try:
                sb.auth.sign_out()
            except Exception:
                pass
            _clear_auth_state()
            st.rerun()
        st.divider()
        st.caption("V2 development sandbox")


def render_campaign_list(sb: Client, user_id: str) -> None:
    st.title("My campaigns")
    st.caption("Campaigns you DM or have joined.")
    try:
        campaigns = _load_campaigns(sb)
    except Exception as exc:
        st.error(f"Could not load campaigns: {_friendly_error(exc)}")
        return
    top_left, top_right = st.columns([1, 1])
    top_left.metric("Campaigns", len(campaigns))
    top_right.metric("V2 modules", 3)
    st.subheader("Create a campaign")
    with st.expander("＋ New campaign", expanded=not campaigns):
        with st.form("create_campaign_form"):
            name = st.text_input("Campaign name", placeholder="Call of the Netherdeep")
            description = st.text_area("Description", placeholder="Optional short description for your table.", height=90)
            submitted = st.form_submit_button("Create campaign", use_container_width=True)
        if submitted:
            if not name.strip():
                st.error("Campaign name cannot be empty.")
            else:
                try:
                    result = sb.table("campaigns").insert({"name": name.strip(), "description": description.strip() or None, "owner_user_id": user_id}).execute()
                    rows = _query_data(result)
                    st.success("Campaign created.")
                    if rows:
                        st.session_state.selected_campaign_id = str(rows[0]["id"])
                    st.rerun()
                except Exception as exc:
                    st.error("Campaign could not be created. " + _friendly_error(exc))
    st.subheader("Campaigns")
    if not campaigns:
        st.info("You don't belong to a campaign yet. Create your first one above.")
        return
    for campaign in campaigns:
        icon = "📦" if campaign["status"] == "archived" else "🗺️"
        st.markdown(f'<div class="v2-card"><strong>{icon} {campaign["name"]}</strong><br/><span class="v2-muted">{campaign.get("description") or "No description yet."}</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Open", key=f'open_{campaign["id"]}', use_container_width=True, disabled=campaign["status"] == "archived"):
                st.session_state.selected_campaign_id = str(campaign["id"])
                st.rerun()
        with c2:
            role = "DM" if str(campaign["owner_user_id"]) == user_id else "Player"
            st.caption(f'{role} · {campaign["status"].title()}')


def _module_state_map(enabled_rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {str(row["module_id"]): bool(row["enabled"]) for row in enabled_rows}


def render_campaign(sb: Client, user_id: str, campaign_id: str) -> None:
    try:
        campaigns = [c for c in _load_campaigns(sb) if str(c["id"]) == str(campaign_id)]
        if not campaigns:
            st.warning("This campaign is not available to your account.")
            st.session_state.pop("selected_campaign_id", None)
            return
        campaign = campaigns[0]
        membership = _load_membership(sb, campaign_id, user_id)
        if membership is None:
            st.warning("No active membership was found.")
            return
        characters = _load_characters(sb, campaign_id)
        modules, enabled_rows = _load_modules(sb, campaign_id)
    except Exception as exc:
        st.error(f"Could not load campaign: {_friendly_error(exc)}")
        return
    is_dm = membership["role"] == "dm"
    st.caption("Campaign")
    st.title(campaign["name"])
    if campaign.get("description"):
        st.write(campaign["description"])
    st.caption(f'Your role: **{"Dungeon Master" if is_dm else "Player"}**')
    overview_tab, party_tab, modules_tab, settings_tab = st.tabs(["Overview", "Party", "Modules", "Settings"])
    with overview_tab:
        enabled_map = _module_state_map(enabled_rows)
        a, b, c = st.columns(3)
        a.metric("Party characters", len([x for x in characters if x["status"] == "active"]))
        b.metric("Enabled modules", sum(1 for v in enabled_map.values() if v))
        c.metric("Your role", "DM" if is_dm else "Player")
        st.subheader("Modules")
        for module in sorted(modules, key=lambda m: APP_MODULE_ORDER.index(m["code"]) if m["code"] in APP_MODULE_ORDER else 99):
            active = enabled_map.get(str(module["id"]), False)
            st.markdown(f'**{module["name"]}** · {"✅ Enabled" if active else "○ Not enabled"}')
            st.caption(module.get("description") or "")
    with party_tab:
        my_char = next((ch for ch in characters if str(ch.get("owner_membership_id")) == str(membership["id"]) and ch["status"] == "active"), None)
        st.subheader("Your character")
        if my_char:
            st.markdown(f'### 🧙 {my_char["name"]}')
            st.caption("Character shell created. Module-specific progression comes next.")
        elif membership["role"] == "player":
            with st.form("create_character_form"):
                char_name = st.text_input("Character name")
                submitted = st.form_submit_button("Create character", use_container_width=True)
            if submitted:
                if not char_name.strip():
                    st.error("Character name cannot be empty.")
                else:
                    try:
                        sb.table("characters").insert({"campaign_id": campaign_id, "owner_membership_id": membership["id"], "name": char_name.strip()}).execute()
                        st.success("Character created.")
                        st.rerun()
                    except Exception as exc:
                        st.error(_friendly_error(exc))
        else:
            st.info("DM character creation is not part of V1. Join as a player in another campaign to create a player character.")
        st.subheader("Party")
        active_chars = [x for x in characters if x["status"] == "active"]
        if not active_chars:
            st.caption("No player characters yet.")
        for char in active_chars:
            st.markdown(f'**🧙 {char["name"]}**')
    with modules_tab:
        if not is_dm:
            st.info("Only the DM can enable or disable campaign modules.")
        else:
            st.subheader("Campaign modules")
            st.caption("Free accounts may enable one module. Premium accounts may enable all three. The database enforces the entitlement.")
            enabled_map = _module_state_map(enabled_rows)
            for module in sorted(modules, key=lambda m: APP_MODULE_ORDER.index(m["code"]) if m["code"] in APP_MODULE_ORDER else 99):
                module_id = str(module["id"])
                current = enabled_map.get(module_id, False)
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f'**{module["name"]}**')
                    st.caption(module.get("description") or "")
                with c2:
                    if current:
                        if st.button("Disable", key=f"disable_module_{module_id}", use_container_width=True):
                            try:
                                sb.table("campaign_modules").update({"enabled": False}).eq("campaign_id", campaign_id).eq("module_id", module_id).execute()
                                st.rerun()
                            except Exception as exc:
                                st.error(_friendly_error(exc))
                    else:
                        if st.button("Enable", key=f"enable_module_{module_id}", use_container_width=True):
                            try:
                                existing = [row for row in enabled_rows if str(row["module_id"]) == module_id]
                                if existing:
                                    sb.table("campaign_modules").update({"enabled": True}).eq("campaign_id", campaign_id).eq("module_id", module_id).execute()
                                else:
                                    sb.table("campaign_modules").insert({"campaign_id": campaign_id, "module_id": module_id, "enabled": True, "settings": {}}).execute()
                                st.rerun()
                            except Exception as exc:
                                st.error("Module could not be enabled. " + _friendly_error(exc))
    with settings_tab:
        if not is_dm:
            st.info("Campaign settings are DM-only.")
        else:
            st.subheader("Campaign settings")
            with st.form("campaign_settings_form"):
                new_name = st.text_input("Campaign name", value=campaign["name"])
                new_description = st.text_area("Description", value=campaign.get("description") or "")
                submitted = st.form_submit_button("Save changes")
            if submitted:
                try:
                    sb.table("campaigns").update({"name": new_name.strip(), "description": new_description.strip() or None}).eq("id", campaign_id).execute()
                    st.success("Campaign updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(_friendly_error(exc))
            st.divider()
            if campaign["status"] == "active" and st.button("Archive campaign", type="secondary"):
                try:
                    sb.table("campaigns").update({"status": "archived"}).eq("id", campaign_id).execute()
                    st.session_state.pop("selected_campaign_id", None)
                    st.rerun()
                except Exception as exc:
                    st.error(_friendly_error(exc))


def main() -> None:
    _inject_css()
    sb = _authed_client()
    if sb is None:
        st.error("Supabase is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY (or SUPABASE_PUBLISHABLE_KEY) to Streamlit secrets.")
        return
    user_id = st.session_state.get("auth_user_id")
    if not user_id:
        render_auth(sb)
        return
    try:
        profile = _load_profile(sb, user_id)
    except Exception:
        profile = {"user_id": user_id, "display_name": None}
    render_sidebar(sb, profile)
    campaign_id = st.session_state.get("selected_campaign_id")
    if campaign_id:
        render_campaign(sb, user_id, campaign_id)
    else:
        render_campaign_list(sb, user_id)
