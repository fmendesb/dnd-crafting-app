from __future__ import annotations

from typing import Any, Optional
from datetime import datetime, timedelta, timezone
import hashlib
import io
import secrets

import streamlit as st

try:
    import qrcode
except Exception:
    qrcode = None

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
        .v2-pill {display:inline-block; padding:.2rem .55rem; border-radius:999px; border:1px solid rgba(128,128,128,.25); font-size:.82rem; margin-right:.35rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pending_invite() -> Optional[str]:
    token = st.query_params.get("invite")
    if isinstance(token, list):
        token = token[0] if token else None
    if token:
        st.session_state.pending_invite = str(token)
    return st.session_state.get("pending_invite")


def _accept_pending_invite(sb: Client) -> None:
    token = _pending_invite()
    if not token or not st.session_state.get("auth_user_id"):
        return
    try:
        response = sb.rpc("accept_campaign_invitation", {"raw_token": token}).execute()
        campaign_id = getattr(response, "data", None)
        if campaign_id:
            st.session_state.selected_campaign_id = str(campaign_id)
        st.session_state.pop("pending_invite", None)
        if "invite" in st.query_params:
            del st.query_params["invite"]
        st.toast("Campaign joined successfully.", icon="🎉")
    except Exception as exc:
        st.session_state.invite_error = _friendly_error(exc)
        st.session_state.pop("pending_invite", None)
        if "invite" in st.query_params:
            del st.query_params["invite"]


def render_auth(sb: Client) -> None:
    invite = _pending_invite()
    st.title("🎲 D&D Companion App")
    st.caption("V2 sandbox · Accounts, campaigns and characters")
    if invite:
        st.info("You followed a campaign invitation. Log in or create an account and we'll take you straight to the campaign.")
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
        st.info("Development sandbox only. Production campaign data is not connected to this interface.")
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
                            st.rerun()
                        else:
                            st.success("Account created. Confirm your email, then return and log in. Your campaign invite will still work if the confirmation redirects back to this app.")
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


def _load_professions(sb: Client) -> list[dict[str, Any]]:
    return _query_data(sb.table("professions").select("id,code,name,category").eq("is_active", True).order("category").order("name").execute())


def _load_character_professions(sb: Client, character_ids: list[str]) -> list[dict[str, Any]]:
    if not character_ids:
        return []
    return _query_data(sb.table("character_professions").select("character_id,slot,profession_id,level,xp").in_("character_id", character_ids).order("slot").execute())


def _load_invitations(sb: Client, campaign_id: str) -> list[dict[str, Any]]:
    return _query_data(sb.table("campaign_invitations").select("id,kind,email,max_uses,use_count,expires_at,revoked_at,created_at").eq("campaign_id", campaign_id).order("created_at", desc=True).execute())


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
    if st.session_state.pop("invite_error", None):
        st.error("Invitation could not be accepted: " + st.session_state.get("invite_error", ""))
    st.title("My campaigns")
    st.caption("Campaigns you DM or have joined.")
    try:
        campaigns = _load_campaigns(sb)
    except Exception as exc:
        st.error(f"Could not load campaigns: {_friendly_error(exc)}")
        return
    a, b = st.columns(2)
    a.metric("Campaigns", len(campaigns))
    b.metric("Available modules", 3)
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
                    if rows:
                        st.session_state.selected_campaign_id = str(rows[0]["id"])
                    st.rerun()
                except Exception as exc:
                    st.error("Campaign could not be created. " + _friendly_error(exc))
    st.subheader("Campaigns")
    if not campaigns:
        st.info("You don't belong to a campaign yet. Create one or join through an invitation.")
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


def _qr_png(value: str) -> Optional[bytes]:
    if qrcode is None:
        return None
    img = qrcode.make(value)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_invites(sb: Client, campaign_id: str) -> None:
    st.subheader("Invite players")
    st.caption("Invitations expire after 7 days and can be revoked at any time.")
    app_base_url = str(st.secrets.get("APP_BASE_URL") or "").rstrip("/")
    tab_link, tab_email = st.tabs(["Shareable link / QR", "Email-targeted invite"])
    with tab_link:
        if st.button("Create shareable invite", use_container_width=True):
            raw = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
            try:
                sb.table("campaign_invitations").insert({"campaign_id": campaign_id, "kind": "share_link", "token_hash": token_hash, "max_uses": None, "expires_at": expires, "created_by": st.session_state.auth_user_id}).execute()
                st.session_state.latest_share_invite = raw
            except Exception as exc:
                st.error(_friendly_error(exc))
        raw = st.session_state.get("latest_share_invite")
        if raw:
            if app_base_url:
                link = f"{app_base_url}?invite={raw}"
                st.success("Invite created. Copy this link or scan the QR code.")
                st.code(link)
                png = _qr_png(link)
                if png:
                    st.image(png, width=190)
            else:
                st.success("Invite created.")
                st.code(raw)
                st.warning("Add APP_BASE_URL to Streamlit secrets to turn this token into a one-click link and QR code.")
            st.caption("For security, the raw invite token is only shown immediately after creation.")
    with tab_email:
        with st.form("email_invite_form"):
            invite_email = st.text_input("Player email")
            submitted = st.form_submit_button("Create email-targeted invite", use_container_width=True)
        if submitted:
            raw = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
            try:
                sb.table("campaign_invitations").insert({"campaign_id": campaign_id, "kind": "email", "email": invite_email.strip().lower(), "token_hash": token_hash, "max_uses": 1, "expires_at": expires, "created_by": st.session_state.auth_user_id}).execute()
                if app_base_url:
                    st.success("Invite created. Automated email delivery comes later; for now send this link manually.")
                    st.code(f"{app_base_url}?invite={raw}")
                else:
                    st.success("Invite created. Send this token manually for now.")
                    st.code(raw)
            except Exception as exc:
                st.error(_friendly_error(exc))
    st.divider()
    st.subheader("Invitation history")
    try:
        invitations = _load_invitations(sb, campaign_id)
    except Exception as exc:
        st.error(_friendly_error(exc))
        return
    if not invitations:
        st.caption("No invitations created yet.")
    for inv in invitations:
        status = "Revoked" if inv.get("revoked_at") else "Active"
        label = inv.get("email") or "Shareable link"
        cols = st.columns([4, 1])
        cols[0].markdown(f"**{label}** · {status} · uses {inv.get('use_count',0)}")
        cols[0].caption(f"Expires: {inv.get('expires_at') or 'No expiry'}")
        if not inv.get("revoked_at") and cols[1].button("Revoke", key=f"revoke_{inv['id']}", use_container_width=True):
            sb.table("campaign_invitations").update({"revoked_at": datetime.now(timezone.utc).isoformat()}).eq("id", inv["id"]).execute()
            st.rerun()


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
        professions = _load_professions(sb)
        char_prof_rows = _load_character_professions(sb, [str(c["id"]) for c in characters])
    except Exception as exc:
        st.error(f"Could not load campaign: {_friendly_error(exc)}")
        return

    is_dm = membership["role"] == "dm"
    enabled_map = _module_state_map(enabled_rows)
    module_by_id = {str(m["id"]): m for m in modules}
    enabled_codes = {module_by_id[mid]["code"] for mid, active in enabled_map.items() if active and mid in module_by_id}
    profession_by_id = {str(p["id"]): p for p in professions}
    profs_by_character: dict[str, list[dict[str, Any]]] = {}
    for row in char_prof_rows:
        prof = profession_by_id.get(str(row["profession_id"]))
        if prof:
            profs_by_character.setdefault(str(row["character_id"]), []).append({**row, **prof})

    st.caption("Campaign")
    st.title(campaign["name"])
    if campaign.get("description"):
        st.write(campaign["description"])
    st.caption(f'Your role: **{"Dungeon Master" if is_dm else "Player"}**')

    tab_labels = ["Overview", "Party", "Modules"]
    if is_dm:
        tab_labels += ["Invites", "Settings"]
    tabs = st.tabs(tab_labels)
    tab_map = dict(zip(tab_labels, tabs))

    with tab_map["Overview"]:
        a, b, c = st.columns(3)
        a.metric("Party characters", len([x for x in characters if x["status"] == "active"]))
        b.metric("Enabled modules", len(enabled_codes))
        c.metric("Your role", "DM" if is_dm else "Player")
        st.subheader("Active modules")
        if not enabled_codes:
            st.caption("No modules enabled yet.")
        for module in sorted(modules, key=lambda m: APP_MODULE_ORDER.index(m["code"]) if m["code"] in APP_MODULE_ORDER else 99):
            if module["code"] in enabled_codes:
                st.markdown(f'<div class="v2-card"><strong>✅ {module["name"]}</strong><br/><span class="v2-muted">{module.get("description") or ""}</span><br/><br/><span class="v2-pill">Module shell active</span></div>', unsafe_allow_html=True)
        if "crafting_gathering" in enabled_codes:
            st.info("Crafting & Gathering is enabled for this campaign. Its full V2 gameplay interface is the next reconstruction stage after onboarding/security tests.")

    with tab_map["Party"]:
        my_char = next((ch for ch in characters if str(ch.get("owner_membership_id")) == str(membership["id"]) and ch["status"] == "active"), None)
        st.subheader("Your character")
        if my_char:
            st.markdown(f'### 🧙 {my_char["name"]}')
            my_profs = profs_by_character.get(str(my_char["id"]), [])
            if my_profs:
                st.write("Professions: " + " · ".join(f"{p['name']} Lv.{p['level']}" for p in my_profs))
            st.caption("Module-specific inventory and progression will attach to this character identity.")
        elif membership["role"] == "player":
            with st.form("create_character_form"):
                char_name = st.text_input("Character name")
                p1 = p2 = None
                if "crafting_gathering" in enabled_codes:
                    options = [str(p["id"]) for p in professions]
                    labels = {str(p["id"]): f"{p['name']} · {p['category'].title()}" for p in professions}
                    st.caption("Choose any two different professions. They can both be Gathering, both Crafting, or one of each.")
                    p1 = st.selectbox("Profession 1", options, format_func=lambda x: labels[x])
                    p2 = st.selectbox("Profession 2", options, index=1 if len(options) > 1 else 0, format_func=lambda x: labels[x])
                submitted = st.form_submit_button("Create character", use_container_width=True)
            if submitted:
                if not char_name.strip():
                    st.error("Character name cannot be empty.")
                elif "crafting_gathering" in enabled_codes and p1 == p2:
                    st.error("Choose two different professions.")
                else:
                    try:
                        sb.rpc("create_my_character", {"target_campaign_id": campaign_id, "character_name": char_name.strip(), "profession_one": p1, "profession_two": p2}).execute()
                        st.success("Character created.")
                        st.rerun()
                    except Exception as exc:
                        st.error(_friendly_error(exc))
        else:
            st.info("DM character creation is not part of V1. Player characters are created by their own accounts after joining.")

        st.subheader("Party")
        active_chars = [x for x in characters if x["status"] == "active"]
        if not active_chars:
            st.caption("No player characters yet.")
        for char in active_chars:
            st.markdown(f'**🧙 {char["name"]}**')
            party_profs = profs_by_character.get(str(char["id"]), [])
            if party_profs:
                st.caption(" · ".join(f"{p['name']} Lv.{p['level']}" for p in party_profs))

    with tab_map["Modules"]:
        if not is_dm:
            st.info("Only the DM can enable or disable campaign modules.")
        else:
            st.subheader("Campaign modules")
            st.caption("Free accounts may enable one module. Premium accounts may enable all three. The database enforces the entitlement.")
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

    if is_dm:
        with tab_map["Invites"]:
            _render_invites(sb, campaign_id)
        with tab_map["Settings"]:
            st.subheader("Campaign settings")
            with st.form("campaign_settings_form"):
                new_name = st.text_input("Campaign name", value=campaign["name"])
                new_description = st.text_area("Description", value=campaign.get("description") or "")
                submitted = st.form_submit_button("Save changes")
            if submitted:
                try:
                    sb.table("campaigns").update({"name": new_name.strip(), "description": new_description.strip() or None}).eq("id", campaign_id).execute()
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
    _pending_invite()
    sb = _authed_client()
    if sb is None:
        st.error("Supabase is not configured. Add SUPABASE_URL and SUPABASE_ANON_KEY (or SUPABASE_PUBLISHABLE_KEY) to Streamlit secrets.")
        return
    user_id = st.session_state.get("auth_user_id")
    if not user_id:
        render_auth(sb)
        return

    _accept_pending_invite(sb)
    user_id = st.session_state.get("auth_user_id")
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


if __name__ == "__main__":
    main()
