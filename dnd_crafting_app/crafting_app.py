import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional, Tuple
import copy
import random
import math
import re
import datetime

import streamlit as st

# --- Optional dependency (Supabase) ---
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = None  # type: ignore

st.set_page_config(page_title="D&D Crafting Simulator", layout="wide")
DATA_DIR = Path(__file__).parent / "data"

# -----------------------------
# Supabase
# -----------------------------
def supabase_client() -> Optional["Client"]:
    if create_client is None:
        return None
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

SB = supabase_client()

# -----------------------------
# Load JSON (cache bust on file change)
# -----------------------------
@st.cache_data(show_spinner=False)
def _load_json(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_fresh(rel: str):
    p = DATA_DIR / rel
    return _load_json(str(p), p.stat().st_mtime)

PLAYERS_DEFAULT: List[Dict[str, Any]] = load_fresh("players.json")
GATHERING_ITEMS: List[Dict[str, Any]] = load_fresh("gathering_items.json")
RECIPES: List[Dict[str, Any]] = load_fresh("recipes.json")
TIER_UNLOCKS: List[Dict[str, Any]] = load_fresh("tier_unlocks.json")
XP_TABLE_RAW: Dict[str, int] = load_fresh("xp_table.json")
XP_TABLE = {int(k): int(v) for k, v in XP_TABLE_RAW.items()}

# -----------------------------
# Constants
# -----------------------------
SELL_RATE = 0.5

VENDOR_MAX_LINES = 3
VENDOR_QTY_MAX = 4
VENDOR_TIER_WEIGHTS = {"T": 0.90, "T+1": 0.20, "T+2": 0.05}
VENDOR_QTY_WEIGHTS = {1: 0.50, 2: 0.35, 3: 0.20, 4: 0.05}
VENDOR_LINECOUNT_WEIGHTS = {0: 0.10, 1: 0.35, 2: 0.35, 3: 0.20}
FALLBACK_TIER_GP = {1: 2, 2: 5, 3: 11, 4: 20, 5: 45, 6: 90, 7: 180}

# -----------------------------
# Helpers
# -----------------------------
TIER_SUFFIX_RE = re.compile(r"\s*\(T(\d)\)\s*$", re.IGNORECASE)

def canon_prof(s: str) -> str:
    return (s or "").strip()

# -----------------------------
# Profession inference from players.json
# -----------------------------
# Some players.json files only store "skills" (e.g., {"Mining": {...}}) without explicit
# gathering_profession / crafting_professions. We infer them so the UI works.
GATHER_PROFS = {canon_prof(x.get("profession", "")) for x in GATHERING_ITEMS if canon_prof(x.get("profession", ""))}
CRAFT_PROFS = {canon_prof(x.get("profession", "")) for x in RECIPES if canon_prof(x.get("profession", ""))}

# Backwards-compatible aliases (older code paths)
gathering_professions = GATHER_PROFS
crafting_professions = CRAFT_PROFS


def normalize_players(players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in players or []:
        p = copy.deepcopy(p)
        skills = p.get("skills") or {}
        skill_names = {canon_prof(k) for k in skills.keys()}

        # Infer gathering profession (single)
        gp = p.get("gathering_profession")
        if not canon_prof(str(gp or "")):
            inferred = [s for s in skill_names if s in GATHER_PROFS]
            p["gathering_profession"] = inferred[0] if inferred else ""

        # Infer crafting professions (list)
        cps = p.get("crafting_professions")
        if not isinstance(cps, list) or not cps:
            inferred = sorted([s for s in skill_names if s in CRAFT_PROFS])
            p["crafting_professions"] = inferred
        else:
            p["crafting_professions"] = [canon_prof(x) for x in cps if canon_prof(x)]

        # Ensure known_recipes exists
        if "known_recipes" not in p or not isinstance(p.get("known_recipes"), list):
            p["known_recipes"] = []

        out.append(p)
    return out
def canon_name(s: str) -> str:
    return " ".join((s or "").strip().split())

def base_name(s: str) -> str:
    s = canon_name(s)
    m = TIER_SUFFIX_RE.search(s)
    if m:
        return canon_name(s[:m.start()])
    return s

def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, str) and not x.strip():
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def safe_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)

def normalize_weights(weights: Dict[Any, float]) -> Dict[Any, float]:
    s = sum(weights.values()) or 1.0
    return {k: v / s for k, v in weights.items()}

def weighted_choice(weights: Dict[Any, float]) -> Any:
    weights = normalize_weights(weights)
    r = random.random()
    acc = 0.0
    for k, w in weights.items():
        acc += w
        if r <= acc:
            return k
    return list(weights.keys())[-1]

def xp_to_next(level: int) -> int:
    return int(XP_TABLE.get(int(level), 0))

def max_tier_for_level(level: int) -> int:
    tier = 1
    for row in TIER_UNLOCKS:
        if int(level) >= int(row.get("unlocks_at_level", 1)):
            tier = max(tier, int(row.get("tier", 1)))
    return tier

def next_unlock_level_for_tier(target_tier: int) -> Optional[int]:
    for row in TIER_UNLOCKS:
        if int(row.get("tier", 0)) == int(target_tier):
            lvl = int(row.get("unlocks_at_level", 0))
            return lvl if lvl > 0 else None
    return None

def dc_for_target_tier(unlocked_tier: int, target_tier: int) -> Optional[int]:
    if target_tier <= unlocked_tier:
        return 10
    if target_tier == unlocked_tier + 1:
        return 15
    if target_tier == unlocked_tier + 2:
        return 20
    return None

def tier_color(unlocked: int, target: int) -> str:
    if target <= unlocked - 1:
        return "#16a34a"
    if target == unlocked:
        return "#111827"
    if target == unlocked + 1:
        return "#b45309"
    return "#b91c1c"

def tier_badge_html(unlocked: int, target: int) -> str:
    col = tier_color(unlocked, target)
    return f'<span style="color:{col};font-weight:700;">T{target}</span>'

# -----------------------------
# Pricing
# -----------------------------
PRICE_KEYS_VENDOR = ["vendor_price_gp", "vendorPriceGp", "vendor_gp", "vendorGp", "vendor_price", "vendorPrice", "vendorCost", "vendor_cost"]
PRICE_KEYS_BASE = ["base_price_gp", "basePriceGp", "base_price", "basePrice", "base_gp", "baseGp", "baseCost", "base_cost", "price", "cost", "gp"]

def get_price(item: Dict[str, Any], keys: List[str]) -> float:
    for k in keys:
        if k in item:
            v = safe_float(item.get(k), 0.0)
            if v > 0:
                return v
    return 0.0

def effective_unit_price(item: Dict[str, Any]) -> float:
    v = get_price(item, PRICE_KEYS_VENDOR)
    if v > 0:
        return v
    b = get_price(item, PRICE_KEYS_BASE)
    if b > 0:
        return b
    t = safe_int(item.get("tier", 1), 1)
    return float(FALLBACK_TIER_GP.get(t, 0))

# -----------------------------
# Build indices
# -----------------------------
ITEM_TIER: Dict[str, int] = {}
ITEM_PROF: Dict[str, str] = {}
ITEM_DESC: Dict[str, str] = {}
ITEM_USE: Dict[str, str] = {}
ITEM_VENDOR: Dict[str, float] = {}

GATHERING_ITEMS_N = []
for it in GATHERING_ITEMS:
    it = dict(it)
    it["name"] = canon_name(it.get("name", ""))
    it["profession"] = canon_prof(it.get("profession", ""))
    GATHERING_ITEMS_N.append(it)
GATHERING_ITEMS = GATHERING_ITEMS_N

RECIPES_N = []
for r in RECIPES:
    r = dict(r)
    r["name"] = canon_name(r.get("name", ""))
    r["profession"] = canon_prof(r.get("profession", ""))
    comps = []
    for c in r.get("components", []) or []:
        c = dict(c)
        c["name"] = canon_name(c.get("name", ""))
        c["qty"] = int(c.get("qty", 1))
        comps.append(c)
    r["components"] = comps
    RECIPES_N.append(r)
RECIPES = RECIPES_N

for it in GATHERING_ITEMS:
    nm = it.get("name", "")
    if not nm:
        continue
    ITEM_TIER[nm] = int(it.get("tier", 1))
    ITEM_PROF[nm] = canon_prof(it.get("profession", "") or "")
    ITEM_DESC[nm] = it.get("description", "") or ""
    ITEM_USE[nm] = it.get("use", "") or ""
    ITEM_VENDOR[nm] = effective_unit_price(it)

for r in RECIPES:
    nm = r.get("name", "")
    if not nm:
        continue
    ITEM_TIER.setdefault(nm, int(r.get("tier", 1)))
    ITEM_DESC.setdefault(nm, r.get("description", "") or "")
    ITEM_USE.setdefault(nm, r.get("use", "") or "")
    ITEM_VENDOR.setdefault(nm, effective_unit_price(r))

CRAFT_PROF_TO_MATS: Dict[str, set] = defaultdict(set)
CRAFT_PROF_TO_MATS_BASE: Dict[str, set] = defaultdict(set)

for r in RECIPES:
    craft_prof = canon_prof(r.get("profession", "") or "")
    for c in r.get("components", []):
        if c.get("name"):
            CRAFT_PROF_TO_MATS[craft_prof].add(c["name"])
            CRAFT_PROF_TO_MATS_BASE[craft_prof].add(base_name(c["name"]))

gathering_by_prof = defaultdict(list)
for it in GATHERING_ITEMS:
    gathering_by_prof[canon_prof(it.get("profession", "") or "")].append(it)

recipes_by_prof = defaultdict(list)
recipes_by_id = {}
for r in RECIPES:
    prof = canon_prof(r.get("profession", "") or "")
    rid = r.get("id") or f"{prof}|T{int(r.get('tier', 1))}|{r.get('name','')}"
    r["id"] = rid
    recipes_by_prof[prof].append(r)
    recipes_by_id[rid] = r

RECIPE_BASE_INDEX: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
for r in RECIPES:
    prof = canon_prof(r.get("profession", "") or "")
    bases = sorted([base_name(c.get("name", "")) for c in r.get("components", [])])
    if len(bases) == 3:
        key = (prof, bases[0], bases[1], bases[2])
        RECIPE_BASE_INDEX[key].append(r)
for k in list(RECIPE_BASE_INDEX.keys()):
    RECIPE_BASE_INDEX[k].sort(key=lambda rr: int(rr.get("tier", 1)))

# -----------------------------
# Supabase persistence (player_state table)
# -----------------------------
def sb_load_player_state(pname: str) -> Optional[Dict[str, Any]]:
    if SB is None:
        return None
    try:
        res = SB.table("player_state").select("state").eq("player_name", pname).execute()
        data = getattr(res, "data", None) or []
        if not data:
            return None
        return data[0].get("state")
    except Exception:
        return None

def sb_save_player_state(pname: str, state: Dict[str, Any]) -> bool:
    if SB is None:
        return False
    try:
        SB.table("player_state").upsert({"player_name": pname, "state": state}).execute()
        return True
    except Exception:
        return False

def init_session_state():
    if "players" not in st.session_state:
        st.session_state.players = normalize_players(copy.deepcopy(PLAYERS_DEFAULT))
    else:
        st.session_state.players = normalize_players(st.session_state.players)
    if "inventories" not in st.session_state:
        st.session_state.inventories = {p["name"]: {} for p in st.session_state.players}
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []
    if "vendor_offers" not in st.session_state:
        st.session_state.vendor_offers = {}
    if "gather_results" not in st.session_state:
        st.session_state.gather_results = {}
    if "jobs" not in st.session_state:
        st.session_state.jobs = {}
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = {}  # per player list of timed jobs
    if "notices" not in st.session_state:
        st.session_state.notices = {}  # per player list of messages

init_session_state()

def get_player(pname: str) -> Dict[str, Any]:
    for p in st.session_state.players:
        if p.get("name") == pname:
            return p
    raise KeyError(pname)

def sb_bootstrap():
    if SB is None:
        return
    for p in st.session_state.players:
        pname = p.get("name")
        if not pname:
            continue
        saved = sb_load_player_state(pname)
        if not saved:
            init_state = {
                "player": {"skills": p.get("skills", {}), "known_recipes": p.get("known_recipes", [])},
                "inventory": st.session_state.inventories.get(pname, {}),
                "jobs": [],
            }
            sb_save_player_state(pname, init_state)
            continue
        inv = saved.get("inventory", {})
        if isinstance(inv, dict):
            st.session_state.inventories[pname] = {canon_name(k): int(v) for k, v in inv.items() if int(v) > 0}
        pdata = saved.get("player", {})
        if isinstance(pdata, dict):
            if "skills" in pdata:
                p["skills"] = pdata.get("skills", p.get("skills", {}))
            if "known_recipes" in pdata:
                p["known_recipes"] = pdata.get("known_recipes", p.get("known_recipes", []))

        # Load timed jobs
        j = saved.get("jobs", [])
        if isinstance(j, list):
            st.session_state.jobs[pname] = j
        else:
            st.session_state.jobs[pname] = []

        # activity log
        st.session_state.activity_log.setdefault(pname, [])
        lg = saved.get('activity_log', [])
        if isinstance(lg, list):
            st.session_state.activity_log[pname] = lg

def save_player_now(pname: str):
    if SB is None:
        return
    pl = get_player(pname)
    state = {
        "player": {"skills": pl.get("skills", {}), "known_recipes": pl.get("known_recipes", [])},
        "inventory": st.session_state.inventories.get(pname, {}),
        "jobs": st.session_state.jobs.get(pname, []),
        "activity_log": st.session_state.get("activity_log", {}).get(pname, []),
    }
    sb_save_player_state(pname, state)

sb_bootstrap()

# -----------------------------
# Undo
# -----------------------------
def push_undo(player_name: str, label: str):
    st.session_state.undo_stack.append({
        "player": player_name,
        "label": label,
        "prev_inv": copy.deepcopy(st.session_state.inventories[player_name]),
        "prev_players": copy.deepcopy(st.session_state.players),
        "prev_vendor": copy.deepcopy(st.session_state.vendor_offers.get(player_name)),
        "prev_gather": copy.deepcopy(st.session_state.gather_results.get(player_name)),
    })

def restore_from_undo():
    last = st.session_state.undo_stack.pop()
    pname = last["player"]
    st.session_state.inventories[pname] = last["prev_inv"]
    st.session_state.players = last["prev_players"]
    if last.get("prev_vendor") is None:
        st.session_state.vendor_offers.pop(pname, None)
    else:
        st.session_state.vendor_offers[pname] = last["prev_vendor"]
    if last.get("prev_gather") is None:
        st.session_state.gather_results.pop(pname, None)
    else:
        st.session_state.gather_results[pname] = last["prev_gather"]
    save_player_now(pname)
    st.success(f"Undid: {last['label']} ({pname})")
    st.rerun()

# -----------------------------
# Core mutations
# -----------------------------
def apply_xp_delta(player: Dict[str, Any], skill: str, delta: int):
    skills = player.setdefault("skills", {})
    data = skills.setdefault(skill, {"level": 1, "xp": 0})
    level = int(data.get("level", 1))
    xp = int(data.get("xp", 0))
    xp = max(0, xp + int(delta))

    while True:
        need = xp_to_next(level)
        if need <= 0:
            break
        if xp >= need:
            xp -= need
            level += 1
        else:
            break

    data["level"] = level
    data["xp"] = xp

def add_item(inv: Dict[str, int], item_name: str, qty: int = 1):
    item_name = canon_name(item_name)
    inv[item_name] = inv.get(item_name, 0) + int(qty)
    if inv[item_name] <= 0:
        inv.pop(item_name, None)

def remove_item(inv: Dict[str, int], item_name: str, qty: int = 1):
    item_name = canon_name(item_name)
    inv[item_name] = inv.get(item_name, 0) - int(qty)
    if inv[item_name] <= 0:
        inv.pop(item_name, None)

def can_craft(inv: Dict[str, int], recipe: Dict[str, Any]) -> bool:
    for c in recipe.get("components", []):
        nm = canon_name(c.get("name", ""))
        need = int(c.get("qty", 1))
        if inv.get(nm, 0) < need:
            return False
    return True

def consume_recipe_mats(inv: Dict[str, int], recipe: Dict[str, Any], times: int = 1):
    """Consume recipe components from inventory. Assumes craftability was checked."""
    times = max(1, int(times))
    for _ in range(times):
        for c in recipe.get("components", []) or []:
            nm = canon_name(c.get("name", ""))
            need = int(c.get("qty", 1) or 1)
            remove_item(inv, nm, need)



def crafting_xp_from_components(recipe: Dict[str, Any]) -> int:
    """XP awarded on a successful craft/discovery.

    XP scales with the *amount* of materials consumed:
    e.g. three T1 components => 3 XP.
    """
    xp = 0
    for c in recipe.get("components", []) or []:
        nm = canon_name(c.get("name", ""))
        qty = int(c.get("qty", 1) or 1)
        tier = int(ITEM_TIER.get(nm, int(recipe.get("tier", 1) or 1)))
        xp += max(1, tier) * max(1, qty)
    return int(xp or int(recipe.get("tier", 1) or 1))

def crafts_possible(inv: Dict[str, int], recipe: Dict[str, Any]) -> int:
    """How many times this recipe can currently be crafted from inventory."""
    mins: List[int] = []
    for c in recipe.get("components", []) or []:
        nm = canon_name(c.get("name", ""))
        need = int(c.get("qty", 1) or 1)
        have = int(inv.get(nm, 0) or 0)
        mins.append(have // need)
    return int(min(mins) if mins else 0)

def gathering_xp_for_item(item_name: str) -> int:
    return int(ITEM_TIER.get(canon_name(item_name), 1))

# -----------------------------
# DC + Timed Actions (lightweight)
# -----------------------------
DC_BY_TIER = {1: 10, 2: 12, 3: 14, 4: 16, 5: 18, 6: 20, 7: 22}

# Real-time timers per tier (seconds). You asked: T6 = 2 hours.
TIMER_DISCOVER_SEC = {1: 60, 2: 5*60, 3: 15*60, 4: 30*60, 5: 60*60, 6: 2*60*60, 7: 3*60*60}
TIMER_CRAFT_SEC    = {1: 60, 2: 5*60, 3: 15*60, 4: 30*60, 5: 60*60, 6: 2*60*60, 7: 3*60*60}

def now_ts() -> int:
    import time as _time
    return int(_time.time())

def fmt_seconds(sec: int) -> str:
    sec = max(0, int(sec))
    h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"

def get_jobs(pname: str) -> List[Dict[str, Any]]:
    j = st.session_state.jobs.setdefault(pname, [])
    if not isinstance(j, list):
        st.session_state.jobs[pname] = []
    return st.session_state.jobs[pname]

def add_notice(pname: str, msg: str, *, kind: str = "note", items: Optional[List[str]] = None):
    """Persist a user-facing message for this player (shown once, also kept in Activity Log).

    kind: short tag like "discover", "craft", "vendor", "gather".
    items: optional list of item names involved (shown in Activity Log).
    """
    msg = str(msg)
    st.session_state.notices.setdefault(pname, []).append(msg)

    st.session_state.activity_log.setdefault(pname, [])
    entry = {"ts": now_ts(), "kind": kind, "msg": msg}
    if items:
        entry["items"] = [canon_name(x) for x in items if x]
    st.session_state.activity_log[pname].append(entry)
    st.session_state.activity_log[pname] = st.session_state.activity_log[pname][-120:]  # cap
def pop_notices(pname: str) -> List[str]:
    """Return only the notices that haven't been shown yet this session."""
    st.session_state.notice_cursor = st.session_state.get("notice_cursor", {})
    cur = int(st.session_state.notice_cursor.get(pname, 0))
    all_msgs = st.session_state.notices.get(pname, [])
    new_msgs = all_msgs[cur:]
    st.session_state.notice_cursor[pname] = len(all_msgs)
    return new_msgs
def active_job(pname: str) -> Optional[Dict[str, Any]]:
    for j in get_jobs(pname):
        if not j.get("done") and int(j.get("ends_at", 0)) > now_ts():
            return j
    return None

def start_job(pname: str, job: Dict[str, Any]) -> bool:
    # Only one active job per player to keep phone UX simple.
    if active_job(pname):
        add_notice(pname, "You already have an action in progress. Finish it before starting another.")
        return False
    job = dict(job)
    job.setdefault("id", f"job_{now_ts()}_{random.randint(1000,9999)}")
    job["started_at"] = now_ts()
    job.setdefault("done", False)
    get_jobs(pname).append(job)
    save_player_now(pname)
    return True

def complete_job(pname: str, job: Dict[str, Any]):
    """Apply job outcome when timer ends."""
    pl = get_player(pname)
    inv = st.session_state.inventories.setdefault(pname, {})
    jtype = job.get("type")

    if jtype == "discover":
        if job.get("success") and job.get("recipe_id") in recipes_by_id:
            rid = job["recipe_id"]
            if rid not in pl.get("known_recipes", []):
                pl.setdefault("known_recipes", []).append(rid)
            r = recipes_by_id[rid]
            add_item(inv, canon_name(r.get("name", "")), 1)
            xp_gain = int(job.get("xp_gain", 0))
            if xp_gain > 0:
                apply_xp_delta(pl, canon_prof(r.get("profession", "")), xp_gain)
            add_notice(pname, f"✅ You discovered **{r.get('name')}** and crafted 1!", kind="discover", items=job.get("items"))
        else:
            add_notice(pname, str(job.get("result_msg", "Discovery finished.")), kind="discover", items=job.get("items"))

    elif jtype == "craft":
        rid = job.get("recipe_id")
        if job.get("success") and rid in recipes_by_id:
            r = recipes_by_id[rid]
            out_qty = int(job.get("output_qty", 0) or 0)
            if out_qty <= 0:
                out_qty = int(r.get("output_qty", 1) or 1)
            add_item(inv, canon_name(r.get("name", "")), out_qty)
            xp_gain = int(job.get("xp_gain", 0))
            if xp_gain > 0:
                apply_xp_delta(pl, canon_prof(r.get("profession", "")), xp_gain)
            add_notice(pname, f"✅ Crafted {out_qty} × **{r.get('name')}**!", kind="craft", items=job.get("items"))
        else:
            add_notice(pname, str(job.get("result_msg", "Crafting finished.")), kind="craft", items=job.get("items"))

    job["done"] = True
    save_player_now(pname)

def process_jobs(pname: str):
    changed = False
    for j in list(get_jobs(pname)):
        if j.get("done"):
            continue
        if now_ts() >= int(j.get("ends_at", 0)):
            complete_job(pname, j)
            changed = True
    if changed:
        st.session_state.jobs[pname] = [x for x in get_jobs(pname) if not x.get("done")]
        save_player_now(pname)

def dc_for_tier(recipe_tier: int, player_tier: int) -> int:
    """Base DC by recipe tier, eased if player tier is higher (cap the discount)."""
    base = int(DC_BY_TIER.get(int(recipe_tier), 10))
    diff = int(player_tier) - int(recipe_tier)
    if diff > 0:
        base -= min(6, 2 * diff)
    elif diff < 0:
        base += min(6, 2 * abs(diff))
    return max(5, base)

def discovery_hint_for_pair(craft_prof: str, chosen: list, unlocked_tier: int) -> str:
    """If two of the three items match a recipe, hint which two and what third to try.

    Uses multiset matching (so recipes that require the same component twice work correctly).
    """
    chosen_bases = [base_name(canon_name(x)) for x in chosen if x]
    if len(chosen_bases) != 3:
        return ""
    chosen_ctr = Counter(chosen_bases)

    # Candidate recipes in the player’s visibility window
    candidates = [r for r in RECIPES if canon_prof(r.get("profession", "")) == canon_prof(craft_prof)]
    candidates = [r for r in candidates if int(r.get("tier", 1)) <= int(unlocked_tier) + 2]

    best = None
    best_overlap = 0
    best_missing = ""
    best_pair = []

    for r in candidates:
        comps = [base_name(canon_name(c.get("name", ""))) for c in r.get("components", [])]
        if len(comps) != 3:
            continue
        r_ctr = Counter(comps)
        overlap = sum(min(chosen_ctr[k], r_ctr[k]) for k in r_ctr)
        # We only hint when at least 2 components align, but the full recipe is not correct
        if overlap < 2:
            continue
        if overlap == 3 and all(chosen_ctr[k] == r_ctr[k] for k in set(list(chosen_ctr.keys()) + list(r_ctr.keys()))):
            continue

        # Determine the missing component (the one where chosen has fewer than recipe requires)
        missing = ""
        for k, need in r_ctr.items():
            if chosen_ctr.get(k, 0) < need:
                missing = k
                break
        if not missing:
            continue

        # Determine which two components of the recipe are matched (respecting duplicate requirements)
        matched = []
        tmp = chosen_ctr.copy()
        for k, need in r_ctr.items():
            take = min(tmp.get(k, 0), need)
            for _ in range(take):
                if len(matched) < 2:
                    matched.append(k)
            if k in tmp:
                tmp[k] = max(0, tmp[k] - take)
        if len(matched) < 2:
            continue

        if overlap > best_overlap:
            best_overlap = overlap
            best = r
            best_missing = missing
            best_pair = matched[:2]

    if best and best_pair and best_missing:
        a, b = best_pair[0], best_pair[1]
        return f"Try pairing **{a}** + **{b}** with something like **{best_missing} (any tier)**."
    return ""
def gathered_tier_from_roll(unlocked_tier: int, roll_total: int) -> Optional[int]:
    if unlocked_tier == 1 and roll_total < 10:
        return None
    if roll_total >= 22:
        target = unlocked_tier + 2
    elif roll_total >= 18:
        target = unlocked_tier + 1
    elif roll_total >= 12:
        target = unlocked_tier
    else:
        target = max(1, unlocked_tier - 1)
    return max(1, min(7, target))

def choose_random_gather_item(prof: str, tier: int) -> Optional[Dict[str, Any]]:
    prof = canon_prof(prof)
    pool = [it for it in gathering_by_prof.get(prof, []) if int(it.get("tier", 1)) == tier]
    return random.choice(pool) if pool else None

# -----------------------------
# Vendor
# -----------------------------
def choose_vendor_tier(unlocked: int) -> int:
    key = weighted_choice(VENDOR_TIER_WEIGHTS)
    if key == "T":
        t = unlocked
    elif key == "T+1":
        t = unlocked + 1
    else:
        t = unlocked + 2
    return max(1, min(7, t))

def choose_vendor_qty() -> int:
    q = int(weighted_choice(VENDOR_QTY_WEIGHTS))
    return max(1, min(VENDOR_QTY_MAX, q))

def generate_vendor_stock_for_prof(player: Dict[str, Any], chosen_prof: str) -> List[Dict[str, Any]]:
    chosen_prof = canon_prof(chosen_prof)
    skills = player.get("skills", {})
    skill_level = int(skills.get(chosen_prof, {}).get("level", 1))
    unlocked = max_tier_for_level(skill_level)

    n_lines = int(weighted_choice(VENDOR_LINECOUNT_WEIGHTS))
    n_lines = max(0, min(VENDOR_MAX_LINES, n_lines))
    if n_lines == 0:
        return []

    if chosen_prof in gathering_by_prof:
        candidates = list(gathering_by_prof[chosen_prof])
    else:
        mats = CRAFT_PROF_TO_MATS.get(chosen_prof, set())
        candidates = [it for it in GATHERING_ITEMS if canon_name(it.get("name", "")) in mats]

    max_allowed_tier = min(7, unlocked + 2)
    by_tier = defaultdict(list)
    for it in candidates:
        t = int(it.get("tier", 1))
        if 1 <= t <= max_allowed_tier:
            by_tier[t].append(it)

    lines = []
    used_names = set()
    attempts = 0
    while len(lines) < n_lines and attempts < 120:
        attempts += 1
        target_tier = min(choose_vendor_tier(unlocked), max_allowed_tier)

        pool = by_tier.get(target_tier, [])
        if not pool:
            tt = target_tier
            while tt >= 1 and not by_tier.get(tt):
                tt -= 1
            pool = by_tier.get(tt, [])
            if not pool:
                continue
            target_tier = tt

        it = random.choice(pool)
        nm = canon_name(it.get("name", ""))
        if not nm:
            continue
        if nm in used_names and len(pool) > 1:
            continue

        qty = choose_vendor_qty()
        unit = effective_unit_price(it)
        total = unit * qty

        used_names.add(nm)
        lines.append({
            "name": nm,
            "tier": int(it.get("tier", 1)),
            "qty": int(qty),
            "unit_price": float(unit),
            "total_price": float(total),
            "unlocked_tier": unlocked,
        })

    return lines

# -----------------------------
# Discovery (3 items, consumed always)
# -----------------------------
def validate_discovery_selection(inv: Dict[str, int], picks: List[str]) -> Tuple[bool, str]:
    counts = defaultdict(int)
    for p in picks:
        p = canon_name(p)
        if not p:
            return False, "Pick 3 items."
        counts[p] += 1
    for nm, need in counts.items():
        if inv.get(nm, 0) < need:
            return False, f"You selected {nm} {need}x but only have {inv.get(nm, 0)}."
    return True, ""

def consume_selected(inv: Dict[str, int], picks: List[str]):
    counts = defaultdict(int)
    for p in picks:
        counts[canon_name(p)] += 1
    for nm, q in counts.items():
        remove_item(inv, nm, q)

def match_recipe_mixed_tiers(craft_prof: str, chosen_items: List[str], unlocked_tier: int) -> Optional[Dict[str, Any]]:
    craft_prof = canon_prof(craft_prof)
    chosen = [canon_name(x) for x in chosen_items if x]
    if len(chosen) != 3:
        return None

    chosen_bases = sorted([base_name(x) for x in chosen])
    key = (craft_prof, chosen_bases[0], chosen_bases[1], chosen_bases[2])
    candidates = RECIPE_BASE_INDEX.get(key, [])
    if not candidates:
        return None

    max_visible = min(7, unlocked_tier + 2)
    candidates = [r for r in candidates if int(r.get("tier", 1)) <= max_visible]
    if not candidates:
        return None

    chosen_tiers = [int(ITEM_TIER.get(x, 1)) for x in chosen]
    tmin = min(chosen_tiers) if chosen_tiers else 1

    preferred = [r for r in candidates if int(r.get("tier", 1)) <= tmin]
    if preferred:
        return preferred[0]
    return candidates[0]

# -----------------------------
# UI
# -----------------------------
st.title("🛠 D&D Crafting Simulator")

if SB is None:
    st.warning("Supabase not configured (or supabase package missing). App will NOT persist after refresh.")

top1, top2 = st.columns([1, 6])
with top1:
    if st.button("↩️ Undo", disabled=(len(st.session_state.undo_stack) == 0)):
        restore_from_undo()
with top2:
    st.caption(
        f"Last: {st.session_state.undo_stack[-1]['label']} ({st.session_state.undo_stack[-1]['player']})"
        if st.session_state.undo_stack else "No actions to undo yet."
    )

tabs = st.tabs([p["name"] for p in st.session_state.players])

for idx, player in enumerate(st.session_state.players):
    pname = player["name"]
    inv: Dict[str, int] = st.session_state.inventories[pname]
    skills: Dict[str, Dict[str, Any]] = player.get("skills", {})

    # Profession data can be stored explicitly OR inferred from the skills dict
    gathering_prof = canon_prof(player.get("gathering_profession", "") or "")
    if not gathering_prof:
        for k in skills.keys():
            ck = canon_prof(k)
            if ck in gathering_professions:
                gathering_prof = ck
                break

    crafting_profs = [canon_prof(x) for x in (player.get("crafting_professions", []) or [])]
    if not crafting_profs:
        for k in skills.keys():
            ck = canon_prof(k)
            if ck in crafting_professions:
                crafting_profs.append(ck)

    with tabs[idx]:
        st.subheader(f"👤 {pname}")

        process_jobs(pname)
        for _msg in pop_notices(pname):
            st.success(_msg)

        with st.expander("📜 Activity log", expanded=False):
            log = st.session_state.get("activity_log", {}).get(pname, []) or []
            if not log:
                st.caption("No activity yet.")
            else:
                for entry in reversed(log[-25:]):
                    ts = datetime.datetime.fromtimestamp(int(entry.get("ts", 0))).strftime("%Y-%m-%d %H:%M:%S")
                    kind = entry.get("kind", "")
                    ktag = f" [{kind}]" if kind and kind != "note" else ""
                    st.write(f"- **{ts}**{ktag}: {entry.get('msg', '')}")
                    it = entry.get("items") or []
                    if it:
                        st.caption("Items: " + ", ".join([str(x) for x in it]))

        _job = active_job(pname)
        if _job:
            remaining = int(_job.get("ends_at", 0)) - now_ts()
            total = int(_job.get("ends_at", 0)) - int(_job.get("started_at", now_ts()))
            total = max(1, total)
            prog = min(max((total - remaining) / total, 0.0), 1.0)
            with st.expander("⏳ Action in progress", expanded=True):
                st.write(f"**{_job.get('type', 'action').title()}** (Tier {_job.get('tier', '?')})")
                st.progress(prog)
                st.caption(f"Time remaining: {fmt_seconds(remaining)}")

        # ---- Skills ----
        st.markdown("### Skills")
        cols = st.columns(2)
        for i, s in enumerate(list(skills.keys())):
            data = skills[s]
            level = int(data.get("level", 1))
            cur_xp = int(data.get("xp", 0))
            needed = xp_to_next(level)
            unlocked_tier = max_tier_for_level(level)
            next_tier = min(7, unlocked_tier + 1)
            next_lvl = next_unlock_level_for_tier(next_tier)
            progress = 0.0 if needed <= 0 else min(cur_xp / needed, 1.0)

            with cols[i % 2]:
                st.write(f"**{s}**")
                st.progress(progress)
                if next_lvl:
                    st.caption(f"Level {level} • Tier {unlocked_tier} • Unlock T{next_tier} at level {next_lvl} • XP {cur_xp}/{needed}")
                else:
                    st.caption(f"Level {level} • Tier {unlocked_tier} • XP {cur_xp}/{needed}")

                b1, b2, _ = st.columns([1, 1, 5])
                with b1:
                    if st.button("−1 XP", key=f"{pname}-{s}-xpminus"):
                        push_undo(pname, f"XP −1 ({s})")
                        apply_xp_delta(get_player(pname), s, -1)
                        save_player_now(pname)
                        st.rerun()
                with b2:
                    if st.button("+1 XP", key=f"{pname}-{s}-xpplus"):
                        push_undo(pname, f"XP +1 ({s})")
                        apply_xp_delta(get_player(pname), s, +1)
                        save_player_now(pname)
                        st.rerun()

        st.divider()

        # ---- Inventory ----
        with st.expander("🎒 Inventory", expanded=False):
            c1, c2, c3 = st.columns([2, 2, 2])
            sort_choice = c1.selectbox("Sort", ["Name", "Tier", "Quantity"], key=f"{pname}-inv-sort")
            filter_mode = c2.selectbox("Filter", ["All", "By profession"], key=f"{pname}-inv-filter-mode")
            tier_filter = c3.selectbox("Tier", ["All"] + [f"T{i}" for i in range(1, 8)], key=f"{pname}-inv-tier-filter")

            prof_filter = None
            if filter_mode == "By profession":
                all_profs = sorted(set(ITEM_PROF.get(nm, "") for nm in inv.keys() if ITEM_PROF.get(nm, "")))
                prof_filter = st.selectbox("Profession", ["All"] + all_profs, key=f"{pname}-inv-prof-filter")

            rows = []
            for nm, qty in inv.items():
                nm = canon_name(nm)
                t = int(ITEM_TIER.get(nm, 1))
                if tier_filter != "All" and t != int(tier_filter[1:]):
                    continue
                if prof_filter and prof_filter != "All" and ITEM_PROF.get(nm, "") != prof_filter:
                    continue
                vend = float(ITEM_VENDOR.get(nm, FALLBACK_TIER_GP.get(t, 0)) or 0)
                sell = math.floor(vend * SELL_RATE)
                rows.append((nm, t, qty, sell))

            if sort_choice == "Name":
                rows.sort(key=lambda x: x[0].lower())
            elif sort_choice == "Tier":
                rows.sort(key=lambda x: (x[1], x[0].lower()))
            else:
                rows.sort(key=lambda x: (-x[2], x[0].lower()))

            if not rows:
                st.caption("Inventory is empty.")
            else:
                for nm, t, qty, sell in rows:
                    left, mid, right = st.columns([5, 2, 2])
                    with left:
                        st.write(f"**{nm}** ({tier_badge_html(99, t)})", unsafe_allow_html=True)
                        if ITEM_DESC.get(nm):
                            st.caption(ITEM_DESC[nm])
                        if ITEM_USE.get(nm):
                            st.caption(ITEM_USE[nm])
                    with mid:
                        st.write(f"Qty: **{qty}**")
                        st.caption(f"Sell: **{sell} gp**")
                                        with right:
                        bminus, bplus, _sp = st.columns([1, 1, 6])
                        with bminus:
                            if st.button("−", key=f"{pname}-inv-{nm}-minus"):
                                push_undo(pname, f"Inventory −1 ({nm})")
                                remove_item(inv, nm, 1)
                                save_player_now(pname)
                                st.rerun()
                        with bplus:
                            # Full-width plus renders more reliably than "+" in some fonts
                            if st.button("＋", key=f"{pname}-inv-{nm}-plus"):
                                push_undo(pname, f"Inventory +1 ({nm})")
                                add_item(inv, nm, 1)
                                save_player_now(pname)
                                st.rerun()

                        other_players = [pp.get("name") for pp in st.session_state.players if pp.get("name") and pp.get("name") != pname]
                        if other_players:
                            # Keep inventory rows compact: use a popover for sending (fallback to expander if unavailable)
                            try:
                                with st.popover("Send"):
                                    to_player = st.selectbox("Recipient", other_players, key=f"{pname}-send-to-{nm}")
                                    send_qty = st.number_input("Amount", min_value=1, max_value=int(qty), value=1, step=1, key=f"{pname}-send-qty-{nm}")
                                    if st.button("Send", key=f"{pname}-send-btn-{nm}"):
                                        push_undo(pname, f"Send {int(send_qty)}x {nm} → {to_player}")
                                        remove_item(inv, nm, int(send_qty))
                                        recv_inv = st.session_state.inventories[to_player]
                                        add_item(recv_inv, nm, int(send_qty))
                                        save_player_now(pname)
                                        save_player_now(to_player)
                                        add_notice(pname, f"Sent {int(send_qty)}× {nm} to {to_player}.", kind="trade", items=[nm])
                                        add_notice(to_player, f"Received {int(send_qty)}× {nm} from {pname}.", kind="trade", items=[nm])
                                        st.rerun()
                            except Exception:
                                with st.expander("Send to another player", expanded=False):
                                    to_player = st.selectbox("Recipient", other_players, key=f"{pname}-send-to-{nm}")
                                    send_qty = st.number_input("Amount", min_value=1, max_value=int(qty), value=1, step=1, key=f"{pname}-send-qty-{nm}")
                                    if st.button("Send", key=f"{pname}-send-btn-{nm}"):
                                        push_undo(pname, f"Send {int(send_qty)}x {nm} → {to_player}")
                                        remove_item(inv, nm, int(send_qty))
                                        recv_inv = st.session_state.inventories[to_player]
                                        add_item(recv_inv, nm, int(send_qty))
                                        save_player_now(pname)
                                        save_player_now(to_player)
                                        add_notice(pname, f"Sent {int(send_qty)}× {nm} to {to_player}.", kind="trade", items=[nm])
                                        add_notice(to_player, f"Received {int(send_qty)}× {nm} from {pname}.", kind="trade", items=[nm])
                                        st.rerun()
                        else:
                            st.caption("No other players to send items to.")


        # ---- Gathering ----
        with st.expander("⛏️ Gathering", expanded=False):
            if not gathering_prof:
                st.caption("No gathering profession.")
            else:
                skill_lvl = int(skills.get(gathering_prof, {}).get("level", 1))
                unlocked = max_tier_for_level(skill_lvl)

                st.markdown("#### Automated gathering (enter your in-game roll)")
                roll_total = st.number_input("Your total roll (d20 + modifiers)", min_value=0, max_value=60, value=10, step=1, key=f"{pname}-g-roll")

                if st.button("Roll gathering", key=f"{pname}-g-roll-btn"):
                    push_undo(pname, "Gathering roll")
                    target_tier = gathered_tier_from_roll(unlocked, int(roll_total))
                    if target_tier is None:
                        st.session_state.gather_results[pname] = {"failed": True}
                    else:
                        found = choose_random_gather_item(gathering_prof, target_tier)
                        st.session_state.gather_results[pname] = {"failed": False, "tier": int(target_tier), "item": found.get("name", "") if found else ""}
                    st.rerun()

                result = st.session_state.gather_results.get(pname)
                if not result:
                    st.caption("No roll yet. Press **Roll gathering**.")
                else:
                    if result.get("failed"):
                        st.error("Gathering failed! You didn’t find anything this time.")
                    else:
                        t = int(result.get("tier", 1))
                        item_name = canon_name(result.get("item", ""))
                        dc = dc_for_target_tier(unlocked, t)
                        st.markdown(f"Result tier: {tier_badge_html(unlocked, t)} • DC {dc if dc else '—'}", unsafe_allow_html=True)

                        if not item_name:
                            st.warning("No item found for that tier (check gathering_items.json).")
                        else:
                            xp_gain = gathering_xp_for_item(item_name)
                            st.write(f"**You found:** {item_name} ({tier_badge_html(unlocked, t)})", unsafe_allow_html=True)
                            if ITEM_DESC.get(item_name):
                                st.caption(ITEM_DESC[item_name])
                            if ITEM_USE.get(item_name):
                                st.caption(f"Use: {ITEM_USE[item_name]}")
                            st.caption(f"XP if gathered: **{xp_gain}**")

                            if st.button("Add to inventory (Gathered)", key=f"{pname}-g-add"):
                                push_undo(pname, f"Gathered {item_name}")
                                add_item(inv, item_name, 1)
                                apply_xp_delta(get_player(pname), gathering_prof, xp_gain)
                                st.session_state.gather_results.pop(pname, None)
                                save_player_now(pname)
                                st.rerun()

        # ---- Crafting ----
        with st.expander("🧪 Crafting", expanded=False):
            if not crafting_profs:
                st.caption("No crafting professions.")
            else:
                craft_prof = st.selectbox("Choose crafting profession", crafting_profs, key=f"{pname}-craft-prof")
                skill_lvl = int(skills.get(craft_prof, {}).get("level", 1))
                unlocked = max_tier_for_level(skill_lvl)

                known_ids = set(player.get("known_recipes", []))
                known_recipes = [recipes_by_id[rid] for rid in known_ids if rid in recipes_by_id and canon_prof(recipes_by_id[rid].get("profession", "")) == craft_prof]
                known_recipes = [r for r in known_recipes if int(r.get("tier", 1)) <= unlocked + 2]

                st.markdown("#### Discover recipes (3 items from your inventory)")
                # Roll input (required)
                roll_total = st.number_input("Your discovery roll total (d20 + modifiers)", min_value=0, max_value=50, value=0, step=1, key=f"disc_roll_{pname}_{craft_prof}")

                mats_allowed_exact = CRAFT_PROF_TO_MATS.get(craft_prof, set())
                mats_allowed_base = CRAFT_PROF_TO_MATS_BASE.get(craft_prof, set())

                inv_options = []
                for nm in inv.keys():
                    cn = canon_name(nm)
                    if cn in mats_allowed_exact or base_name(cn) in mats_allowed_base:
                        inv_options.append(cn)
                inv_options = sorted(set(inv_options), key=lambda x: x.lower())

                if not inv_options:
                    st.caption("You don’t have any valid crafting components in your inventory for this profession yet.")
                    st.caption("If you think you should: check that component names in recipes.json match gathering_items.json exactly (including (Tn)).")
                else:
                    d1, d2, d3 = st.columns(3)
                    m1 = d1.selectbox("Item 1", [""] + inv_options, key=f"{pname}-disc-1")
                    m2 = d2.selectbox("Item 2", [""] + inv_options, key=f"{pname}-disc-2")
                    m3 = d3.selectbox("Item 3", [""] + inv_options, key=f"{pname}-disc-3")
                    chosen = [m1, m2, m3]

                    if st.button("Try combination", key=f"{pname}-disc-try"):
                        ok, msg = validate_discovery_selection(inv, chosen)
                        if not ok:
                            st.error(msg)
                        else:
                            if roll_total <= 0:
                                st.error("Please enter your discovery roll total first.")

                            push_undo(pname, f"Discovery attempt ({craft_prof})")
                            # Consume items immediately (success or fail). No XP on failure.
                            consume_selected(inv, chosen)
                            
                            # Attempt to match a recipe (mixed tiers allowed, crafts at lowest tier that fits).
                            recipe = match_recipe_mixed_tiers(craft_prof, chosen, unlocked)
                            
                            # Determine target tier for DC/timer: matched recipe tier, else highest chosen tier capped by visibility.
                            chosen_tiers = []
                            for _nm in chosen:
                                m = TIER_SUFFIX_RE.search(str(_nm))
                                chosen_tiers.append(int(m.group(1)) if m else 1)
                            max_chosen = max(chosen_tiers) if chosen_tiers else 1
                            min_chosen = min(chosen_tiers) if chosen_tiers else 1

                            # Mixed-tier discovery crafts at the LOWEST tier used (and timer uses the same tier)
                            target_tier = min(7, max(1, min_chosen))
                            
                            dc = dc_for_tier(target_tier, unlocked)
                            timer_sec = int(TIMER_DISCOVER_SEC.get(target_tier, 60))
                            
                            success = False
                            result_msg = ""
                            rid = None
                            xp_gain = 0
                            
                            if recipe:
                                rid = recipe.get("id")
                                if roll_total >= dc:
                                    success = True
                                    xp_gain = 0  # xp is granted on completion only if success
                                    result_msg = f"Discovery underway... You feel the materials resonate. (DC {dc})"
                                else:
                                    # Valid recipe but failed: consume items, no XP.
                                    if roll_total < dc - 5:
                                        result_msg = "You’re unsure these materials belong together."
                                    else:
                                        result_msg = "You felt the materials respond to each other, but the crafting failed."
                            else:
                                # Invalid recipe: give tiered feedback
                                if roll_total < dc - 5:
                                    result_msg = "You’re unsure these materials belong together."
                                elif roll_total < dc:
                                    result_msg = "Something about this feels close, like 2 of these might belong to a real recipe."
                                else:
                                    hint = discovery_hint_for_pair(craft_prof, chosen, unlocked)
                                    result_msg = "You’re confident at least two of these materials resonate, but the full combination is wrong."
                                    if hint:
                                        result_msg += f"\n\n**Hint:** {hint}"
                            
                            ends_at = now_ts() + timer_sec
                            job = {
                                "type": "discover",
                                "profession": craft_prof,
                                "items": chosen,
                                "roll_total": int(roll_total),
                                "dc": int(dc),
                                "recipe_id": rid,
                                "success": bool(success),
                                "xp_gain": int(crafting_xp_from_components(recipe) if (success and recipe) else 0),
                                "result_msg": result_msg,
                                "ends_at": int(ends_at),
                                "tier": int(target_tier),
                            }
                            
                            if start_job(pname, job):
                                st.info(f"⏳ Discovery started (T{target_tier}). Time remaining: {fmt_seconds(timer_sec)}")
                                save_player_now(pname)
                                st.rerun()
                st.divider()
                st.markdown(f"#### Known recipes (visible up to T{unlocked + 2})")

                if not known_recipes:
                    st.caption("You haven’t learned any recipes yet.")
                else:
                    for t in range(1, min(8, unlocked + 3)):
                        tier_recipes = [r for r in known_recipes if int(r.get("tier", 1)) == t]
                        if not tier_recipes:
                            continue
                        with st.expander(f"T{t} recipes", expanded=(t == unlocked)):
                            st.markdown(f"**Tier:** {tier_badge_html(unlocked, t)}", unsafe_allow_html=True)
                            tier_recipes.sort(key=lambda x: (x.get("name", "") or "").lower())
                            for r in tier_recipes:
                                nm = canon_name(r.get("name", ""))
                                craftable_n = crafts_possible(inv, r)
                                xp_gain = crafting_xp_from_components(r)
                                dc = dc_for_target_tier(unlocked, t)

                                st.write(f"**{nm}** ({tier_badge_html(unlocked, t)})", unsafe_allow_html=True)
                                if r.get("description"):
                                    st.caption(r["description"])
                                status = "✅ Craftable" if craftable_n > 0 else "❌ Missing materials"
                                st.caption(f"{status} • Can craft: **{craftable_n}** • DC {dc if dc else '—'} • XP per craft: **{xp_gain}**")

                                with st.expander("Show recipe details", expanded=False):
                                    for c in r.get("components", []):
                                        cname = canon_name(c.get("name", ""))
                                        need = int(c.get("qty", 1))
                                        have = int(inv.get(cname, 0))
                                        st.write(f"- {cname}: **{have} / {need}** {'✅' if have >= need else '❌'}")
                                    if r.get("use"):
                                        st.caption(f"Use: {r['use']}")

                                left, right = st.columns([2, 6])
with left:
    # Require an in-game roll (total).
    roll_total = st.number_input(
        "Craft roll total",
        min_value=0,
        step=1,
        key=f"craft_roll_{pname}_{r['id']}"
    )

    # Quantity (bounded by what inventory supports)
    max_q = max(1, int(craftable_n or 1))
    craft_qty = st.number_input(
        "Quantity",
        min_value=1,
        max_value=max_q,
        step=1,
        value=1,
        key=f"craft_qty_{pname}_{r['id']}"
    )

    tier = int(r.get('tier', 1) or 1)
    dc_eff = dc_for_tier(tier, unlocked)
    timer_sec = int(TIMER_CRAFT_SEC.get(tier, 60))

    st.caption(f"DC {dc_eff} • Time: {fmt_seconds(timer_sec)}")

    busy = active_job(pname) is not None
    disabled = (craftable_n <= 0) or busy or roll_total <= 0

    if st.button("Craft", key=f"craft_btn_{pname}_{r['id']}", disabled=disabled):
        push_undo(pname, f"Craft attempt ({r.get('name')})")
        # Consume immediately (success or fail), consistent with your design.
        consume_recipe_mats(inv, r, times=int(craft_qty))

        success = int(roll_total) >= int(dc_eff)
        xp_per = int(crafting_xp_from_components(r))
        xp_gain = int(xp_per * int(craft_qty)) if success else 0

        if success:
            msg = f"Crafting underway... (DC {dc_eff})"
        else:
            if int(roll_total) < int(dc_eff) - 5:
                msg = "The work falls apart early. The materials are spent."
            else:
                msg = "So close... but the craft fails at the last moment. The materials are spent."

        job = {
            "type": "craft",
            "profession": craft_prof,
            "recipe_id": r.get("id"),
            "items": [canon_name(c.get("name","")) for c in (r.get("components") or [])],
            "roll_total": int(roll_total),
            "dc": int(dc_eff),
            "success": bool(success),
            "xp_gain": int(xp_gain),
            "result_msg": msg,
            "ends_at": int(now_ts() + timer_sec),
            "tier": int(tier),
            "craft_qty": int(craft_qty),
            "output_qty": int(r.get("output_qty", 1) or 1) * int(craft_qty),
        }

        if start_job(pname, job):
            st.info(f"⏳ Craft started (T{tier}). Time remaining: {fmt_seconds(timer_sec)}")
            save_player_now(pname)
            st.rerun()


        # ---- Vendor ----
        with st.expander("🧾 Vendor", expanded=False):
            prof_options = []
            if gathering_prof:
                prof_options.append(gathering_prof)
            prof_options.extend(crafting_profs)

            if not prof_options:
                st.caption("No professions available for vendor.")
            else:
                chosen_prof = st.selectbox("Shop type (your professions)", prof_options, key=f"{pname}-vendor-prof")
                st.caption("If you pick a crafting profession, the vendor sells ONLY components used by that craft.")
                st.caption("Per visit: 0–3 items, max 4 qty each. Vendor never sells above **T+2**.")

                if st.button("Generate vendor stock", key=f"{pname}-vendor-roll"):
                    push_undo(pname, f"Vendor roll ({chosen_prof})")
                    lines = generate_vendor_stock_for_prof(get_player(pname), chosen_prof)
                    st.session_state.vendor_offers[pname] = {"profession": chosen_prof, "lines": lines}
                    st.rerun()

                offer = st.session_state.vendor_offers.get(pname)
                if offer is None:
                    st.caption("No vendor stock yet. Click “Generate vendor stock”.")
                else:
                    st.markdown(f"**Vendor stock for:** {offer.get('profession')}")
                    lines = offer.get("lines", []) or []
                    if not lines:
                        st.info("This vendor has nothing useful right now.")
                    for i, line in enumerate(lines):
                        nm = canon_name(line.get("name", ""))
                        t = safe_int(line.get("tier", 1), 1)
                        qty = safe_int(line.get("qty", 1), 1)
                        unit = safe_float(line.get("unit_price", 0), 0.0)
                        total = unit * qty

                        unit_disp = f"{safe_int(unit)} gp" if unit > 0 else "—"
                        total_disp = f"{safe_int(total)} gp" if total > 0 else "—"

                        st.write(
                            f"**{nm}** ({tier_badge_html(line.get('unlocked_tier', 99), t)}) • "
                            f"Qty: **{qty}** • Unit: **{unit_disp}** • Total: **{total_disp}**",
                            unsafe_allow_html=True
                        )

                        b1, _ = st.columns([2, 6])
                        with b1:
                            # Buy purchases 1 unit at a time and decrements the vendor stock.
                            if st.button("Buy", key=f"{pname}-vendor-buy-{i}-{nm}"):
                                if qty <= 0:
                                    st.warning("Out of stock.")
                                    st.rerun()

                                push_undo(pname, f"Vendor buy {nm} x1")
                                add_item(inv, nm, 1)

                                # Decrement remaining stock for this line
                                offer = st.session_state.vendor_offers.get(pname, {})
                                lines2 = list(offer.get("lines", []) or [])
                                if 0 <= i < len(lines2):
                                    lines2[i]["qty"] = max(0, safe_int(lines2[i].get("qty", 1), 1) - 1)

                                # Remove any lines that hit 0
                                lines2 = [ln for ln in lines2 if safe_int(ln.get("qty", 0), 0) > 0]
                                offer["lines"] = lines2
                                st.session_state.vendor_offers[pname] = offer

                                save_player_now(pname)
                                st.rerun()
                        st.divider()
