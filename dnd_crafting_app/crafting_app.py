import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
import copy
import random
import math
import re
import time

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

# Vendor limits
VENDOR_MAX_LINES = 3
VENDOR_QTY_MAX = 4
VENDOR_TIER_WEIGHTS = {"T": 0.90, "T+1": 0.45, "T+2": 0.15}  # capped at T+2
VENDOR_QTY_WEIGHTS = {1: 0.50, 2: 0.35, 3: 0.20, 4: 0.05}
VENDOR_LINECOUNT_WEIGHTS = {0: 0.10, 1: 0.35, 2: 0.35, 3: 0.20}

# Fallback GP per tier if vendor_price missing
FALLBACK_TIER_GP = {1: 2, 2: 5, 3: 11, 4: 20, 5: 45, 6: 90, 7: 180}

# Timers (real time)
# User constraint: T6 = 2 hours; no XP for failed crafting/discovery
TIER_SECONDS = {
    1: 60,
    2: 5 * 60,
    3: 15 * 60,
    4: 30 * 60,
    5: 60 * 60,
    6: 2 * 60 * 60,
    7: 4 * 60 * 60,
}

# -----------------------------
# Helpers
# -----------------------------
TIER_SUFFIX_RE = re.compile(r"\s*\(T(\d)\)\s*$", re.IGNORECASE)

def canon_prof(s: str) -> str:
    return (s or "").strip()

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
        if isinstance(x, str) and x.strip().lower() == "nan":
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
    # Simple, readable DC model consistent with earlier versions:
    # within tier: 10, +1 tier: 15, +2 tiers: 20 (anything above hidden anyway)
    if target_tier <= unlocked_tier:
        return 10
    if target_tier == unlocked_tier + 1:
        return 15
    if target_tier == unlocked_tier + 2:
        return 20
    return None

def tier_color(unlocked: int, target: int) -> str:
    if target <= unlocked - 1:
        return "#16a34a"  # green
    if target == unlocked:
        return "#111827"  # black
    if target == unlocked + 1:
        return "#b45309"  # yellow-ish
    return "#b91c1c"      # red

def tier_badge_html(unlocked: int, target: int) -> str:
    col = tier_color(unlocked, target)
    return f'<span style="color:{col};font-weight:700;">T{target}</span>'

# -----------------------------
# Pricing
# -----------------------------
PRICE_KEYS_VENDOR = ["vendor_price", "vendorPrice", "vendor_gp", "vendorGp", "vendorCost", "vendor_cost"]
PRICE_KEYS_BASE = ["base_price", "basePrice", "base_gp", "baseGp", "baseCost", "base_cost", "price", "cost", "gp"]

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

gathering_by_prof = defaultdict(list)
for it in GATHERING_ITEMS:
    nm = canon_name(it.get("name", ""))
    if not nm:
        continue
    t = safe_int(it.get("tier", 1), 1)
    ITEM_TIER[nm] = t
    ITEM_PROF[nm] = canon_prof(it.get("profession", "") or "")
    ITEM_DESC[nm] = (it.get("description") or "").strip()
    ITEM_USE[nm] = (it.get("use") or "").strip()
    ITEM_VENDOR[nm] = effective_unit_price(it)
    gathering_by_prof[ITEM_PROF[nm]].append(it)

recipes_by_prof = defaultdict(list)
recipes_by_id: Dict[str, Dict[str, Any]] = {}

RECIPE_BASE_INDEX: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
PAIR_HINT_INDEX: Dict[Tuple[str, str, str], Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
# key: (prof, a, b) -> {third: [recipe_ids...]}

for r in RECIPES:
    prof = canon_prof(r.get("profession", "") or "")
    rid = r.get("id") or f"{prof}|T{safe_int(r.get('tier', 1), 1)}|{r.get('name','')}"
    r["id"] = rid
    recipes_by_prof[prof].append(r)
    recipes_by_id[rid] = r

    bases = sorted([base_name(c.get("name", "")) for c in r.get("components", []) if c.get("name")])
    if len(bases) == 3:
        key = (prof, bases[0], bases[1], bases[2])
        RECIPE_BASE_INDEX[key].append(r)

        a, b, c = bases
        PAIR_HINT_INDEX[(prof, a, b)][c].append(rid)
        PAIR_HINT_INDEX[(prof, a, c)][b].append(rid)
        PAIR_HINT_INDEX[(prof, b, c)][a].append(rid)

for k in list(RECIPE_BASE_INDEX.keys()):
    RECIPE_BASE_INDEX[k].sort(key=lambda rr: safe_int(rr.get("tier", 1), 1))

# For vendor: map crafting profession -> mats used in its recipes (exact and base)
CRAFT_PROF_TO_MATS: Dict[str, set] = defaultdict(set)
CRAFT_PROF_TO_MATS_BASE: Dict[str, set] = defaultdict(set)

for r in RECIPES:
    craft_prof = canon_prof(r.get("profession", "") or "")
    for c in r.get("components", []):
        if c.get("name"):
            CRAFT_PROF_TO_MATS[craft_prof].add(canon_name(c["name"]))
            CRAFT_PROF_TO_MATS_BASE[craft_prof].add(base_name(c["name"]))

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
        st.session_state.players = copy.deepcopy(PLAYERS_DEFAULT)
    if "inventories" not in st.session_state:
        st.session_state.inventories = {p["name"]: {} for p in st.session_state.players}
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []
    if "vendor_offers" not in st.session_state:
        st.session_state.vendor_offers = {}
    if "gather_results" not in st.session_state:
        st.session_state.gather_results = {}
    if "jobs" not in st.session_state:
        st.session_state.jobs = {}  # pname -> job dict

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
                "jobs": {},
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

        job = saved.get("jobs", {})
        if isinstance(job, dict):
            st.session_state.jobs[pname] = job

def save_player_now(pname: str):
    if SB is None:
        return
    pl = get_player(pname)
    state = {
        "player": {"skills": pl.get("skills", {}), "known_recipes": pl.get("known_recipes", [])},
        "inventory": st.session_state.inventories.get(pname, {}),
        "jobs": st.session_state.jobs.get(pname, {}),
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
        "prev_jobs": copy.deepcopy(st.session_state.jobs.get(player_name)),
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
    st.session_state.jobs[pname] = last.get("prev_jobs") or {}
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

def crafting_xp_from_components(recipe: Dict[str, Any]) -> int:
    tiers = [int(ITEM_TIER.get(canon_name(c.get("name", "")), 1)) for c in recipe.get("components", [])]
    return max(tiers) if tiers else safe_int(recipe.get("tier", 1), 1)

def gathering_xp_for_item(item_name: str) -> int:
    return int(ITEM_TIER.get(canon_name(item_name), 1))

# -----------------------------
# Jobs (timers)
# -----------------------------
def start_job(pname: str, job: Dict[str, Any]):
    st.session_state.jobs[pname] = job
    save_player_now(pname)

def clear_job(pname: str):
    st.session_state.jobs[pname] = {}
    save_player_now(pname)

def job_remaining(job: Dict[str, Any]) -> int:
    end_ts = float(job.get("end_ts", 0) or 0)
    return max(0, int(math.ceil(end_ts - time.time())))

def fmt_seconds(sec: int) -> str:
    if sec <= 0:
        return "0s"
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def complete_job_if_ready(pname: str):
    job = st.session_state.jobs.get(pname) or {}
    if not job:
        return
    if job_remaining(job) > 0:
        return

    payload = job.get("payload") or {}
    kind = job.get("kind")

    player = get_player(pname)
    inv = st.session_state.inventories[pname]

    if kind == "craft":
        # components were already consumed at start
        if payload.get("success"):
            out_name = payload.get("output_name", "")
            add_item(inv, out_name, 1)
            xp_gain = int(payload.get("xp_gain", 0))
            if xp_gain > 0:
                apply_xp_delta(player, payload.get("skill", ""), xp_gain)
            st.success(f"✅ Craft completed: **{out_name}**")
        else:
            st.error("❌ Crafting failed. Materials were consumed.")
        clear_job(pname)
        st.rerun()

    if kind == "discover":
        if payload.get("success"):
            rid = payload.get("recipe_id")
            if rid:
                player.setdefault("known_recipes", [])
                if rid not in player["known_recipes"]:
                    player["known_recipes"].append(rid)
            out_name = payload.get("output_name", "")
            add_item(inv, out_name, 1)
            xp_gain = int(payload.get("xp_gain", 0))
            if xp_gain > 0:
                apply_xp_delta(player, payload.get("skill", ""), xp_gain)
            st.success(f"✅ Discovery completed: **{out_name}** (recipe learned)")
        else:
            st.error("❌ Discovery failed. Materials were consumed.")
        clear_job(pname)
        st.rerun()

# -----------------------------
# Vendor
# -----------------------------
def vendor_allowed_items_for_prof(selected_prof: str) -> List[Dict[str, Any]]:
    selected_prof = canon_prof(selected_prof)
    # If a player chooses a crafting profession, vendor should sell only components used in that profession's recipes.
    if selected_prof in recipes_by_prof:
        allowed_exact = CRAFT_PROF_TO_MATS.get(selected_prof, set())
        allowed_base = CRAFT_PROF_TO_MATS_BASE.get(selected_prof, set())
        pool = []
        for it in GATHERING_ITEMS:
            nm = canon_name(it.get("name", ""))
            if nm in allowed_exact or base_name(nm) in allowed_base:
                pool.append(it)
        return pool
    # Otherwise treat it as a gathering profession and sell items from that profession.
    return list(gathering_by_prof.get(selected_prof, []))

def generate_vendor_offer(player_tier: int, items_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Choose line count (0..3)
    n_lines = int(weighted_choice(VENDOR_LINECOUNT_WEIGHTS))
    if n_lines <= 0:
        return []

    # Partition pool by tier relative to player tier (cap at +2)
    by_bucket = {"T": [], "T+1": [], "T+2": []}
    for it in items_pool:
        t = safe_int(it.get("tier", 1), 1)
        if t == player_tier:
            by_bucket["T"].append(it)
        elif t == player_tier + 1:
            by_bucket["T+1"].append(it)
        elif t == player_tier + 2:
            by_bucket["T+2"].append(it)

    offer = []
    attempts = 0
    while len(offer) < min(VENDOR_MAX_LINES, n_lines) and attempts < 50:
        attempts += 1
        bucket = weighted_choice(VENDOR_TIER_WEIGHTS)
        pool = by_bucket.get(bucket, [])
        if not pool:
            continue
        it = random.choice(pool)
        qty = min(VENDOR_QTY_MAX, int(weighted_choice(VENDOR_QTY_WEIGHTS)))
        unit = effective_unit_price(it)
        offer.append({
            "name": canon_name(it.get("name", "")),
            "tier": safe_int(it.get("tier", 1), 1),
            "qty": int(qty),
            "unit_gp": float(unit),
            "total_gp": float(unit) * int(qty),
            "rarity_type": it.get("rarity_type", "Standard"),
        })

    # De-dupe by name (keep highest qty)
    merged = {}
    for line in offer:
        nm = line["name"]
        if nm not in merged:
            merged[nm] = line
        else:
            merged[nm]["qty"] = max(merged[nm]["qty"], line["qty"])
            merged[nm]["total_gp"] = merged[nm]["qty"] * merged[nm]["unit_gp"]

    return list(merged.values())[:VENDOR_MAX_LINES]

# -----------------------------
# Automated Gathering
# -----------------------------
def gathered_tier_from_roll(unlocked_tier: int, roll_total: int) -> Optional[int]:
    # User rule:
    # If Tier 1 and roll < 10 => fail (no gathering)
    if unlocked_tier == 1 and roll_total < 10:
        return None
    if roll_total >= 20:
        target = unlocked_tier + 2
    elif roll_total >= 15:
        target = unlocked_tier + 1
    elif roll_total >= 10:
        target = unlocked_tier
    else:
        target = max(1, unlocked_tier - 1)
    return max(1, min(7, target))

def choose_random_gather_item(gather_prof: str, tier: int) -> Optional[Dict[str, Any]]:
    pool = [it for it in gathering_by_prof.get(canon_prof(gather_prof), []) if safe_int(it.get("tier", 1), 1) == int(tier)]
    if not pool:
        return None
    return random.choice(pool)

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
    candidates = [r for r in candidates if safe_int(r.get("tier", 1), 1) <= max_visible]
    if not candidates:
        return None

    chosen_tiers = [int(ITEM_TIER.get(x, 1)) for x in chosen]
    tmin = min(chosen_tiers) if chosen_tiers else 1

    # If player used mixed tiers, resolve to the lowest tier they put in (your rule).
    preferred = [r for r in candidates if safe_int(r.get("tier", 1), 1) <= tmin]
    if preferred:
        return preferred[0]
    return candidates[0]

def discovery_hint_for_invalid(craft_prof: str, chosen_items: List[str]) -> Optional[Tuple[str, str]]:
    """
    If roll >= DC but recipe invalid: give a useful hint.
    Returns (pair_text, third_hint)
    """
    prof = canon_prof(craft_prof)
    bases = [base_name(x) for x in chosen_items if x]
    if len(bases) != 3:
        return None
    a, b, c = bases
    pairs = [(a, b), (a, c), (b, c)]
    random.shuffle(pairs)
    for x, y in pairs:
        k = (prof, *sorted([x, y]))
        third_map = PAIR_HINT_INDEX.get(k)
        if not third_map:
            continue
        third_opts = sorted(list(third_map.keys()))
        if not third_opts:
            continue
        third = random.choice(third_opts)
        return (f"{sorted([x,y])[0]} + {sorted([x,y])[1]}", f"Try adding something like: **{third} (any tier)**")
    return None

# -----------------------------
# Crafting roll checks
# -----------------------------
def roll_feedback_invalid(roll: int, dc: int) -> str:
    if roll < dc - 5:
        return "You’re unsure these materials belong together."
    if dc - 5 <= roll < dc:
        return "Something about this feels close, like 2 of these might belong to a real recipe."
    return "You’re confident at least two of these materials resonate, but the full combination is wrong."

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

    gathering_prof = canon_prof(player.get("gathering_profession", "") or "")
    crafting_profs = [canon_prof(x) for x in (player.get("crafting_professions", []) or [])]

    with tabs[idx]:
        # Complete timed jobs if done
        complete_job_if_ready(pname)

        st.subheader(f"👤 {pname}")

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
                    st.caption(f"Tier {unlocked_tier} • Unlock T{next_tier} at level {next_lvl} • XP {cur_xp}/{needed}")
                else:
                    st.caption(f"Tier {unlocked_tier} • XP {cur_xp}/{needed}")

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
                    with mid:
                        st.write(f"Qty: **{qty}**")
                        st.caption(f"Sell: **{sell} gp**")
                    with right:
                        bminus, bplus = st.columns(2)
                        with bminus:
                            if st.button("−", key=f"{pname}-inv-{nm}-minus"):
                                push_undo(pname, f"Inventory −1 ({nm})")
                                remove_item(inv, nm, 1)
                                save_player_now(pname)
                                st.rerun()
                        with bplus:
                            if st.button("+", key=f"{pname}-inv-{nm}-plus"):
                                push_undo(pname, f"Inventory +1 ({nm})")
                                add_item(inv, nm, 1)
                                save_player_now(pname)
                                st.rerun()

        # ---- Gathering ----
        with st.expander("⛏️ Gathering", expanded=False):
            if not gathering_prof:
                st.caption("No gathering profession.")
            else:
                skill_lvl = int(skills.get(gathering_prof, {}).get("level", 1))
                unlocked = max_tier_for_level(skill_lvl)

                st.caption(f"Profession: **{gathering_prof}** • Tier {unlocked} unlocked")
                roll_total = st.number_input("Enter your gathering roll total (d20 + modifiers)", min_value=0, max_value=60, value=0, step=1, key=f"{pname}-g-roll")

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

                # Job banner (craft/discover)
                job = st.session_state.jobs.get(pname) or {}
                if job and job.get("kind") in ("craft","discover"):
                    remain = job_remaining(job)
                    if remain > 0:
                        pct = 1.0 - (remain / float(TIER_SECONDS.get(int(job.get("tier", 1)), 60)))
                        st.info(f"⏳ {job.get('kind').title()} in progress (T{job.get('tier', 1)}). Remaining: {fmt_seconds(remain)}")
                        st.progress(max(0.0, min(pct, 1.0)))
                        st.caption("When the timer hits 0, refresh the page (or click any button) and it will complete.")
                        st.divider()

                known_ids = set(player.get("known_recipes", []))
                known_recipes = [recipes_by_id[rid] for rid in known_ids if rid in recipes_by_id and canon_prof(recipes_by_id[rid].get("profession", "")) == craft_prof]
                known_recipes = [r for r in known_recipes if safe_int(r.get("tier", 1), 1) <= unlocked + 2]

                st.markdown("#### Discover recipes (3 items from your inventory)")
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
                else:
                    d1, d2, d3 = st.columns(3)
                    m1 = d1.selectbox("Item 1", [""] + inv_options, key=f"{pname}-disc-1")
                    m2 = d2.selectbox("Item 2", [""] + inv_options, key=f"{pname}-disc-2")
                    m3 = d3.selectbox("Item 3", [""] + inv_options, key=f"{pname}-disc-3")
                    chosen = [m1, m2, m3]
                    disc_roll = st.number_input("Enter your discovery roll total (d20 + modifiers)", min_value=0, max_value=60, value=0, step=1, key=f"{pname}-disc-roll")

                    if st.button("Try combination", key=f"{pname}-disc-try", disabled=bool(job)):
                        ok, msg = validate_discovery_selection(inv, chosen)
                        if not ok:
                            st.error(msg)
                        else:
                            push_undo(pname, f"Discovery attempt ({craft_prof})")
                            consume_selected(inv, chosen)

                            recipe = match_recipe_mixed_tiers(craft_prof, chosen, unlocked)
                            # Determine target tier for DC + timer
                            chosen_tiers = [int(ITEM_TIER.get(canon_name(x), 1)) for x in chosen if x]
                            tmin = min(chosen_tiers) if chosen_tiers else unlocked
                            target_tier = safe_int(recipe.get("tier", tmin), tmin) if recipe else tmin
                            target_tier = min(7, max(1, target_tier))
                            dc = dc_for_target_tier(unlocked, target_tier) or 20

                            # If invalid recipe: feedback + maybe hint
                            if recipe is None:
                                st.error("Not a valid recipe.")
                                st.info(roll_feedback_invalid(int(disc_roll), int(dc)))
                                if int(disc_roll) >= int(dc):
                                    hint = discovery_hint_for_invalid(craft_prof, [canon_name(x) for x in chosen if x])
                                    if hint:
                                        st.warning(f"Hint: **{hint[0]}** seems to match something. {hint[1]}")
                                save_player_now(pname)
                                st.rerun()

                            # Valid recipe: roll check
                            out_name = canon_name(recipe.get("name", ""))
                            xp_gain = crafting_xp_from_components(recipe)

                            if int(disc_roll) < int(dc):
                                # FAIL: consume already done, no XP
                                st.error(f"Discovery failed (DC {dc}). You felt these materials react… but the work didn’t hold.")
                                start_job(pname, {
                                    "kind": "discover",
                                    "tier": int(target_tier),
                                    "end_ts": time.time() + float(TIER_SECONDS.get(int(target_tier), 60)),
                                    "payload": {"success": False, "skill": craft_prof}
                                })
                                st.rerun()

                            # SUCCESS: start timer; completion gives item + XP + learns recipe
                            start_job(pname, {
                                "kind": "discover",
                                "tier": int(target_tier),
                                "end_ts": time.time() + float(TIER_SECONDS.get(int(target_tier), 60)),
                                "payload": {
                                    "success": True,
                                    "skill": craft_prof,
                                    "recipe_id": recipe.get("id"),
                                    "output_name": out_name,
                                    "xp_gain": int(xp_gain),
                                }
                            })
                            st.success(f"Discovery started! If successful you will create **{out_name}**. (DC {dc})")
                            st.rerun()

                st.divider()
                st.markdown(f"#### Known recipes (visible up to T{unlocked + 2})")

                if not known_recipes:
                    st.caption("You haven’t learned any recipes yet.")
                else:
                    for t in range(1, min(8, unlocked + 3)):
                        tier_recipes = [r for r in known_recipes if safe_int(r.get("tier", 1), 1) == t]
                        if not tier_recipes:
                            continue
                        with st.expander(f"T{t} recipes", expanded=(t == unlocked)):
                            st.markdown(f"**Tier:** {tier_badge_html(unlocked, t)}", unsafe_allow_html=True)
                            tier_recipes.sort(key=lambda x: (x.get("name", "") or "").lower())
                            for r in tier_recipes:
                                nm = canon_name(r.get("name", ""))
                                can = can_craft(inv, r)
                                xp_gain = crafting_xp_from_components(r)
                                dc = dc_for_target_tier(unlocked, t)

                                st.write(f"**{nm}** ({tier_badge_html(unlocked, t)})", unsafe_allow_html=True)
                                if r.get("description"):
                                    st.caption(r["description"])
                                st.caption(f"DC {dc if dc else '—'} • XP if crafted (on success): **{xp_gain}** • Time: {fmt_seconds(TIER_SECONDS.get(t, 60))}")

                                with st.expander("Show recipe details", expanded=False):
                                    for c in r.get("components", []):
                                        cname = canon_name(c.get("name", ""))
                                        need = int(c.get("qty", 1))
                                        have = int(inv.get(cname, 0))
                                        st.write(f"- {cname}: **{have} / {need}** {'✅' if have >= need else '❌'}")
                                    if r.get("use"):
                                        st.caption(f"Use: {r['use']}")

                                craft_roll = st.number_input("Enter your crafting roll total (d20 + modifiers)", min_value=0, max_value=60, value=0, step=1, key=f"{pname}-craftroll-{r.get('id', nm)}")

                                left, right = st.columns([2, 6])
                                with left:
                                    if st.button("Craft", key=f"{pname}-craft-{r.get('id', nm)}", disabled=(not can or bool(job))):
                                        push_undo(pname, f"Craft attempt {nm}")
                                        # consume materials now
                                        for c in r.get("components", []):
                                            remove_item(inv, c.get("name", ""), int(c.get("qty", 1)))

                                        # roll check
                                        dc_val = int(dc) if dc else 20
                                        if int(craft_roll) < dc_val:
                                            # fail: no xp (rule), but time passes
                                            start_job(pname, {
                                                "kind": "craft",
                                                "tier": int(t),
                                                "end_ts": time.time() + float(TIER_SECONDS.get(int(t), 60)),
                                                "payload": {"success": False, "skill": craft_prof}
                                            })
                                            st.rerun()

                                        # success: timer then grants output + xp
                                        start_job(pname, {
                                            "kind": "craft",
                                            "tier": int(t),
                                            "end_ts": time.time() + float(TIER_SECONDS.get(int(t), 60)),
                                            "payload": {
                                                "success": True,
                                                "skill": craft_prof,
                                                "output_name": nm,
                                                "xp_gain": int(xp_gain),
                                            }
                                        })
                                        st.rerun()
                                with right:
                                    if not can:
                                        missing = []
                                        for c in r.get("components", []):
                                            cname = canon_name(c.get("name", ""))
                                            need = int(c.get("qty", 1))
                                            have = int(inv.get(cname, 0))
                                            if have < need:
                                                missing.append(f"{cname} ({have}/{need})")
                                        if missing:
                                            st.caption("Missing: " + ", ".join(missing))
                                st.divider()

        # ---- Vendor ----
        with st.expander("🧾 Vendor", expanded=False):
            prof_options = []
            if gathering_prof:
                prof_options.append(gathering_prof)
            prof_options.extend(crafting_profs)

            if not prof_options:
                st.caption("No professions available for vendors.")
            else:
                chosen_prof = st.selectbox("Choose vendor type (one of your professions)", prof_options, key=f"{pname}-vendor-prof")
                # Vendor tier is based on the chosen profession's skill level (gathering or crafting)
                skill_lvl = int(skills.get(chosen_prof, {}).get("level", 1))
                player_tier = max_tier_for_level(skill_lvl)

                st.caption(f"Vendor focuses on **{chosen_prof}** • Your tier: T{player_tier}")
                if st.button("Generate vendor offers", key=f"{pname}-vendor-roll"):
                    push_undo(pname, f"Vendor roll ({chosen_prof})")
                    pool = vendor_allowed_items_for_prof(chosen_prof)
                    offers = generate_vendor_offer(player_tier, pool)
                    st.session_state.vendor_offers[pname] = {
                        "profession": chosen_prof,
                        "tier": player_tier,
                        "offers": offers,
                        "ts": time.time(),
                    }
                    save_player_now(pname)
                    st.rerun()

                offer_state = st.session_state.vendor_offers.get(pname) or {}
                offers = offer_state.get("offers") or []
                if not offers:
                    st.caption("No vendor offers yet (or this vendor has nothing today).")
                else:
                    st.markdown("#### Offers")
                    total_gp = 0.0
                    for i, line in enumerate(offers):
                        nm = canon_name(line.get("name", ""))
                        qty = int(line.get("qty", 1))
                        unit = float(line.get("unit_gp", 0.0))
                        tot = float(line.get("total_gp", unit * qty))
                        total_gp += tot

                        left, mid, right = st.columns([5, 2, 2])
                        with left:
                            st.write(f"**{nm}** ({tier_badge_html(99, int(line.get('tier', 1)))})", unsafe_allow_html=True)
                            if ITEM_DESC.get(nm):
                                st.caption(ITEM_DESC[nm])
                        with mid:
                            st.write(f"Qty: **{qty}**")
                            st.caption(f"Unit: **{int(unit)} gp**")
                        with right:
                            st.caption(f"Total: **{int(tot)} gp**")
                            if st.button("Buy", key=f"{pname}-buy-{i}-{nm}"):
                                push_undo(pname, f"Bought {nm}")
                                add_item(inv, nm, qty)
                                save_player_now(pname)
                                st.rerun()
                    st.info(f"Total value shown: **{int(total_gp)} gp** (you can decide what they can afford in-game)")
