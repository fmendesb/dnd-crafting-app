import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
import copy
import random
import math

import streamlit as st

st.set_page_config(page_title="D&D Crafting Simulator", layout="wide")
DATA_DIR = Path(__file__).parent / "data"

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
SELL_RATE = 0.5  # sell price = vendor price * 50%

# Vendor:
# - offers 0..3 distinct items (never more than 3)
# - each line qty max 4
VENDOR_MAX_LINES = 3
VENDOR_QTY_MAX = 4

# Tier odds (normalized in code)
# T ~ 90%, T+1 ~ 45%, T+2 ~ 15%
VENDOR_TIER_WEIGHTS = {"T": 0.90, "T+1": 0.45, "T+2": 0.15}

# Qty odds (normalized in code) 1..4
VENDOR_QTY_WEIGHTS = {1: 0.50, 2: 0.35, 3: 0.20, 4: 0.05}

# Number of distinct lines (0..3). You wanted "can be none" too.
# (Reasonable distribution, tweak anytime.)
VENDOR_LINECOUNT_WEIGHTS = {0: 0.10, 1: 0.35, 2: 0.35, 3: 0.20}

# Fallback gp prices (if an item has 0/missing vendor_price in JSON).
# Chosen to match the pattern you’re already seeing (T2≈5, T3≈11, T5≈45).
FALLBACK_TIER_GP = {1: 2, 2: 5, 3: 11, 4: 20, 5: 45, 6: 90, 7: 180}

# -----------------------------
# Helpers
# -----------------------------
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

def canon_prof(s: str) -> str:
    return (s or "").strip()

def xp_to_next(level: int) -> int:
    return int(XP_TABLE.get(int(level), 0))

def max_tier_for_level(level: int) -> int:
    tier = 1
    for row in TIER_UNLOCKS:
        if int(level) >= int(row.get("unlocks_at_level", 1)):
            tier = max(tier, int(row.get("tier", 1)))
    return tier

def next_unlock_level_for_tier(target_tier: int) -> Optional[int]:
    """Return the unlock level for a specific tier (exact match)."""
    lvl = None
    for row in TIER_UNLOCKS:
        if int(row.get("tier", 0)) == int(target_tier):
            lvl = int(row.get("unlocks_at_level", 0))
            break
    return lvl if lvl and lvl > 0 else None

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
        return "#16a34a"  # green
    if target == unlocked:
        return "#111827"  # black-ish
    if target == unlocked + 1:
        return "#b45309"  # amber
    return "#b91c1c"      # red

def tier_badge(unlocked: int, target: int) -> str:
    col = tier_color(unlocked, target)
    return f'<span style="color:{col};font-weight:700;">T{target}</span>'

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

# Pricing: tolerate variations in key names
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
    # Fallback by tier, so Woodcrafter vendor never shows "—"
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

# For each craft prof, what mats does it use?
CRAFT_PROF_TO_MATS: Dict[str, set] = defaultdict(set)
for r in RECIPES:
    craft_prof = canon_prof(r.get("profession", "") or "")
    for c in r.get("components", []):
        if c.get("name"):
            CRAFT_PROF_TO_MATS[craft_prof].add(c["name"])

# Gathering items by profession
gathering_by_prof = defaultdict(list)
for it in GATHERING_ITEMS:
    gathering_by_prof[canon_prof(it.get("profession", "") or "")].append(it)

# Recipes indices
recipes_by_prof = defaultdict(list)
recipes_by_id = {}
for r in RECIPES:
    prof = canon_prof(r.get("profession", "") or "")
    recipes_by_prof[prof].append(r)
    rid = r.get("id") or f"{prof}|T{int(r.get('tier', 1))}|{r.get('name','')}"
    recipes_by_id[rid] = r

# -----------------------------
# State
# -----------------------------
def init_state():
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

init_state()

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
    st.success(f"Undid: {last['label']} ({pname})")
    st.rerun()

# -----------------------------
# Core mutations
# -----------------------------
def get_player(pname: str) -> Dict[str, Any]:
    for p in st.session_state.players:
        if p.get("name") == pname:
            return p
    raise KeyError(pname)

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
    inv[item_name] = inv.get(item_name, 0) + int(qty)
    if inv[item_name] <= 0:
        inv.pop(item_name, None)

def remove_item(inv: Dict[str, int], item_name: str, qty: int = 1):
    inv[item_name] = inv.get(item_name, 0) - int(qty)
    if inv[item_name] <= 0:
        inv.pop(item_name, None)

def can_craft(inv: Dict[str, int], recipe: Dict[str, Any]) -> bool:
    for c in recipe.get("components", []):
        nm = c.get("name", "")
        need = int(c.get("qty", 1))
        if inv.get(nm, 0) < need:
            return False
    return True

def crafting_xp_from_components(recipe: Dict[str, Any]) -> int:
    tiers = [int(ITEM_TIER.get(c.get("name", ""), 1)) for c in recipe.get("components", [])]
    return max(tiers) if tiers else int(recipe.get("tier", 1))

def gathering_xp_for_item(item_name: str) -> int:
    return int(ITEM_TIER.get(item_name, 1))

# -----------------------------
# Automated Gathering (roll input)
# -----------------------------
def gathered_tier_from_roll(unlocked_tier: int, roll_total: int) -> Optional[int]:
    # Your rules:
    # - If T1 and roll < 10 => FAIL
    # - Else: 20+ => T+2, 15+ => T+1, 10+ => T, <10 => T-1
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

    # How many distinct items this visit?
    n_lines = int(weighted_choice(VENDOR_LINECOUNT_WEIGHTS))
    n_lines = max(0, min(VENDOR_MAX_LINES, n_lines))
    if n_lines == 0:
        return []

    # Candidate pool:
    if chosen_prof in gathering_by_prof:
        candidates = list(gathering_by_prof[chosen_prof])
    else:
        mats = CRAFT_PROF_TO_MATS.get(chosen_prof, set())
        candidates = [it for it in GATHERING_ITEMS if it.get("name") in mats]

    # Only allow up to T+2 (hard clamp)
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
        target_tier = choose_vendor_tier(unlocked)
        if target_tier > max_allowed_tier:
            target_tier = max_allowed_tier

        pool = by_tier.get(target_tier, [])
        if not pool:
            # fallback downwards
            tt = target_tier
            while tt >= 1 and not by_tier.get(tt):
                tt -= 1
            pool = by_tier.get(tt, [])
            if not pool:
                continue
            target_tier = tt

        it = random.choice(pool)
        nm = it.get("name", "")
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
# Discovery (3 items from inventory)
# -----------------------------
def validate_discovery_selection(inv: Dict[str, int], picks: List[str]) -> Tuple[bool, str]:
    counts = defaultdict(int)
    for p in picks:
        if not p:
            return False, "Pick 3 items."
        counts[p] += 1
    for nm, need in counts.items():
        if inv.get(nm, 0) < need:
            return False, f"You selected {nm} {need}x but only have {inv.get(nm, 0)}."
    return True, ""

def best_partial_match(player: Dict[str, Any], craft_prof: str, chosen_items: List[str]) -> Tuple[int, Optional[str]]:
    craft_prof = canon_prof(craft_prof)
    known = set(player.get("known_recipes", []))
    chosen = [x for x in chosen_items if x]
    if len(chosen) != 3:
        return (0, None)
    chosen_set = set(chosen)

    lvl = int(player.get("skills", {}).get(craft_prof, {}).get("level", 1))
    unlocked = max_tier_for_level(lvl)

    best = 0
    best_id = None
    for r in recipes_by_prof.get(craft_prof, []):
        rid = r.get("id")
        if rid in known:
            continue
        t = int(r.get("tier", 1))
        if t > unlocked + 2:
            continue
        comps = [c.get("name", "") for c in r.get("components", [])]
        overlap = len(set(comps) & chosen_set)
        if overlap > best:
            best = overlap
            best_id = rid
        if best == 3:
            break
    return best, best_id

# -----------------------------
# UI
# -----------------------------
st.title("🛠 D&D Crafting Simulator")

top1, top2 = st.columns([1, 6])
with top1:
    if st.button("↩️ Undo", disabled=(len(st.session_state.undo_stack) == 0)):
        restore_from_undo()
with top2:
    st.caption(
        f"Last: {st.session_state.undo_stack[-1]['label']} ({st.session_state.undo_stack[-1]['player']})"
        if st.session_state.undo_stack else "No actions to undo yet."
    )

if not st.session_state.players:
    st.info("No players found. Add them in data/players.json.")
    st.stop()

tabs = st.tabs([p["name"] for p in st.session_state.players])

for idx, player in enumerate(st.session_state.players):
    pname = player["name"]
    inv: Dict[str, int] = st.session_state.inventories[pname]
    skills: Dict[str, Dict[str, Any]] = player.get("skills", {})

    gathering_prof = canon_prof(player.get("gathering_profession", "") or "")
    crafting_profs = [canon_prof(x) for x in (player.get("crafting_professions", []) or [])]

    with tabs[idx]:
        st.subheader(f"👤 {pname}")

        # ---- Skills with tier + next unlock level ----
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
                        st.rerun()
                with b2:
                    if st.button("+1 XP", key=f"{pname}-{s}-xpplus"):
                        push_undo(pname, f"XP +1 ({s})")
                        apply_xp_delta(get_player(pname), s, +1)
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
                t = int(ITEM_TIER.get(nm, 1))
                if tier_filter != "All" and t != int(tier_filter[1:]):
                    continue
                if prof_filter and prof_filter != "All" and ITEM_PROF.get(nm, "") != prof_filter:
                    continue
                vend = float(ITEM_VENDOR.get(nm, 0) or 0)
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
                        st.write(f"**{nm}** ({tier_badge(99, t)})", unsafe_allow_html=True)
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
                                st.rerun()
                        with bplus:
                            if st.button("+", key=f"{pname}-inv-{nm}-plus"):
                                push_undo(pname, f"Inventory +1 ({nm})")
                                add_item(inv, nm, 1)
                                st.rerun()

        # ---- Gathering (roll-based; empty until button click) ----
        with st.expander("⛏️ Gathering", expanded=False):
            if not gathering_prof:
                st.caption("No gathering profession.")
            else:
                skill_lvl = int(skills.get(gathering_prof, {}).get("level", 1))
                unlocked = max_tier_for_level(skill_lvl)

                st.markdown("#### Automated gathering (enter your in-game roll)")
                roll_total = st.number_input(
                    "Your total roll (d20 + modifiers)",
                    min_value=0, max_value=60, value=10, step=1,
                    key=f"{pname}-g-roll"
                )

                if st.button("Roll gathering", key=f"{pname}-g-roll-btn"):
                    push_undo(pname, "Gathering roll")
                    target_tier = gathered_tier_from_roll(unlocked, int(roll_total))
                    if target_tier is None:
                        st.session_state.gather_results[pname] = {"failed": True}
                    else:
                        found = choose_random_gather_item(gathering_prof, target_tier)
                        st.session_state.gather_results[pname] = {
                            "failed": False,
                            "tier": int(target_tier),
                            "item": found.get("name", "") if found else ""
                        }
                    st.rerun()

                result = st.session_state.gather_results.get(pname)
                if not result:
                    st.caption("No roll yet. Press **Roll gathering**.")
                else:
                    if result.get("failed"):
                        st.error("Gathering failed! You didn’t find anything this time.")
                    else:
                        t = int(result.get("tier", 1))
                        item_name = result.get("item", "")
                        dc = dc_for_target_tier(unlocked, t)
                        st.markdown(f"Result tier: {tier_badge(unlocked, t)} • DC {dc if dc else '—'}", unsafe_allow_html=True)

                        if not item_name:
                            st.warning("No item found for that tier (check gathering_items.json).")
                        else:
                            xp_gain = gathering_xp_for_item(item_name)
                            st.write(f"**You found:** {item_name} ({tier_badge(unlocked, t)})", unsafe_allow_html=True)
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
                known_recipes = [
                    recipes_by_id[rid] for rid in known_ids
                    if rid in recipes_by_id and canon_prof(recipes_by_id[rid].get("profession", "")) == craft_prof
                ]
                known_recipes = [r for r in known_recipes if int(r.get("tier", 1)) <= unlocked + 2]

                st.markdown("#### Discover recipes (3 items from your inventory)")
                mats_allowed = CRAFT_PROF_TO_MATS.get(craft_prof, set())
                inv_options = sorted([nm for nm in inv.keys() if nm in mats_allowed], key=lambda x: x.lower())

                if not inv_options:
                    st.caption("You don’t have any valid crafting components in your inventory for this profession yet.")
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
                            overlap, rid = best_partial_match(get_player(pname), craft_prof, chosen)
                            if overlap == 3 and rid:
                                push_undo(pname, f"Discovered recipe ({craft_prof})")
                                get_player(pname).setdefault("known_recipes", [])
                                if rid not in get_player(pname)["known_recipes"]:
                                    get_player(pname)["known_recipes"].append(rid)
                                st.success("Success! You discovered a recipe. It’s now visible in your recipe list.")
                                st.rerun()
                            elif overlap == 2:
                                st.warning("So close! 2 of 3 components match an undiscovered recipe for this profession.")
                            elif overlap == 1:
                                st.info("You feel a faint resonance, but it’s not quite right. (1/3 match)")
                            else:
                                st.error("Nope. That combination doesn’t seem to lead anywhere.")

                st.divider()
                st.markdown(f"#### Known recipes (visible up to T{unlocked + 2})")
                if not known_recipes:
                    st.caption("You haven’t learned any recipes yet.")
                else:
                    for t in range(1, 8):
                        if t > unlocked + 2:
                            continue
                        tier_recipes = [r for r in known_recipes if int(r.get("tier", 1)) == t]
                        if not tier_recipes:
                            continue

                        header = f"{tier_badge(unlocked, t)} recipes"
                        with st.expander(header, expanded=(t == unlocked)):
                            tier_recipes.sort(key=lambda x: (x.get("name", "") or "").lower())
                            for r in tier_recipes:
                                nm = r.get("name", "")
                                can = can_craft(inv, r)
                                xp_gain = crafting_xp_from_components(r)
                                dc = dc_for_target_tier(unlocked, t)

                                st.write(f"**{nm}** ({tier_badge(unlocked, t)})", unsafe_allow_html=True)
                                if r.get("description"):
                                    st.caption(r["description"])
                                st.caption(f"DC {dc if dc else '—'} • XP if crafted: **{xp_gain}**")

                                with st.expander("Show recipe details", expanded=False):
                                    for c in r.get("components", []):
                                        cname = c.get("name", "")
                                        need = int(c.get("qty", 1))
                                        have = int(inv.get(cname, 0))
                                        st.write(f"- {cname}: **{have} / {need}** {'✅' if have >= need else '❌'}")
                                    if r.get("use"):
                                        st.caption(f"Use: {r['use']}")

                                left, right = st.columns([2, 6])
                                with left:
                                    if st.button("Craft", key=f"{pname}-craft-{r.get('id', nm)}", disabled=(not can)):
                                        push_undo(pname, f"Crafted {nm}")
                                        for c in r.get("components", []):
                                            remove_item(inv, c.get("name", ""), int(c.get("qty", 1)))
                                        add_item(inv, nm, 1)
                                        apply_xp_delta(get_player(pname), craft_prof, xp_gain)
                                        st.rerun()
                                with right:
                                    if not can:
                                        missing = []
                                        for c in r.get("components", []):
                                            cname = c.get("name", "")
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
                        nm = line.get("name", "")
                        t = safe_int(line.get("tier", 1), 1)
                        qty = safe_int(line.get("qty", 1), 1)
                        unit = safe_float(line.get("unit_price", 0), 0.0)
                        total = safe_float(line.get("total_price", unit * qty), unit * qty)

                        unit_disp = f"{safe_int(unit)} gp" if unit > 0 else "—"
                        total_disp = f"{safe_int(total)} gp" if total > 0 else "—"

                        st.write(
                            f"**{nm}** ({tier_badge(line.get('unlocked_tier', 99), t)}) • "
                            f"Qty: **{qty}** • Unit: **{unit_disp}** • Total: **{total_disp}**",
                            unsafe_allow_html=True
                        )

                        b1, _ = st.columns([2, 6])
                        with b1:
                            if st.button("Buy", key=f"{pname}-vendor-buy-{i}-{nm}"):
                                push_undo(pname, f"Vendor buy {nm} x{qty}")
                                add_item(inv, nm, qty)  # buying gives NO XP
                                st.rerun()
                        st.divider()
