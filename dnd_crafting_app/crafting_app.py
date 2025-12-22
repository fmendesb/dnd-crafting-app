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

# ---------- Load JSON (no stale cache: include file mtime) ----------
@st.cache_data(show_spinner=False)
def load_json(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_fresh(rel: str):
    p = DATA_DIR / rel
    return load_json(str(p), p.stat().st_mtime)

PLAYERS_DEFAULT: List[Dict[str, Any]] = load_fresh("players.json")
GATHERING_ITEMS: List[Dict[str, Any]] = load_fresh("gathering_items.json")
RECIPES: List[Dict[str, Any]] = load_fresh("recipes.json")
TIER_UNLOCKS: List[Dict[str, Any]] = load_fresh("tier_unlocks.json")
XP_TABLE: Dict[str, int] = load_fresh("xp_table.json")
XP_TABLE = {int(k): int(v) for k, v in XP_TABLE.items()}

# ---------- Constants ----------
SELL_RATE = 0.5

VENDOR_STOCK_LINES = 3
VENDOR_QTY_MAX = 4
VENDOR_TIER_WEIGHTS = {"T": 0.90, "T+1": 0.45, "T+3": 0.15}
VENDOR_QTY_WEIGHTS = {1: 0.50, 2: 0.35, 3: 0.20, 4: 0.05}

# ---------- Helpers ----------
def title_case_prof(s: str) -> str:
    return (s or "").strip()

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

def xp_to_next(level: int) -> int:
    return int(XP_TABLE.get(int(level), 0))

def max_tier_for_level(level: int) -> int:
    tier = 1
    for row in TIER_UNLOCKS:
        if int(level) >= int(row.get("unlocks_at_level", 1)):
            tier = max(tier, int(row.get("tier", 1)))
    return tier

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

# ---------- Pricing ----------
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
    return get_price(item, PRICE_KEYS_BASE)

# ---------- Build indices ----------
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
    ITEM_PROF[nm] = title_case_prof(it.get("profession", "") or "")
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
for r in RECIPES:
    craft_prof = title_case_prof(r.get("profession", "") or "")
    for c in r.get("components", []):
        if c.get("name"):
            CRAFT_PROF_TO_MATS[craft_prof].add(c["name"])

gathering_by_prof = defaultdict(list)
for it in GATHERING_ITEMS:
    gathering_by_prof[title_case_prof(it.get("profession", "") or "")].append(it)

recipes_by_prof = defaultdict(list)
recipes_by_id = {}
for r in RECIPES:
    prof = title_case_prof(r.get("profession", "") or "")
    recipes_by_prof[prof].append(r)
    rid = r.get("id") or f"{prof}|T{int(r.get('tier', 1))}|{r.get('name','')}"
    recipes_by_id[rid] = r

# ---------- State ----------
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

# ---------- Mutations ----------
def push_undo(player_name: str, label: str):
    st.session_state.undo_stack.append({
        "player": player_name,
        "label": label,
        "prev_inv": copy.deepcopy(st.session_state.inventories[player_name]),
        "prev_players": copy.deepcopy(st.session_state.players),
        "prev_gather": copy.deepcopy(st.session_state.gather_results.get(player_name)),
    })

def restore_from_undo():
    last = st.session_state.undo_stack.pop()
    pname = last["player"]
    st.session_state.inventories[pname] = last["prev_inv"]
    st.session_state.players = last["prev_players"]
    if last.get("prev_gather") is None:
        st.session_state.gather_results.pop(pname, None)
    else:
        st.session_state.gather_results[pname] = last["prev_gather"]
    st.success(f"Undid: {last['label']} ({pname})")
    st.rerun()

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
        nm = c.get("name","")
        need = int(c.get("qty", 1))
        if inv.get(nm, 0) < need:
            return False
    return True

def crafting_xp_from_components(recipe: Dict[str, Any]) -> int:
    tiers = [int(ITEM_TIER.get(c.get("name",""), 1)) for c in recipe.get("components", [])]
    return max(tiers) if tiers else int(recipe.get("tier", 1))

def gathering_xp_for_item(item_name: str) -> int:
    return int(ITEM_TIER.get(item_name, 1))

# ---------- Vendor logic ----------
def choose_vendor_tier(unlocked: int) -> int:
    key = weighted_choice(VENDOR_TIER_WEIGHTS)
    if key == "T":
        t = unlocked
    elif key == "T+1":
        t = unlocked + 1
    else:
        t = unlocked + 3
    return max(1, min(7, t))

def choose_vendor_qty() -> int:
    q = int(weighted_choice(VENDOR_QTY_WEIGHTS))
    return max(1, min(VENDOR_QTY_MAX, q))

def generate_vendor_stock_for_prof(player: Dict[str, Any], chosen_prof: str) -> List[Dict[str, Any]]:
    chosen_prof = title_case_prof(chosen_prof)
    skills = player.get("skills", {})
    skill_level = int(skills.get(chosen_prof, {}).get("level", 1))
    unlocked = max_tier_for_level(skill_level)

    if chosen_prof in gathering_by_prof:
        candidates = list(gathering_by_prof[chosen_prof])
    else:
        mats = CRAFT_PROF_TO_MATS.get(chosen_prof, set())
        candidates = [it for it in GATHERING_ITEMS if it.get("name") in mats]

    by_tier = defaultdict(list)
    for it in candidates:
        t = int(it.get("tier", 1))
        if t <= unlocked + 3:
            by_tier[t].append(it)

    lines = []
    used_names = set()
    for _ in range(80):
        if len(lines) >= VENDOR_STOCK_LINES:
            break
        target = choose_vendor_tier(unlocked)
        pool = by_tier.get(target, [])
        if not pool:
            # fall back down
            tt = target
            while tt >= 1 and not by_tier.get(tt):
                tt -= 1
            pool = by_tier.get(tt, [])
            if not pool:
                continue

        it = random.choice(pool)
        nm = it.get("name","")
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

# ---------- Discovery logic ----------
def best_partial_match(player: Dict[str, Any], craft_prof: str, chosen_items: List[str]) -> Tuple[int, Optional[str]]:
    craft_prof = title_case_prof(craft_prof)
    known = set(player.get("known_recipes", []))
    chosen = [x for x in chosen_items if x]
    chosen_set = set(chosen)
    if len(chosen) != 3:
        return (0, None)

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
        comps = [c.get("name","") for c in r.get("components", [])]
        overlap = len(set(comps) & chosen_set)
        if overlap > best:
            best = overlap
            best_id = rid
        if best == 3:
            break
    return best, best_id

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

# ---------- Automated gathering ----------
def gathered_tier_from_roll(unlocked_tier: int, roll_total: int) -> Optional[int]:
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
    pool = [it for it in gathering_by_prof.get(title_case_prof(prof), []) if int(it.get("tier", 1)) == tier]
    return random.choice(pool) if pool else None

# ---------- UI ----------
st.title("🛠 D&D Crafting Simulator")

u1, u2 = st.columns([1, 6])
with u1:
    if st.button("↩️ Undo", disabled=(len(st.session_state.undo_stack) == 0)):
        restore_from_undo()
with u2:
    st.caption(
        f"Last: {st.session_state.undo_stack[-1]['label']} ({st.session_state.undo_stack[-1]['player']})"
        if st.session_state.undo_stack else "No actions to undo yet."
    )

tabs = st.tabs([p["name"] for p in st.session_state.players])

for idx, player in enumerate(st.session_state.players):
    pname = player["name"]
    inv: Dict[str, int] = st.session_state.inventories[pname]
    skills: Dict[str, Dict[str, Any]] = player.get("skills", {})

    gathering_prof = title_case_prof(player.get("gathering_profession", "") or "")
    crafting_profs = [title_case_prof(x) for x in (player.get("crafting_professions", []) or [])]

    with tabs[idx]:
        st.subheader(f"👤 {pname}")

        st.markdown("### Skills")
        cols = st.columns(2)
        for i, s in enumerate(list(skills.keys())):
            data = skills[s]
            level = int(data.get("level", 1))
            cur_xp = int(data.get("xp", 0))
            needed = xp_to_next(level)
            unlocked_tier = max_tier_for_level(level)
            progress = 0.0 if needed <= 0 else min(cur_xp / needed, 1.0)

            with cols[i % 2]:
                st.write(f"**{s}**")
                st.progress(progress)
                st.caption(f"Lvl {level} • {cur_xp}/{needed} XP • Unlock T{unlocked_tier}")
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

        with st.expander("🎒 Inventory", expanded=False):
            c1, c2, c3 = st.columns([2,2,2])
            sort_choice = c1.selectbox("Sort", ["Name","Tier","Quantity"], key=f"{pname}-inv-sort")
            filter_mode = c2.selectbox("Filter", ["All","By profession"], key=f"{pname}-inv-filter-mode")
            tier_filter = c3.selectbox("Tier", ["All"] + [f"T{i}" for i in range(1,8)], key=f"{pname}-inv-tier-filter")

            prof_filter = None
            if filter_mode == "By profession":
                all_profs = sorted(set(ITEM_PROF.get(nm,"") for nm in inv.keys() if ITEM_PROF.get(nm,"")))
                prof_filter = st.selectbox("Profession", ["All"] + all_profs, key=f"{pname}-inv-prof-filter")

            rows=[]
            for nm, qty in inv.items():
                t = int(ITEM_TIER.get(nm, 1))
                if tier_filter != "All" and t != int(tier_filter[1:]):
                    continue
                if prof_filter and prof_filter != "All" and ITEM_PROF.get(nm,"") != prof_filter:
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
                for nm,t,qty,sell in rows:
                    left, mid, right = st.columns([5,2,2])
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
                        st.session_state.gather_results[pname] = {"failed": False, "tier": int(target_tier), "item": found.get("name","") if found else ""}
                    st.rerun()

                result = st.session_state.gather_results.get(pname)
                if not result:
                    st.caption("No roll yet. Press **Roll gathering**.")
                else:
                    if result.get("failed"):
                        st.error("Gathering failed! You didn’t find anything this time.")
                    else:
                        t = int(result.get("tier", 1))
                        item_name = result.get("item","")
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

                            if st.button("Add to inventory (Gathered)", key=f"{pname}-g-add-found"):
                                push_undo(pname, f"Gathered {item_name}")
                                add_item(inv, item_name, 1)
                                apply_xp_delta(get_player(pname), gathering_prof, xp_gain)
                                st.session_state.gather_results.pop(pname, None)
                                st.rerun()

        with st.expander("🧪 Crafting", expanded=False):
            st.caption("Crafting section unchanged in this patch (v11). Use your existing crafting logic here.")
            # If you want, paste your current Crafting block here. This patch is focused on vendor price refresh.

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
                st.caption("Per visit: max **3 items**, max **4 qty** each.")

                if st.button("Generate vendor stock", key=f"{pname}-vendor-roll"):
                    lines = generate_vendor_stock_for_prof(get_player(pname), chosen_prof)
                    st.session_state.vendor_offers[pname] = {"profession": chosen_prof, "lines": lines}
                    st.rerun()

                offer = st.session_state.vendor_offers.get(pname)
                if offer and offer.get("lines"):
                    st.markdown(f"**Vendor stock for:** {offer.get('profession')}")
                    for i, line in enumerate(offer["lines"]):
                        nm = line.get("name","")
                        t = safe_int(line.get("tier", 1), 1)
                        qty = safe_int(line.get("qty", 1), 1)
                        unit = safe_float(line.get("unit_price", 0), 0.0)
                        total = safe_float(line.get("total_price", unit * qty), unit * qty)

                        unit_disp = f"{safe_int(unit)} gp" if unit > 0 else "—"
                        total_disp = f"{safe_int(total)} gp" if total > 0 else "—"

                        st.write(
                            f"**{nm}** ({tier_badge(line.get('unlocked_tier',99), t)}) • "
                            f"Qty: **{qty}** • Unit: **{unit_disp}** • Total: **{total_disp}**",
                            unsafe_allow_html=True
                        )
                        b1, _ = st.columns([2,6])
                        with b1:
                            if st.button("Buy", key=f"{pname}-vendor-buy-{i}-{nm}"):
                                push_undo(pname, f"Vendor buy {nm} x{qty}")
                                add_item(inv, nm, qty)
                                st.rerun()
                        st.divider()
                else:
                    st.caption("No vendor stock yet. Click “Generate vendor stock”.")
