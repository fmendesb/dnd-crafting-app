import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
import copy
import random

import streamlit as st

st.set_page_config(page_title="D&D Crafting Simulator", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

# ---------- Load JSON ----------
@st.cache_data
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

PLAYERS_DEFAULT: List[Dict[str, Any]] = load_json(str(DATA_DIR / "players.json"))
GATHERING_ITEMS: List[Dict[str, Any]] = load_json(str(DATA_DIR / "gathering_items.json"))
RECIPES: List[Dict[str, Any]] = load_json(str(DATA_DIR / "recipes.json"))
TIER_UNLOCKS: List[Dict[str, Any]] = load_json(str(DATA_DIR / "tier_unlocks.json"))
XP_TABLE: Dict[str, int] = load_json(str(DATA_DIR / "xp_table.json"))
XP_TABLE = {int(k): int(v) for k, v in XP_TABLE.items()}

# ---------- Constants / tuning ----------
ALIASES = {"arcana extraction": "arcane extraction"}  # legacy typo
SELL_RATE = 0.5  # sell price = vendor_price * SELL_RATE

# Vendor roll parameters (simple, editable)
VENDOR_STOCK_LINES = 8
VENDOR_QTY_RANGE = (1, 5)

# ---------- Helpers ----------
def canon(s: str) -> str:
    s = (s or "").strip().lower()
    return ALIASES.get(s, s)

def title_case_prof(s: str) -> str:
    return (s or "").strip()

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
    # below: green, equal: black, +1: yellow, +2: red
    if target <= unlocked - 1:
        return "#16a34a"  # green
    if target == unlocked:
        return "#111827"  # near-black
    if target == unlocked + 1:
        return "#b45309"  # amber
    return "#b91c1c"      # red

def tier_badge(unlocked: int, target: int) -> str:
    col = tier_color(unlocked, target)
    return f'<span style="color:{col};font-weight:700;">T{target}</span>'

# ---------- Build item indices ----------
ITEM_TIER: Dict[str, int] = {}
ITEM_PROF: Dict[str, str] = {}
ITEM_DESC: Dict[str, str] = {}
ITEM_USE: Dict[str, str] = {}
ITEM_VENDOR: Dict[str, float] = {}  # for inventory sale display

for it in GATHERING_ITEMS:
    nm = it.get("name", "")
    if not nm:
        continue
    ITEM_TIER[nm] = int(it.get("tier", 1))
    ITEM_PROF[nm] = title_case_prof(it.get("profession", "") or "")
    ITEM_DESC[nm] = it.get("description", "") or ""
    ITEM_USE[nm] = it.get("use", "") or ""
    try:
        ITEM_VENDOR[nm] = float(it.get("vendor_price", 0) or 0)
    except Exception:
        ITEM_VENDOR[nm] = 0.0

# For crafted items, vendor/base may be None in current sheet export; we keep at 0 unless you fill later.
for r in RECIPES:
    nm = r.get("name","")
    if not nm:
        continue
    ITEM_TIER.setdefault(nm, int(r.get("tier", 1)))
    ITEM_DESC.setdefault(nm, r.get("description","") or "")
    ITEM_USE.setdefault(nm, r.get("use","") or "")
    try:
        ITEM_VENDOR.setdefault(nm, float(r.get("vendor_price", 0) or 0))
    except Exception:
        ITEM_VENDOR.setdefault(nm, 0.0)

# Crafting mats used per craft profession
CRAFT_PROF_TO_MATS: Dict[str, set] = defaultdict(set)
for r in RECIPES:
    craft_prof = title_case_prof(r.get("profession", "") or "")
    for c in r.get("components", []):
        if c.get("name"):
            CRAFT_PROF_TO_MATS[craft_prof].add(c["name"])

# Group gathering items by profession
gathering_by_prof = defaultdict(list)
for it in GATHERING_ITEMS:
    gathering_by_prof[title_case_prof(it.get("profession","") or "")].append(it)

# Group recipes by profession then tier
recipes_by_prof_tier = defaultdict(lambda: defaultdict(list))
recipes_by_prof = defaultdict(list)
recipes_by_id = {}
for r in RECIPES:
    prof = title_case_prof(r.get("profession","") or "")
    t = int(r.get("tier", 1))
    recipes_by_prof_tier[prof][t].append(r)
    recipes_by_prof[prof].append(r)
    rid = r.get("id") or f"{prof}|T{t}|{r.get('name','')}"
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
        # per player: {"profession": str, "lines": [ {item, tier, qty, unit, total, source_prof} ] }
        st.session_state.vendor_offers = {}

init_state()

# ---------- Mutation helpers (safe) ----------
def push_undo(player_name: str, label: str):
    st.session_state.undo_stack.append({
        "player": player_name,
        "label": label,
        "prev_inv": copy.deepcopy(st.session_state.inventories[player_name]),
        "prev_players": copy.deepcopy(st.session_state.players),
    })

def restore_from_undo():
    last = st.session_state.undo_stack.pop()
    pname = last["player"]
    st.session_state.inventories[pname] = last["prev_inv"]
    st.session_state.players = last["prev_players"]
    st.success(f"Undid: {last['label']} ({pname})")
    st.rerun()

def get_player(pname: str) -> Dict[str, Any]:
    for p in st.session_state.players:
        if p.get("name") == pname:
            return p
    raise KeyError(pname)

def apply_xp_delta(player: Dict[str, Any], skill: str, delta: int):
    """Apply +/- XP and recalc level up/down cleanly."""
    skills = player.setdefault("skills", {})
    data = skills.setdefault(skill, {"level": 1, "xp": 0})
    level = int(data.get("level", 1))
    xp = int(data.get("xp", 0))

    xp = max(0, xp + int(delta))

    # Level up
    while True:
        need = xp_to_next(level)
        if need <= 0:
            break
        if xp >= need:
            xp -= need
            level += 1
        else:
            break

    # Level down (if xp is 0 and user subtracts below 0 we already clamp)
    # Optional: allow manual level-down by subtracting XP only; not doing auto-down across levels
    # to keep it simple and avoid surprises.

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
    # XP equals the HIGHEST tier among components (simple + intuitive)
    tiers = []
    for c in recipe.get("components", []):
        tiers.append(int(ITEM_TIER.get(c.get("name",""), 1)))
    return max(tiers) if tiers else int(recipe.get("tier", 1))

def gathering_xp_for_item(item_name: str) -> int:
    return int(ITEM_TIER.get(item_name, 1))

# ---------- Vendor generation ----------
def vendor_tier_weights(unlocked_tier: int) -> Dict[int, float]:
    # Up to +2 tiers visible; higher tiers not shown
    # Most weight on unlocked, some on below, less on +1, rare on +2
    weights = {}
    for t in range(1, 8):
        if t > unlocked_tier + 2:
            continue
        if t <= unlocked_tier - 1:
            weights[t] = 0.20
        elif t == unlocked_tier:
            weights[t] = 1.00
        elif t == unlocked_tier + 1:
            weights[t] = 0.35
        elif t == unlocked_tier + 2:
            weights[t] = 0.12
    # normalize
    s = sum(weights.values()) or 1.0
    return {k: v / s for k, v in weights.items()}

def weighted_choice(weights: Dict[int, float]) -> int:
    r = random.random()
    acc = 0.0
    for k, w in sorted(weights.items()):
        acc += w
        if r <= acc:
            return k
    return sorted(weights.keys())[-1]

def generate_vendor_stock_for_prof(player: Dict[str, Any], chosen_prof: str) -> List[Dict[str, Any]]:
    """If chosen_prof is a gathering profession: sell gathering items for that profession.
       If chosen_prof is a crafting profession: sell ONLY items used as components for that profession.
    """
    chosen_prof = title_case_prof(chosen_prof)
    # Determine which skill governs tier
    # If chosen_prof is in player's skills, use that; else try to map to gathering/crafting
    skill_level = int(player.get("skills", {}).get(chosen_prof, {}).get("level", 1))
    unlocked = max_tier_for_level(skill_level)

    # Candidate item pool
    candidates = []
    if chosen_prof in gathering_by_prof:
        candidates = [it for it in gathering_by_prof[chosen_prof]]
    else:
        # crafting profession: mats used by that profession
        mats = CRAFT_PROF_TO_MATS.get(chosen_prof, set())
        candidates = [it for it in GATHERING_ITEMS if it.get("name") in mats]

    # Index by tier
    by_tier = defaultdict(list)
    for it in candidates:
        t = int(it.get("tier", 1))
        if t <= unlocked + 2:
            by_tier[t].append(it)

    weights = vendor_tier_weights(unlocked)
    lines = []
    for _ in range(VENDOR_STOCK_LINES):
        t = weighted_choice(weights)
        pool = by_tier.get(t) or []
        if not pool:
            continue
        it = random.choice(pool)
        nm = it.get("name","")
        qty = random.randint(*VENDOR_QTY_RANGE)
        unit = float(it.get("vendor_price", 0) or 0)
        total = unit * qty
        lines.append({
            "name": nm,
            "tier": int(it.get("tier", 1)),
            "qty": qty,
            "unit_price": unit,
            "total_price": total,
            "source_profession": chosen_prof,
            "unlocked_tier": unlocked,
        })
    return lines

# ---------- Recipe discovery: "2/3 correct" without revealing names ----------
def best_partial_match(player: Dict[str, Any], craft_prof: str, chosen_items: List[str]) -> Tuple[int, Optional[str]]:
    """Return (best_overlap, matching_recipe_id) for undiscovered recipes in prof."""
    craft_prof = title_case_prof(craft_prof)
    known = set(player.get("known_recipes", []))
    chosen = [x for x in chosen_items if x]
    chosen_set = set(chosen)
    if len(chosen_set) != 3:
        return (0, None)

    # limit to visible tiers: up to unlocked+2
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

# ---------- UI ----------
st.title("🛠 D&D Crafting Simulator")

# Global Undo
u1, u2 = st.columns([1, 6])
with u1:
    if st.button("↩️ Undo", disabled=(len(st.session_state.undo_stack) == 0)):
        restore_from_undo()
with u2:
    if st.session_state.undo_stack:
        st.caption(f"Last: {st.session_state.undo_stack[-1]['label']} ({st.session_state.undo_stack[-1]['player']})")
    else:
        st.caption("No actions to undo yet.")

if not st.session_state.players:
    st.info("Add players in data/players.json, then refresh.")
    st.stop()

tabs = st.tabs([p["name"] for p in st.session_state.players])

for idx, player in enumerate(st.session_state.players):
    pname = player["name"]
    inv: Dict[str, int] = st.session_state.inventories[pname]
    skills: Dict[str, Dict[str, Any]] = player.get("skills", {})

    gathering_prof = title_case_prof(player.get("gathering_profession", "") or "")
    crafting_profs = [title_case_prof(x) for x in (player.get("crafting_professions", []) or [])]

    with tabs[idx]:
        st.subheader(f"👤 {pname}")

        # ---------- Skills (with +/- XP) ----------
        st.markdown("### Skills")
        cols = st.columns(2)
        skill_names = list(skills.keys())

        for i, s in enumerate(skill_names):
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

                # XP +/- buttons
                b1, b2, b3 = st.columns([1,1,5])
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

        # ---------- Expanders (mobile-friendly) ----------
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
                if prof_filter and prof_filter != "All":
                    if ITEM_PROF.get(nm,"") != prof_filter:
                        continue
                vend = float(ITEM_VENDOR.get(nm, 0) or 0)
                sell = math.floor(vend * SELL_RATE)
                rows.append((nm, t, qty, ITEM_PROF.get(nm,""), vend, sell))

            if sort_choice == "Name":
                rows.sort(key=lambda x: x[0].lower())
            elif sort_choice == "Tier":
                rows.sort(key=lambda x: (x[1], x[0].lower()))
            else:
                rows.sort(key=lambda x: (-x[2], x[0].lower()))

            if not rows:
                st.caption("Inventory is empty.")
            else:
                for nm,t,qty,prof,vend,sell in rows:
                    left, mid, right = st.columns([5,2,2])
                    with left:
                        st.write(f"**{nm}** ({tier_badge(99, t)})")
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

                # Filters
                f1,f2 = st.columns([2,2])
                tier_pick = f1.selectbox("Show tier", ["All","≤ Unlocked+2"] + [f"T{i}" for i in range(1,8)], key=f"{pname}-g-tier")
                search = f2.text_input("Search", "", key=f"{pname}-g-search").strip().lower()

                items = gathering_by_prof.get(gathering_prof, [])
                # visibility: not show above unlocked+2
                items = [it for it in items if int(it.get("tier",1)) <= unlocked + 2]

                if tier_pick == "≤ Unlocked+2":
                    pass
                elif tier_pick.startswith("T"):
                    tv = int(tier_pick[1:])
                    items = [it for it in items if int(it.get("tier",1)) == tv]
                if search:
                    items = [it for it in items if search in (it.get("name","").lower())]

                items.sort(key=lambda x: (int(x.get("tier",1)), x.get("name","").lower()))

                for it in items:
                    nm = it.get("name","")
                    t = int(it.get("tier",1))
                    dc = dc_for_target_tier(unlocked, t)
                    xp_gain = gathering_xp_for_item(nm)
                    col = tier_color(unlocked, t)

                    st.markdown(f"**{nm}** ({tier_badge(unlocked, t)})  \n"
                                f"<span style='color:{col};'>DC {dc if dc else '—'}</span> • XP if gathered: **{xp_gain}**",
                                unsafe_allow_html=True)
                    if it.get("description"):
                        st.caption(it["description"])
                    if it.get("use"):
                        st.caption(f"Use: {it['use']}")

                    b1,b2,b3 = st.columns([2,2,6])
                    with b1:
                        if st.button("Gathered (+1)", key=f"{pname}-gather-{nm}"):
                            push_undo(pname, f"Gathered {nm}")
                            add_item(inv, nm, 1)
                            # XP only if gathered
                            apply_xp_delta(get_player(pname), gathering_prof, xp_gain)
                            st.rerun()
                    with b2:
                        if st.button("Buy (+1)", key=f"{pname}-buy-{nm}"):
                            push_undo(pname, f"Bought {nm}")
                            add_item(inv, nm, 1)
                            # No XP
                            st.rerun()
                    st.divider()

        with st.expander("🧪 Crafting", expanded=False):
            if not crafting_profs:
                st.caption("No crafting professions.")
            else:
                # Profession picker (player's current crafting profs only)
                craft_prof = st.selectbox("Choose crafting profession", crafting_profs, key=f"{pname}-craft-prof")

                skill_lvl = int(skills.get(craft_prof, {}).get("level", 1))
                unlocked = max_tier_for_level(skill_lvl)

                # Known recipes only
                known_ids = set(player.get("known_recipes", []))
                known_recipes = [recipes_by_id[rid] for rid in known_ids if rid in recipes_by_id and title_case_prof(recipes_by_id[rid].get("profession","")) == craft_prof]
                known_recipes = [r for r in known_recipes if int(r.get("tier",1)) <= unlocked + 2]  # keep same visibility logic

                # Discovery: try 3 items
                st.markdown("#### Discover recipes (3 items)")
                mats = sorted(list(CRAFT_PROF_TO_MATS.get(craft_prof, set())))
                if not mats:
                    st.caption("No components found for this profession.")
                else:
                    d1,d2,d3 = st.columns(3)
                    m1 = d1.selectbox("Item 1", [""] + mats, key=f"{pname}-disc-1")
                    m2 = d2.selectbox("Item 2", [""] + mats, key=f"{pname}-disc-2")
                    m3 = d3.selectbox("Item 3", [""] + mats, key=f"{pname}-disc-3")

                    chosen = [m1,m2,m3]
                    if st.button("Try combination", key=f"{pname}-disc-try"):
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

                st.markdown(f"#### Known recipes (visible up to T{unlocked+2})")
                if not known_recipes:
                    st.caption("You haven’t learned any recipes yet.")
                else:
                    # group by tier with colored header, only show tiers <= unlocked+2
                    for t in range(1, 8):
                        if t > unlocked + 2:
                            continue
                        tier_recipes = [r for r in known_recipes if int(r.get("tier",1)) == t]
                        if not tier_recipes:
                            continue

                        header = f"{tier_badge(unlocked, t)} recipes"
                        with st.expander(header, expanded=(t == unlocked),):
                            tier_recipes.sort(key=lambda x: x.get("name","").lower())
                            for r in tier_recipes:
                                nm = r.get("name","")
                                can = can_craft(inv, r)
                                xp_gain = crafting_xp_from_components(r)
                                dc = dc_for_target_tier(unlocked, t)

                                st.write(f"**{nm}** ({tier_badge(unlocked, t)})")
                                if r.get("description"):
                                    st.caption(r["description"])
                                st.caption(f"DC {dc if dc else '—'} • XP if crafted: **{xp_gain}**")

                                # Details: components with "have/need"
                                with st.expander("Show recipe details", expanded=False):
                                    for c in r.get("components", []):
                                        cname = c.get("name","")
                                        need = int(c.get("qty",1))
                                        have = int(inv.get(cname, 0))
                                        ok = have >= need
                                        st.write(f"- {cname}: **{have} / {need}** {'✅' if ok else '❌'}")
                                    if r.get("use"):
                                        st.caption(f"Use: {r['use']}")

                                bcol1, bcol2 = st.columns([2,6])
                                with bcol1:
                                    if st.button("Craft", key=f"{pname}-craft-{r.get('id',nm)}", disabled=(not can)):
                                        push_undo(pname, f"Crafted {nm}")
                                        # consume
                                        for c in r.get("components", []):
                                            remove_item(inv, c.get("name",""), int(c.get("qty",1)))
                                        # output: ALWAYS 1
                                        add_item(inv, nm, 1)
                                        # XP to crafting skill = highest component tier
                                        apply_xp_delta(get_player(pname), craft_prof, xp_gain)
                                        st.rerun()
                                with bcol2:
                                    if not can:
                                        # show missing quickly
                                        missing=[]
                                        for c in r.get("components", []):
                                            cname=c.get("name","")
                                            need=int(c.get("qty",1))
                                            have=int(inv.get(cname,0))
                                            if have<need:
                                                missing.append(f"{cname} ({have}/{need})")
                                        if missing:
                                            st.caption("Missing: " + ", ".join(missing))

                                st.divider()

        # ---------- Vendor ----------
        with st.expander("🧾 Vendor", expanded=False):
            # player chooses from their CURRENT professions:
            prof_options = []
            if gathering_prof:
                prof_options.append(gathering_prof)
            prof_options.extend(crafting_profs)

            if not prof_options:
                st.caption("No professions available for vendor.")
            else:
                chosen_prof = st.selectbox("Shop type (your professions)", prof_options, key=f"{pname}-vendor-prof")
                st.caption("If you pick a crafting profession, the vendor sells ONLY components used by that craft.")
                if st.button("Generate vendor stock", key=f"{pname}-vendor-roll"):
                    lines = generate_vendor_stock_for_prof(get_player(pname), chosen_prof)
                    st.session_state.vendor_offers[pname] = {"profession": chosen_prof, "lines": lines}
                    st.rerun()

                offer = st.session_state.vendor_offers.get(pname)
                if offer and offer.get("lines"):
                    st.markdown(f"**Vendor stock for:** {offer.get('profession')}")
                    for i, line in enumerate(offer["lines"]):
                        nm=line["name"]
                        t=line["tier"]
                        qty=line["qty"]
                        unit=line["unit_price"]
                        total=line["total_price"]
                        st.write(f"**{nm}** ({tier_badge(line.get('unlocked_tier',99), t)}) • Qty: **{qty}** • Unit: **{int(unit)} gp** • Total: **{int(total)} gp**")
                        b1,b2 = st.columns([2,6])
                        with b1:
                            if st.button("Buy", key=f"{pname}-vendor-buy-{i}-{nm}"):
                                push_undo(pname, f"Vendor buy {nm} x{qty}")
                                add_item(inv, nm, qty)
                                # Purchased => no XP
                                st.rerun()
                        st.divider()
                else:
                    st.caption("No vendor stock yet. Click “Generate vendor stock”.")
