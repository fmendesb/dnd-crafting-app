import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Tuple, List, Optional
import copy

import streamlit as st

st.set_page_config(page_title="D&D Crafting Simulator", layout="wide")

# ---------- Load data (JSON) ----------
@st.cache_data
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

DATA_DIR = Path(__file__).parent / "data"

PLAYERS_DEFAULT = load_json(str(DATA_DIR / "players.json"))
GATHERING_ITEMS = load_json(str(DATA_DIR / "gathering_items.json"))
RECIPES = load_json(str(DATA_DIR / "recipes.json"))
TIER_UNLOCKS = load_json(str(DATA_DIR / "tier_unlocks.json"))
XP_TABLE = load_json(str(DATA_DIR / "xp_table.json"))
XP_TABLE = {int(k): int(v) for k, v in XP_TABLE.items()}

# ---------- Name normalization / aliases ----------
ALIASES = {
    "arcana extraction": "arcane extraction",
}

def canon(s: str) -> str:
    s = (s or "").strip().lower()
    return ALIASES.get(s, s)

# ---------- Rolls / DC rules ----------
def dc_for_target_tier(unlocked_tier: int, target_tier: int) -> Optional[int]:
    if target_tier <= unlocked_tier:
        return 10
    diff = target_tier - unlocked_tier
    if diff == 1:
        return 15
    if diff == 2:
        return 20
    return None

# ---------- XP / Leveling ----------
def xp_to_next(level: int) -> int:
    if level in XP_TABLE:
        return int(XP_TABLE[level])
    return int(XP_TABLE[max(XP_TABLE.keys())])

def award_xp(skill_data: Dict[str, Any], amount: int) -> Tuple[int, int]:
    if amount <= 0:
        return (0, 0)

    skill_data["level"] = int(skill_data.get("level", 1))
    skill_data["xp"] = float(skill_data.get("xp", 0)) + float(amount)

    levels_gained = 0
    while skill_data["xp"] >= xp_to_next(skill_data["level"]):
        skill_data["xp"] -= xp_to_next(skill_data["level"])
        skill_data["level"] += 1
        levels_gained += 1

    return (levels_gained, amount)

# ---------- Tier helpers ----------
def max_tier_for_level(level: int) -> int:
    unlocked = 1
    for row in TIER_UNLOCKS:
        if level >= int(row["unlocks_at_level"]):
            unlocked = int(row["tier"])
    return unlocked

def tier_badge(unlocked: int, t: int) -> str:
    if t < unlocked:
        return f"<span style='color:#16a34a;font-weight:700;'>T{t}</span>"
    if t == unlocked:
        return f"<span style='color:#111827;font-weight:800;'>T{t}</span>"
    if t == unlocked + 1:
        return f"<span style='color:#ca8a04;font-weight:800;'>T{t}</span>"
    return f"<span style='color:#dc2626;font-weight:800;'>T{t}</span>"

# ---------- Crafting helpers ----------
def recipe_is_craftable(inv: Dict[str, int], recipe: Dict[str, Any]) -> bool:
    for c in recipe.get("components", []):
        if inv.get(c["name"], 0) < int(c.get("qty", 1)):
            return False
    return True

def consume_components(inv: Dict[str, int], recipe: Dict[str, Any]):
    for c in recipe.get("components", []):
        inv[c["name"]] -= int(c.get("qty", 1))
        if inv[c["name"]] <= 0:
            del inv[c["name"]]

def add_output(inv: Dict[str, int], recipe: Dict[str, Any]):
    inv[recipe["name"]] = inv.get(recipe["name"], 0) + 1

# ---------- XP from items ----------
ITEM_TIER = {i["name"]: int(i.get("tier", 1)) for i in GATHERING_ITEMS}
ITEM_GATHER_PROF = {i["name"]: i.get("profession", "") for i in GATHERING_ITEMS}

def crafting_xp(recipe: Dict[str, Any]) -> int:
    return sum(int(c.get("qty", 1)) * ITEM_TIER.get(c["name"], recipe.get("tier", 1))
               for c in recipe.get("components", []))

# ---------- Undo ----------
def push_undo(player, inv, skills, label):
    st.session_state.undo.append({
        "player": player,
        "inv": copy.deepcopy(inv),
        "skills": copy.deepcopy(skills),
        "label": label
    })

# ---------- Init ----------
if "players" not in st.session_state:
    st.session_state.players = copy.deepcopy(PLAYERS_DEFAULT)
if "inventories" not in st.session_state:
    st.session_state.inventories = {p["name"]: {} for p in st.session_state.players}
if "undo" not in st.session_state:
    st.session_state.undo = []

# ---------- Index ----------
gather_by_prof = defaultdict(list)
for i in GATHERING_ITEMS:
    gather_by_prof[canon(i["profession"])].append(i)

recipes_by_prof_tier = defaultdict(lambda: defaultdict(list))
for r in RECIPES:
    recipes_by_prof_tier[canon(r["profession"])][int(r.get("tier", 1))].append(r)

# ---------- UI ----------
st.title("🛠 D&D Crafting Simulator")

if st.button("↩️ Undo last action", disabled=not st.session_state.undo):
    last = st.session_state.undo.pop()
    st.session_state.inventories[last["player"]] = last["inv"]
    for p in st.session_state.players:
        if p["name"] == last["player"]:
            p["skills"] = last["skills"]
    st.rerun()

tabs = st.tabs([p["name"] for p in st.session_state.players])

for idx, player in enumerate(st.session_state.players):
    name = player["name"]
    inv = st.session_state.inventories[name]
    skills = player["skills"]

    with tabs[idx]:
        # Skills
        st.markdown("### Skills")
        for s, data in skills.items():
            lvl = data["level"]
            xp = data["xp"]
            need = xp_to_next(lvl)
            st.write(f"**{s}** — Level {lvl}")
            st.progress(min(xp / need, 1.0))
            st.caption(f"{int(xp)} / {need} XP")

        # Inventory + Gathering collapsible
        with st.expander("🎒 Inventory & ⛏️ Gathering", expanded=False):
            st.markdown("#### Inventory")
            for item, qty in inv.items():
                c = st.columns([6, 2, 2])
                c[0].write(item)
                c[1].write(f"x{qty}")
                if c[2].button("➖", key=f"{name}-{item}-minus"):
                    push_undo(name, inv, skills, f"Remove {item}")
                    inv[item] -= 1
                    if inv[item] <= 0:
                        del inv[item]
                    st.rerun()

            st.divider()
            st.markdown("#### Gathering")
            for skill in skills:
                if canon(skill) in gather_by_prof:
                    unlocked = max_tier_for_level(skills[skill]["level"])
                    for it in gather_by_prof[canon(skill)]:
                        if it["tier"] > unlocked + 2:
                            continue
                        xp = it["tier"]
                        dc = dc_for_target_tier(unlocked, it["tier"])
                        row = st.columns([4, 2, 2, 2])
                        row[0].markdown(f"{it['name']} ({tier_badge(unlocked, it['tier'])})", unsafe_allow_html=True)
                        row[1].write(f"DC {dc}")
                        row[2].write(f"+{xp} XP")
                        if row[3].button("Gather", key=f"{name}-{skill}-{it['name']}"):
                            push_undo(name, inv, skills, f"Gather {it['name']}")
                            inv[it["name"]] = inv.get(it["name"], 0) + 1
                            award_xp(skills[skill], xp)
                            st.rerun()

        # Recipes collapsible
        with st.expander("📜 Recipes", expanded=True):
            for skill in skills:
                if canon(skill) in recipes_by_prof_tier:
                    unlocked = max_tier_for_level(skills[skill]["level"])
                    st.markdown(f"### {skill}")
                    for t in range(1, unlocked + 3):
                        if t not in recipes_by_prof_tier[canon(skill)]:
                            continue
                        with st.expander(f"Tier {t} recipes", expanded=(t == unlocked)):
                            st.markdown(f"Tier: {tier_badge(unlocked, t)}", unsafe_allow_html=True)
                            for r in recipes_by_prof_tier[canon(skill)][t]:
                                can = recipe_is_craftable(inv, r)
                                xp = crafting_xp(r)
                                dc = dc_for_target_tier(unlocked, t)
                                cols = st.columns([5, 2, 2, 2])
                                cols[0].markdown(r["name"], unsafe_allow_html=True)
                                cols[1].write(f"DC {dc}")
                                cols[2].write(f"+{xp} XP")
                                if cols[3].button("Craft", disabled=not can, key=f"{name}-{r['name']}"):
                                    push_undo(name, inv, skills, f"Craft {r['name']}")
                                    consume_components(inv, r)
                                    add_output(inv, r)
                                    award_xp(skills[skill], xp)
                                    st.rerun()
