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
# Per your sheet: within tier -> DC 10, +1 tier -> DC 15, +2 tiers -> DC 20
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
    if not XP_TABLE:
        return 10
    if level in XP_TABLE:
        return int(XP_TABLE[level])
    return int(XP_TABLE[max(XP_TABLE.keys())])

def award_xp(skill_data: Dict[str, Any], amount: int) -> Tuple[int, int]:
    """Returns (levels_gained, xp_added). Assumes xp is within current level."""
    if amount <= 0:
        return (0, 0)

    skill_data["level"] = int(skill_data.get("level", 1))
    skill_data["xp"] = float(skill_data.get("xp", 0)) + float(amount)

    levels_gained = 0
    while True:
        needed = xp_to_next(int(skill_data["level"]))
        if needed <= 0:
            break
        if skill_data["xp"] >= needed:
            skill_data["xp"] -= needed
            skill_data["level"] = int(skill_data["level"]) + 1
            levels_gained += 1
        else:
            break

    return (levels_gained, amount)

# ---------- Tier unlock helpers ----------
def max_tier_for_level(level: int) -> int:
    unlocked = 1
    for row in TIER_UNLOCKS:
        if level >= int(row["unlocks_at_level"]):
            unlocked = int(row["tier"])
    return unlocked

def tier_bucket(unlocked: int, t: int) -> str:
    """Return 'below', 'current', 'plus1', 'plus2', 'hidden'."""
    if t <= unlocked - 1:
        return "below"
    if t == unlocked:
        return "current"
    if t == unlocked + 1:
        return "plus1"
    if t == unlocked + 2:
        return "plus2"
    return "hidden"

def tier_badge(unlocked: int, t: int) -> str:
    """HTML badge for tiers: below=green, current=black, +1=yellow, +2=red."""
    b = tier_bucket(unlocked, t)
    if b == "below":
        return f"<span style='color:#1a7f37;font-weight:700;'>T{t}</span>"
    if b == "current":
        return f"<span style='color:#111827;font-weight:800;'>T{t}</span>"
    if b == "plus1":
        return f"<span style='color:#b45309;font-weight:800;'>T{t}</span>"
    if b == "plus2":
        return f"<span style='color:#b91c1c;font-weight:800;'>T{t}</span>"
    return f"<span style='color:#6b7280;'>T{t}</span>"

# ---------- Crafting helpers ----------
def recipe_is_craftable(inventory: Dict[str, int], recipe: Dict[str, Any]) -> bool:
    for c in recipe.get("components", []):
        if inventory.get(c["name"], 0) < int(c.get("qty", 1)):
            return False
    return True

def consume_components(inventory: Dict[str, int], recipe: Dict[str, Any]) -> None:
    for c in recipe.get("components", []):
        nm = c["name"]
        inventory[nm] = inventory.get(nm, 0) - int(c.get("qty", 1))
        if inventory[nm] <= 0:
            inventory.pop(nm, None)

def add_output(inventory: Dict[str, int], recipe: Dict[str, Any]) -> None:
    # Always craft ONE output
    inventory[recipe["name"]] = inventory.get(recipe["name"], 0) + 1

# ---------- XP gain rules ----------
ITEM_TIER: Dict[str, int] = {}
ITEM_GATHER_PROF: Dict[str, str] = {}

for it in GATHERING_ITEMS:
    nm = it.get("name", "")
    if nm:
        ITEM_TIER[nm] = int(it.get("tier", 1))
        ITEM_GATHER_PROF[nm] = it.get("profession", "")

# Map crafting profession -> materials used (for inventory filtering)
CRAFT_PROF_TO_MATS: Dict[str, set] = defaultdict(set)
for r in RECIPES:
    craft_prof = r.get("profession", "")
    if not craft_prof:
        continue
    for c in r.get("components", []):
        CRAFT_PROF_TO_MATS[craft_prof].add(c["name"])

def crafting_xp_from_components(recipe: Dict[str, Any]) -> int:
    """XP = sum(component_qty * tier_of_component_item)."""
    total = 0
    fallback = int(recipe.get("tier", 1))
    for c in recipe.get("components", []):
        nm = c["name"]
        qty = int(c.get("qty", 1))
        tier = int(ITEM_TIER.get(nm, fallback))
        total += tier * qty
    return max(0, total)

# ---------- Undo stack ----------
def push_undo(player_name: str, prev_inv: Dict[str, int], prev_skills: Dict[str, Any], label: str):
    st.session_state.undo_stack.append({
        "player": player_name,
        "prev_inv": copy.deepcopy(prev_inv),
        "prev_skills": copy.deepcopy(prev_skills),
        "label": label,
    })

def init_state():
    if "players" not in st.session_state:
        st.session_state.players = copy.deepcopy(PLAYERS_DEFAULT)
    if "inventories" not in st.session_state:
        st.session_state.inventories = {p["name"]: {} for p in st.session_state.players}
    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []

init_state()

# ---------- Build indices ----------
gathering_professions = sorted(set(canon(x.get("profession","")) for x in GATHERING_ITEMS if x.get("profession")))
crafting_professions = sorted(set(canon(x.get("profession","")) for x in RECIPES if x.get("profession")))

gathering_by_prof = defaultdict(list)
for item in GATHERING_ITEMS:
    gathering_by_prof[canon(item.get("profession",""))].append(item)

recipes_by_prof_tier = defaultdict(lambda: defaultdict(list))
for r in RECIPES:
    recipes_by_prof_tier[canon(r.get("profession",""))][int(r.get("tier", 1))].append(r)

# ---------- UI ----------
st.title("🛠 D&D Crafting Simulator")

# Global undo button (undoes last action across all players)
undo_col1, undo_col2 = st.columns([1, 6])
with undo_col1:
    if st.button("↩️ Undo last action", disabled=(len(st.session_state.undo_stack) == 0)):
        last = st.session_state.undo_stack.pop()
        pname = last["player"]
        st.session_state.inventories[pname] = last["prev_inv"]
        for p in st.session_state.players:
            if p["name"] == pname:
                p["skills"] = last["prev_skills"]
                break
        st.success(f"Undid: {last['label']} ({pname})")
        st.rerun()
with undo_col2:
    if st.session_state.undo_stack:
        st.caption(f"Last: {st.session_state.undo_stack[-1]['label']} ({st.session_state.undo_stack[-1]['player']})")
    else:
        st.caption("No actions to undo yet.")

player_names = [p["name"] for p in st.session_state.players]
tabs = st.tabs(player_names if player_names else ["No players found"])

for idx, player in enumerate(st.session_state.players):
    name = player["name"]
    inv: Dict[str, int] = st.session_state.inventories[name]
    skills: Dict[str, Dict[str, Any]] = player.get("skills", {})
    skill_names = list(skills.keys())

    with tabs[idx]:
        st.subheader(f"👤 {name}")

        # Skills (progress bars)
        st.markdown("### Skills")
        colA, colB = st.columns(2)

        for i, s in enumerate(skill_names):
            data = skills[s]
            level = int(data.get("level", 1))
            cur_xp = float(data.get("xp", 0))
            needed = xp_to_next(level)
            progress = 0.0 if needed <= 0 else min(cur_xp / needed, 1.0)
            unlocked_tier = max_tier_for_level(level)

            with (colA if i % 2 == 0 else colB):
                st.write(f"**{s}**")
                st.progress(progress)
                st.caption(f"Level {level} — {int(cur_xp)} / {needed} XP   |   Unlocked Tier: T{unlocked_tier}")

        st.divider()

        # Inventory display + filters
        st.markdown("### 🎒 Inventory")

        inv_controls = st.columns([2, 2, 2, 4])
        sort_choice = inv_controls[0].selectbox("Sort", ["Name", "Tier", "Quantity"], key=f"{name}-inv-sort")
        filter_mode = inv_controls[1].selectbox("Filter", ["All", "Gathering profession", "Used for crafting"], key=f"{name}-inv-filter-mode")

        gather_filter = None
        craft_filter = None
        if filter_mode == "Gathering profession":
            options = ["All"] + sorted(set(ITEM_GATHER_PROF.values()))
            gather_filter = inv_controls[2].selectbox("Profession", options, key=f"{name}-inv-gath-prof")
        elif filter_mode == "Used for crafting":
            options = ["All"] + sorted(set([r.get("profession","") for r in RECIPES if r.get("profession")]))
            craft_filter = inv_controls[2].selectbox("Crafting", options, key=f"{name}-inv-craft-prof")
        else:
            inv_controls[2].write("")

        left, right = st.columns([2, 3], gap="large")

        def inv_items_filtered() -> List[Tuple[str,int]]:
            items = list(inv.items())
            if filter_mode == "Gathering profession" and gather_filter and gather_filter != "All":
                items = [(n,q) for (n,q) in items if ITEM_GATHER_PROF.get(n,"") == gather_filter]
            if filter_mode == "Used for crafting" and craft_filter and craft_filter != "All":
                allowed = CRAFT_PROF_TO_MATS.get(craft_filter, set())
                items = [(n,q) for (n,q) in items if n in allowed]
            if sort_choice == "Name":
                items.sort(key=lambda x: x[0].lower())
            elif sort_choice == "Tier":
                items.sort(key=lambda x: (ITEM_TIER.get(x[0], 999), x[0].lower()))
            else:
                items.sort(key=lambda x: (-x[1], x[0].lower()))
            return items

        with left:
            if inv:
                for item_name, qty in inv_items_filtered():
                    t = ITEM_TIER.get(item_name)
                    badge = f" (T{t})" if t is not None else ""
                    row = st.columns([6, 2, 2])
                    row[0].markdown(f"{item_name}{badge}")
                    row[1].write(f"x{qty}")
                    if row[2].button("➖", key=f"{name}-inv-minus-{item_name}"):
                        prev_inv = copy.deepcopy(inv)
                        prev_skills = copy.deepcopy(skills)
                        push_undo(name, prev_inv, prev_skills, f"Inventory -1 {item_name}")

                        inv[item_name] = max(0, inv.get(item_name, 0) - 1)
                        if inv[item_name] == 0:
                            inv.pop(item_name, None)
                        st.rerun()
            else:
                st.info("Inventory is empty. Add items on the right ➕")

        # Gathering add/remove
        with right:
            st.markdown("#### Add / Remove Gathering Items")
            player_gather_skills = [s for s in skill_names if canon(s) in gathering_professions]

            if not player_gather_skills:
                st.warning("This player has no gathering profession listed.")
            else:
                chosen = st.selectbox("Choose gathering profession", player_gather_skills, key=f"{name}-gather-choice")
                gather_data = skills[chosen]
                unlocked_tier = max_tier_for_level(int(gather_data.get("level", 1)))

                # Visible tiers: <= unlocked + 2
                vis_max = unlocked_tier + 2
                items_all = gathering_by_prof[canon(chosen)]
                items_all = [it for it in items_all if int(it.get("tier", 1)) <= vis_max]

                tier_values = sorted(set(int(x.get("tier", 1)) for x in items_all))
                tier_options = ["All"] + [f"Tier {t}" for t in tier_values]
                tier_choice = st.selectbox("Filter by tier", tier_options, key=f"{name}-tier-filter-{canon(chosen)}")

                items = items_all
                if tier_choice != "All":
                    wanted = int(tier_choice.split()[-1])
                    items = [it for it in items if int(it.get("tier", 1)) == wanted]

                items = sorted(items, key=lambda x: (int(x.get("tier", 1)), x.get("name","").lower()))

                st.caption("Colors: below tier = green, your tier = black, +1 = yellow, +2 = red (higher hidden).")

                for item in items:
                    t = int(item.get("tier", 1))
                    dc = dc_for_target_tier(unlocked_tier, t)
                    xp_gain = t  # gathering xp = tier

                    cols2 = st.columns([4, 1.2, 1.2, 1.5, 1.2, 5])
                    label = f"**{item.get('name','')}** ({tier_badge(unlocked_tier, t)})"
                    cols2[0].markdown(label, unsafe_allow_html=True)
                    cols2[1].markdown(f"DC **{dc}**" if dc is not None else "DC —")
                    cols2[2].markdown(f"+**{xp_gain}** XP")

                    if cols2[3].button("🌿 Gather", key=f"{name}-gather-{canon(chosen)}-{item.get('name','')}"):
                        prev_inv = copy.deepcopy(inv)
                        prev_skills = copy.deepcopy(skills)
                        push_undo(name, prev_inv, prev_skills, f"Gather {item['name']}")

                        inv[item["name"]] = inv.get(item["name"], 0) + 1
                        levels_gained, xp_added = award_xp(gather_data, xp_gain)
                        msg = f"Gathered {item['name']} (+{xp_added} XP)"
                        if levels_gained:
                            msg += f"  🎉 Level up! (+{levels_gained})"
                        st.success(msg)
                        st.rerun()

                    if cols2[4].button("🛒 Buy", key=f"{name}-buy-{canon(chosen)}-{item.get('name','')}"):
                        prev_inv = copy.deepcopy(inv)
                        prev_skills = copy.deepcopy(skills)
                        push_undo(name, prev_inv, prev_skills, f"Buy {item['name']}")

                        inv[item["name"]] = inv.get(item["name"], 0) + 1
                        st.rerun()

                    with cols2[5]:
                        with st.expander("Details", expanded=False):
                            if item.get("description"):
                                st.write(item["description"])
                            if item.get("use"):
                                st.write(item["use"])

        st.divider()

        # Recipes / Crafting
        st.markdown("### 📜 Recipes")
        player_craft_skills = [s for s in skill_names if canon(s) in crafting_professions]

        if not player_craft_skills:
            st.info("This player has no crafting profession listed.")
            continue

        craft_skill = st.selectbox("Choose crafting profession", player_craft_skills, key=f"{name}-craft-choice")
        craft_data = skills[craft_skill]
        craft_unlocked_tier = max_tier_for_level(int(craft_data.get("level", 1)))
        vis_max = craft_unlocked_tier + 2

        st.write(f"**{craft_skill}** shows recipes up to **Tier {vis_max}** (your unlocked tier: {craft_unlocked_tier})")
        st.caption("DC: within tier=10, +1 tier=15, +2 tiers=20. Higher tiers hidden.")

        for t in range(1, vis_max + 1):
            tier_recipes = recipes_by_prof_tier[canon(craft_skill)].get(t, [])
            if not tier_recipes:
                continue

            # IMPORTANT: expander label must be plain text (Streamlit won't render HTML there)
            with st.expander(f"Tier {t} recipes", expanded=(t == craft_unlocked_tier)):
                st.markdown(f"**Tier:** {tier_badge(craft_unlocked_tier, t)}", unsafe_allow_html=True)

                for r in sorted(tier_recipes, key=lambda x: x.get("name","").lower()):
                    can = recipe_is_craftable(inv, r)
                    recipe_tier = int(r.get("tier", 1))
                    dc = dc_for_target_tier(craft_unlocked_tier, recipe_tier)
                    xp_gain = crafting_xp_from_components(r)

                    top = st.columns([5, 1.2, 1.2, 0.8, 1.6])
                    top[0].markdown(f"**{r['name']}** ({tier_badge(craft_unlocked_tier, recipe_tier)})", unsafe_allow_html=True)
                    top[1].markdown(f"DC **{dc}**" if dc is not None else "DC —")
                    top[2].markdown(f"+**{xp_gain}** XP")
                    top[3].write("✅" if can else "❌")

                    if top[4].button("Craft", disabled=not can, key=f"{name}-craft-{canon(craft_skill)}-T{t}-{r['name']}"):
                        prev_inv = copy.deepcopy(inv)
                        prev_skills = copy.deepcopy(skills)
                        push_undo(name, prev_inv, prev_skills, f"Craft {r['name']}")

                        consume_components(inv, r)
                        add_output(inv, r)

                        levels_gained, xp_added = award_xp(craft_data, xp_gain)
                        msg = f"Crafted {r['name']} (+{xp_added} XP)"
                        if levels_gained:
                            msg += f"  🎉 Level up! (+{levels_gained})"
                        st.success(msg)
                        st.rerun()

                    with st.expander("Show recipe details", expanded=False):
                        if r.get("description"):
                            st.write(r["description"])
                        if r.get("use"):
                            st.write(r["use"])
                        st.markdown("**Components:**")
                        for c in r.get("components", []):
                            have = inv.get(c["name"], 0)
                            ctier = ITEM_TIER.get(c["name"], recipe_tier)
                            st.write(f"- {c['name']} (T{ctier}) x{int(c.get('qty',1))} | you have: **{have}**")
