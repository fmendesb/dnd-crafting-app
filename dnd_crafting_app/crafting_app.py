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
ALIASES = {"arcana extraction": "arcane extraction"}

def canon(s: str) -> str:
    s = (s or "").strip().lower()
    return ALIASES.get(s, s)

# ---------- Rolls / DC rules ----------
def dc_for_target_tier(unlocked_tier: int, target_tier: int) -> Optional[int]:
    # within tier -> 10, +1 -> 15, +2 -> 20 (higher tiers not shown)
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

# ---------- Tier helpers ----------
def max_tier_for_level(level: int) -> int:
    unlocked = 1
    for row in TIER_UNLOCKS:
        if level >= int(row["unlocks_at_level"]):
            unlocked = int(row["tier"])
    return unlocked

def tier_bucket(unlocked: int, t: int) -> str:
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
    b = tier_bucket(unlocked, t)
    if b == "below":
        return f"<span style='color:#16a34a;font-weight:700;'>T{t}</span>"
    if b == "current":
        return f"<span style='color:#111827;font-weight:800;'>T{t}</span>"
    if b == "plus1":
        return f"<span style='color:#ca8a04;font-weight:800;'>T{t}</span>"
    if b == "plus2":
        return f"<span style='color:#dc2626;font-weight:800;'>T{t}</span>"
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

# ---------- XP rules ----------
ITEM_TIER: Dict[str, int] = {}
ITEM_GATHER_PROF: Dict[str, str] = {}
ITEM_DESC: Dict[str, str] = {}
ITEM_USE: Dict[str, str] = {}

for it in GATHERING_ITEMS:
    nm = it.get("name", "")
    if nm:
        ITEM_TIER[nm] = int(it.get("tier", 1))
        ITEM_GATHER_PROF[nm] = it.get("profession", "") or ""
        ITEM_DESC[nm] = it.get("description", "") or ""
        ITEM_USE[nm] = it.get("use", "") or ""

CRAFT_PROF_TO_MATS: Dict[str, set] = defaultdict(set)
for r in RECIPES:
    craft_prof = r.get("profession", "") or ""
    for c in r.get("components", []):
        CRAFT_PROF_TO_MATS[craft_prof].add(c["name"])

def crafting_xp_from_components(recipe: Dict[str, Any]) -> int:
    total = 0
    fallback = int(recipe.get("tier", 1))
    for c in recipe.get("components", []):
        nm = c["name"]
        qty = int(c.get("qty", 1))
        tier = int(ITEM_TIER.get(nm, fallback))
        total += tier * qty
    return max(0, total)

# ---------- Undo ----------
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

# ---------- Indices ----------
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

# Undo (global)
u1, u2 = st.columns([1, 6])
with u1:
    if st.button("↩️ Undo", disabled=(len(st.session_state.undo_stack) == 0)):
        last = st.session_state.undo_stack.pop()
        pname = last["player"]
        st.session_state.inventories[pname] = last["prev_inv"]
        for p in st.session_state.players:
            if p["name"] == pname:
                p["skills"] = last["prev_skills"]
                break
        st.success(f"Undid: {last['label']} ({pname})")
        st.rerun()
with u2:
    if st.session_state.undo_stack:
        st.caption(f"Last: {st.session_state.undo_stack[-1]['label']} ({st.session_state.undo_stack[-1]['player']})")
    else:
        st.caption("No actions to undo yet.")

tabs = st.tabs([p["name"] for p in st.session_state.players])

for idx, player in enumerate(st.session_state.players):
    name = player["name"]
    inv: Dict[str, int] = st.session_state.inventories[name]
    skills: Dict[str, Dict[str, Any]] = player.get("skills", {})
    skill_names = list(skills.keys())

    with tabs[idx]:
        st.subheader(f"👤 {name}")

        # -------- Skills (compact, good on phone) --------
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
                st.caption(f"Lvl {level} • {int(cur_xp)}/{needed} XP • Unlock T{unlocked_tier}")

        # -------- Inventory (expander) --------
        with st.expander("🎒 Inventory", expanded=False):
            inv_controls = st.columns([2, 2, 2])
            sort_choice = inv_controls[0].selectbox("Sort", ["Name", "Tier", "Quantity"], key=f"{name}-inv-sort")
            filter_mode = inv_controls[1].selectbox("Filter", ["All", "Gathering profession", "Used for crafting"], key=f"{name}-inv-filter-mode")

            gather_filter = None
            craft_filter = None
            if filter_mode == "Gathering profession":
                options = ["All"] + sorted(set(ITEM_GATHER_PROF.values()))
                gather_filter = inv_controls[2].selectbox("Profession", options, key=f"{name}-inv-gath-prof")
            elif filter_mode == "Used for crafting":
                options = ["All"] + sorted(sorted(set([r.get("profession","") for r in RECIPES if r.get("profession")])))
                craft_filter = inv_controls[2].selectbox("Crafting", options, key=f"{name}-inv-craft-prof")
            else:
                inv_controls[2].write("")

            def filtered_inv_items() -> List[Tuple[str, int]]:
                items = list(inv.items())
                if filter_mode == "Gathering profession" and gather_filter and gather_filter != "All":
                    items = [(n, q) for (n, q) in items if ITEM_GATHER_PROF.get(n, "") == gather_filter]
                if filter_mode == "Used for crafting" and craft_filter and craft_filter != "All":
                    allowed = CRAFT_PROF_TO_MATS.get(craft_filter, set())
                    items = [(n, q) for (n, q) in items if n in allowed]

                if sort_choice == "Name":
                    items.sort(key=lambda x: x[0].lower())
                elif sort_choice == "Tier":
                    items.sort(key=lambda x: (ITEM_TIER.get(x[0], 999), x[0].lower()))
                else:
                    items.sort(key=lambda x: (-x[1], x[0].lower()))
                return items

            if not inv:
                st.info("Inventory is empty.")
            else:
                for item_name, qty in filtered_inv_items():
                    t = ITEM_TIER.get(item_name)
                    badge = f"(T{t})" if t is not None else ""
                    st.markdown(f"**{item_name}** {badge} • x{qty}")

                    bcols = st.columns([1, 1, 6])
                    if bcols[0].button("➖ Remove 1", key=f"{name}-inv-minus-{item_name}"):
                        prev_inv = copy.deepcopy(inv)
                        prev_skills = copy.deepcopy(skills)
                        push_undo(name, prev_inv, prev_skills, f"Inventory -1 {item_name}")

                        inv[item_name] = max(0, inv.get(item_name, 0) - 1)
                        if inv[item_name] == 0:
                            inv.pop(item_name, None)
                        st.rerun()

                    # Item details on demand (phone-friendly)
                    with bcols[2]:
                        with st.expander("Details", expanded=False):
                            d = ITEM_DESC.get(item_name, "")
                            u = ITEM_USE.get(item_name, "")
                            if d:
                                st.write(d)
                            if u:
                                st.write(u)

        # -------- Gathering (expander) --------
        with st.expander("⛏️ Gathering", expanded=False):
            player_gather_skills = [s for s in skill_names if canon(s) in gathering_professions]
            if not player_gather_skills:
                st.warning("This player has no gathering profession listed.")
            else:
                chosen = st.selectbox("Profession", player_gather_skills, key=f"{name}-gather-choice")
                gather_data = skills[chosen]
                unlocked_tier = max_tier_for_level(int(gather_data.get("level", 1)))
                vis_max = unlocked_tier + 2  # show up to +2 tiers
                items_all = [it for it in gathering_by_prof[canon(chosen)] if int(it.get("tier", 1)) <= vis_max]

                tier_values = sorted(set(int(x.get("tier", 1)) for x in items_all))
                tier_options = ["All"] + [f"Tier {t}" for t in tier_values]
                tier_choice = st.selectbox("Tier filter", tier_options, key=f"{name}-gather-tier-filter-{canon(chosen)}")

                items = items_all
                if tier_choice != "All":
                    wanted = int(tier_choice.split()[-1])
                    items = [it for it in items if int(it.get("tier", 1)) == wanted]

                items = sorted(items, key=lambda x: (int(x.get("tier", 1)), x.get("name", "").lower()))

                st.caption("Legend: below tier=green, current=black, +1=yellow, +2=red (higher hidden).")
                st.caption("🌿 Gather gives XP. 🛒 Buy gives no XP.")

                for it in items:
                    iname = it.get("name", "")
                    t = int(it.get("tier", 1))
                    dc = dc_for_target_tier(unlocked_tier, t)
                    xp_gain = t

                    # Mobile-friendly: stacked card
                    st.markdown(f"**{iname}** ({tier_badge(unlocked_tier, t)})", unsafe_allow_html=True)
                    st.caption(f"DC {dc if dc is not None else '—'} • +{xp_gain} XP")

                    action_cols = st.columns(2)
                    if action_cols[0].button("🌿 Gather", key=f"{name}-gather-{canon(chosen)}-{iname}"):
                        prev_inv = copy.deepcopy(inv)
                        prev_skills = copy.deepcopy(skills)
                        push_undo(name, prev_inv, prev_skills, f"Gather {iname}")

                        inv[iname] = inv.get(iname, 0) + 1
                        levels_gained, xp_added = award_xp(gather_data, xp_gain)
                        msg = f"Gathered {iname} (+{xp_added} XP)"
                        if levels_gained:
                            msg += f" • 🎉 Level up (+{levels_gained})"
                        st.success(msg)
                        st.rerun()

                    if action_cols[1].button("🛒 Buy", key=f"{name}-buy-{canon(chosen)}-{iname}"):
                        prev_inv = copy.deepcopy(inv)
                        prev_skills = copy.deepcopy(skills)
                        push_undo(name, prev_inv, prev_skills, f"Buy {iname}")

                        inv[iname] = inv.get(iname, 0) + 1
                        st.rerun()

                    with st.expander("Details", expanded=False):
                        if it.get("description"):
                            st.write(it["description"])
                        if it.get("use"):
                            st.write(it["use"])

                    st.divider()

        # -------- Crafting (expander) --------
        with st.expander("🛠️ Crafting", expanded=True):
            player_craft_skills = [s for s in skill_names if canon(s) in crafting_professions]
            if not player_craft_skills:
                st.info("This player has no crafting profession listed.")
            else:
                craft_skill = st.selectbox("Profession", player_craft_skills, key=f"{name}-craft-choice")
                craft_data = skills[craft_skill]
                craft_unlocked = max_tier_for_level(int(craft_data.get("level", 1)))
                vis_max = craft_unlocked + 2

                st.caption(f"Visible tiers: up to T{vis_max} (unlock T{craft_unlocked}). DC 10/15/20 based on tier difference.")

                # Optional: show only craftable recipes filter
                show_craftable_only = st.checkbox("Show craftable only", value=False, key=f"{name}-craftable-only-{canon(craft_skill)}")

                # Search
                search = st.text_input("Search recipe", value="", key=f"{name}-recipe-search-{canon(craft_skill)}").strip().lower()

                for t in range(1, vis_max + 1):
                    tier_recipes = recipes_by_prof_tier[canon(craft_skill)].get(t, [])
                    if not tier_recipes:
                        continue

                    with st.expander(f"Tier {t} recipes", expanded=(t == craft_unlocked)):
                        st.markdown(f"**Tier:** {tier_badge(craft_unlocked, t)}", unsafe_allow_html=True)

                        for r in sorted(tier_recipes, key=lambda x: x.get("name", "").lower()):
                            rname = r.get("name", "")
                            if search and search not in rname.lower():
                                continue

                            can = recipe_is_craftable(inv, r)
                            if show_craftable_only and not can:
                                continue

                            recipe_tier = int(r.get("tier", 1))
                            dc = dc_for_target_tier(craft_unlocked, recipe_tier)
                            xp_gain = crafting_xp_from_components(r)

                            st.markdown(f"**{rname}** ({tier_badge(craft_unlocked, recipe_tier)})", unsafe_allow_html=True)
                            st.caption(f"DC {dc if dc is not None else '—'} • +{xp_gain} XP • {'✅ Craftable' if can else '❌ Missing mats'}")

                            craft_btn = st.button("Craft", disabled=not can, key=f"{name}-craft-{canon(craft_skill)}-{t}-{rname}")
                            if craft_btn:
                                prev_inv = copy.deepcopy(inv)
                                prev_skills = copy.deepcopy(skills)
                                push_undo(name, prev_inv, prev_skills, f"Craft {rname}")

                                consume_components(inv, r)
                                add_output(inv, r)
                                levels_gained, xp_added = award_xp(craft_data, xp_gain)

                                msg = f"Crafted {rname} (+{xp_added} XP)"
                                if levels_gained:
                                    msg += f" • 🎉 Level up (+{levels_gained})"
                                st.success(msg)
                                st.rerun()

                            # Required items + have/missing
                            with st.expander("Show requirements", expanded=False):
                                st.markdown("**Components:**")
                                for c in r.get("components", []):
                                    cname = c["name"]
                                    need = int(c.get("qty", 1))
                                    have = int(inv.get(cname, 0))
                                    ok = have >= need
                                    icon = "✅" if ok else "❌"
                                    ctier = ITEM_TIER.get(cname, recipe_tier)
                                    st.write(f"{icon} {cname} (T{ctier}) — need {need}, have {have}")

                                # Optional recipe details
                                if r.get("description") or r.get("use"):
                                    st.divider()
                                    if r.get("description"):
                                        st.write(r["description"])
                                    if r.get("use"):
                                        st.write(r["use"])

                            st.divider()
